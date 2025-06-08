from math import e
from turtle import forward
from sympy import im
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data import *
from src.utils import *
import math
import regex as re

class Time_Series_VAE(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = config['model']['input_dim']
        latent_dim = config['model']['latent_dim']
        hidden_dim = config['model']['hidden_dim']
        
        RNN = nn.LSTM if config['model']['rnn_type'] == 'lstm' else nn.GRU
        
        self.encoder_rnn = RNN(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = RNN(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        
        _, h = self.encoder_rnn(x)
        h = h[0] if isinstance(h, tuple) else h
        h = torch.cat([h[-2], h[-1]], dim=1)
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        return z_mean, z_logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)
    
    def decode(self, z, seq_len, time_step: int = None):
        base_decoder_input = self.decoder_input(z).unsqueeze(1) # (B, 1, hidden_dim)

        if time_step is not None:

            time_emb = torch.tensor([float(time_step) / seq_len], device=z.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            decoder_input_at_step = base_decoder_input + time_emb # (B, 1, hidden_dim)
            out, _ = self.decoder_rnn(decoder_input_at_step) # out: (B, 1, hidden_dim)
            return self.output_layer(out) # recon: (B, 1, input_dim)
        else:
            
            time_emb = torch.arange(seq_len, device=z.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2) / seq_len
            decoder_input_seq = base_decoder_input + time_emb # (B, seq_len, hidden_dim)
            out, _ = self.decoder_rnn(decoder_input_seq) # out: (B, seq_len, hidden_dim)
            return self.output_layer(out) # recon: (B, seq_len, input_dim)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        predicted_next_frame = self.decode(z, seq_len=x.size(1), time_step=x.size(1)) 

        return predicted_next_frame, mu, logvar 
    # inference, may need fix
    def predict_future_frames(self, x_init, n_future):
        preds = []
        current_seq = x_init.clone()  # (1, window_size, input_dim)
        window_size = x_init.size(1) 
        
        for k in range(n_future): 
            with torch.no_grad():
                mu, logvar = self.encode(current_seq)
                z = self.reparameterize(mu, logvar)

                pred_next_frame_output = self.decode(z, seq_len=window_size, time_step=window_size) 
                # pred_next_frame_output: (1, 1, input_dim)
                
                preds.append(pred_next_frame_output.squeeze(1).squeeze(0).cpu()) 
                
                current_seq = torch.cat([current_seq[:, 1:], pred_next_frame_output], dim=1)


        return torch.stack(preds)


    
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config['model']['latent_dim'], config['model']['hidden_dim']),
            nn.SiLU(),
            nn.Linear(config['model']['hidden_dim'], config['model']['latent_dim'])
        )
        self.norm = nn.LayerNorm(config['model']['latent_dim'])
        
    def forward(self, x):
        logits = self.classifier(x)
        return self.norm(x + logits)
    
class Golf_Pose_Classifier(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.MLP_STACK = nn.Sequential(*[MLP(config) for _ in range(config['model']['num_mlp_iter'])])
        self.a_fn = nn.GELU()
        self.out_proj = nn.Linear(config['model']['latent_dim'], config['model']['num_classes'])
    
    def forward(self, x):
        
        x = self.MLP_STACK(x)
        logits = self.out_proj(self.a_fn(x))
        
        return logits
        
# class Transformer        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)  # x: (B, T, D)
    
class KeypointTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['model']['input_dim']
        self.positional_encoding = PositionalEncoding(config['model']['hidden_dim'])
        self.input_proj = nn.Linear(self.input_dim, config['model']['hidden_dim'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model']['hidden_dim'],
            nhead=config['model']['num_heads'],
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['model']['num_layers'])
        self.output_proj = nn.Linear(config['model']['hidden_dim'], config['model']['input_dim'])
        
    def forward(self, x, mask=None):
        # x: (B, T, D)
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)
        
        if mask is not None:
            x = self.encoder(x, src_key_padding_mask=mask)  # (T, B, hidden_dim)
        else:  
            x = self.encoder(x)
            
        x = x[-1]
        x = self.output_proj(x)
        return x

    def predict_future(self, input_seq, future_steps, device='cpu'):
        """
        input_seq: Tensor of shape (1, T, D), must be 1 sample only
        future_steps: how many future frames to predict
        return: Tensor of shape (future_steps, D)
        """
        self.eval()
        input_seq = input_seq.clone().detach().to(device)
        generated = []

        for _ in range(future_steps):
            # print(f'input_seq.shape = {input_seq.shape}')
            x = self.input_proj(input_seq)           # (1, T, hidden_dim)
            x = self.positional_encoding(x)          # (1, T, hidden_dim)
            x = x.permute(1, 0, 2)                   # (T, 1, hidden_dim)
            x = self.encoder(x)                      # (T, 1, hidden_dim)
            last = x[-1]                             # (1, hidden_dim)
            pred = self.output_proj(last)            # (1, D)
            generated.append(pred.squeeze(0))        # (D,)

            pred_reshaped = pred.unsqueeze(0)        # (1, 1, D)
            input_seq = torch.cat([input_seq[:, 1:, :], pred_reshaped], dim=1)  # (1, T, D)
            # input_seq = torch.cat([input_seq, pred_reshaped], dim=1)  # (1, T, D)


        return torch.stack(generated, dim=0)  # (future_steps, D)
    
  
# from tiktoken import encoding_for_model
  
class SinusoidalEmbedding(nn.Module):
    
    def __init__(self, block_size, n_embd):
        super().__init__()
        self.emb_wei = torch.zeros(block_size, n_embd)
        wei = torch.tensor([1 / 10000 ** (2 * j / n_embd) for j in range(n_embd)]).view(1, n_embd)
        t = torch.arange(block_size).view(block_size, 1)
        # even idx embedding
        self.emb_wei[:, ::2] = torch.sin(t * wei[:, ::2])
        self.emb_wei[:, 1::2] = torch.cos(t * wei[:, ::2])
        
        self.embedding = nn.Embedding(block_size, n_embd)
        self.embedding.weight.data = self.emb_wei  
        
    def forward(self, x):
        """
        x: Tensor of shape (B, T)
        return: Tensor of shape (B, T, n_embd)
        """
        return self.embedding(x)

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.n_embd = config['advice_model']['n_embd']
        self.n_head = config['advice_model']['n_head']
        
        self.c_attn = nn.Linear(config['advice_model']['n_embd'], config['advice_model']['n_embd'] * 3)
        self.c_proj = nn.Linear(config['advice_model']['n_embd'], config['advice_model']['n_embd'])
        
        self.register_buffer('bias', torch.tril(torch.ones(config['advice_model']['block_size'], config['advice_model']['block_size']))
                             .view(1, 1, config['advice_model']['block_size'], config['advice_model']['block_size']))
        
    def forward(self, x):
        # batch_size, Seq_len, embedding dim
        B, T, C = x.shape
        # print(x.shape)
        # after c_attn(x), the shape is B, T, n_embd * 3
        a = self.c_attn(x)
        q, k, v = a.split(self.n_embd, dim=2)
        # start view() & transpose()
        # shape after transpose (Batch_size, n_head, Seq_len, n_embd // n_head) 
        # or (B, n_head, T, C // n_head)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(2, 1)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(2, 1)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(2, 1)
        # the formula : softmax(QK^T / sqrt(embd_dim(k)))V
        # shape after q @ k : (B, n_head, T, T) 
        attn = q @ k.transpose(-2, -1) * (1 / math.sqrt(self.n_embd * 3 // self.n_head))
        # encoder
        attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        # shape after attn @ v : (B, n_head, T, C // n_head)
        y = attn @ v
        y = y.transpose(2, 1).contiguous().view(B, T, C)
        self.out = self.c_proj(y)
        return self.out   

class FeedForward(nn.Module):
    
    def __init__(self, config, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['advice_model']['n_embd'], 4 * config['advice_model']['n_embd']),
            nn.ReLU(),
            nn.Linear(4 * config['advice_model']['n_embd'], config['advice_model']['n_embd']),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        head_size = config['advice_model']['n_embd'] // config['advice_model']['n_head']
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config['advice_model']['n_embd'])
        self.ln2 = nn.LayerNorm(config['advice_model']['n_embd'])
        
        
    def forward(self, x):
        # x shape (B, T, C)
        x = x + self.sa(self.ln1(x))        # (B, T, C)
        x = x + self.ffwd(self.ln2(x))      # (B, T, C)
        return x

class AdviceTransformer(nn.Module):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.token_embedding = nn.Embedding(config['advice_model']['vocab_size'], config['advice_model']['n_embd'])
        self.positional_embedding = SinusoidalEmbedding(config['advice_model']['block_size'], config['advice_model']['n_embd'])
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['advice_model']['n_layer'])])
        self.lm_head = nn.Linear(config['advice_model']['n_embd'], config['advice_model']['vocab_size'], bias=False)
        self.device = config['data']['device']
        self.block_size = config['advice_model']['block_size']
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.positional_embedding(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, ignore_index=self.config['data']['pad_token_id'])
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        idx: (B, T) tensor of input token indices
        max_new_tokens: maximum number of new tokens to generate
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] # prevent longer than block size
            logits, loss = self.forward(idx_cond)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
            # if idx_next is end token, stop generating
            if (idx_next == self.tokenizer.eot_token).any():
                break
        return idx
    
    def generate_advice(self, max_new_tokens, input_seq=None):
        """
        input_seq: Tensor of shape (B, T)
        max_new_tokens: maximum number of new tokens to generate
        """
        if input_seq is None:
            input_seq = '<|fim_middle|>'
        advice = self.tokenizer.encode(input_seq, allowed_special={'<|fim_middle|>'})
        advice = torch.tensor(advice, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, T)
        generated_ids = self.generate(advice, max_new_tokens)
        # print(f"Generated IDs: {generated_ids}")
        generated_ids = generated_ids.squeeze(0).tolist()
        # print(f"Generated IDs after squeeze: {generated_ids}")
        decoded_text = self.tokenizer.decode(generated_ids)
        return decoded_text
        # return advice