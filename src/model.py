from math import e
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


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