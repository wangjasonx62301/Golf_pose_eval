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
    