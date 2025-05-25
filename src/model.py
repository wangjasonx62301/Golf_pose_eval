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
    
    def decode(self, z, seq_len):
        h0 = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_rnn(h0)
        return self.output_layer(out)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar
    
    def predict_future_frames(self, x_init, n_future):
        preds = []
        current_seq = x_init.clone()  # (1, T, 51)
        for _ in range(n_future):
            with torch.no_grad():
                mu, logvar = self.encode(current_seq)
                z = self.reparameterize(mu, logvar)
                h = self.decoder_input(z).unsqueeze(1)
                out, _ = self.decoder_rnn(h)
                pred = self.output_layer(out).squeeze(1)  # (1, 51)
            preds.append(pred.squeeze(0).cpu())
            current_seq = torch.cat([current_seq[:, 1:], pred.unsqueeze(1)], dim=1)
        return torch.stack(preds)  # (n_future, 51)
    
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
        
        
    