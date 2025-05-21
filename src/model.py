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