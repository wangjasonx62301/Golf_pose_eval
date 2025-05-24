import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Time_Series_VAE, Golf_Pose_Classifier
from data import load_json_to_dataform, Keypoint_dataset
from loss import vae_loss
import os
import math

def eval_vae(model, config):
    
    model.eval()
    total_loss = 0

    device = torch.device(config["device"])

    json_dir = config["data"]["eval_json"]
    all_jsons = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

    for path in all_jsons:
        seq = load_json_to_dataform(path)
        if len(seq) < config["data"]["window_size"]:
            continue
        dataset = Keypoint_dataset(seq, config["data"]["window_size"])
        # print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
        
        for batch in dataloader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar,
                            beta=config["training"]["beta"],
                            confidence_thresh=config["training"]["confidence_threshold"])

            total_loss += loss.item()

    print(f"Eval Loss: {total_loss:.4f}")
    

def get_lr(config):
    # 1) linear warmup for config['training']['warm_iters'] steps
    if config['training']['iters'] < config['training']['warm_iters']:
        return config["training"]["learning_rate"] * (config['training']['iters'] + 1) / (config['training']['warm_iters'] + 1)
    # 2) if config['training']['iters'] > config['training']['lr_decay_iters'], return min learning rate
    if config['training']['iters'] > config['training']['lr_decay_iters']:
        return config["training"]["min_lr"]
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (config['training']['iters'] - config['training']['warm_iters']) / (config['training']['lr_decay_iters'] - config['training']['warm_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config["training"]["min_lr"] + coeff * (config["training"]["learning_rate"] - config["training"]["min_lr"])

def train_vae(cfg_path='../cfg/time_series_vae.yaml'):
    
    # ../cfg/time_series_vae.yaml
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["device"])

    json_dir = config["data"]["json_dir"]
    all_jsons = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

    model = Time_Series_VAE(config=config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        total_loss = 0
        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for path in all_jsons:
            seq = load_json_to_dataform(path)
            if len(seq) < config["data"]["window_size"]:
                continue
            dataset = Keypoint_dataset(seq, config["data"]["window_size"])
            # print(len(dataset))
            dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
            
            for batch in dataloader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss, _, _ = vae_loss(recon, batch, mu, logvar,
                                beta=config["training"]["beta"],
                                confidence_thresh=config["training"]["confidence_threshold"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        print(f"[Epoch {epoch+1:5d}] Loss: {total_loss:.4f}")
        
    torch.save(model.state_dict(), f"Time_Series_VAE_{config["training"]["num_epochs"]}.pt")
    
    return model


        ##################################
        #     Need Fix Dataset First     #
        ##################################
        
def train_classifier(vae_ckpt=None, cfg_path=None):
    
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    if vae_ckpt is None:
        VAE = train_vae(cfg_path)
    else: 
        VAE = Time_Series_VAE(config)    
        VAE.load_state_dict(vae_ckpt)
        
    VAE.eval()
    
    model = Golf_Pose_Classifier(config)
    
    device = torch.device(config["device"])
    json_dir = config["data"]["json_dir"]
    all_jsons = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    dataset = Keypoint_dataset(seq, config["data"]["window_size"])
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        total_loss = 0
        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for path in all_jsons:
            seq = load_json_to_dataform(path)
            if len(seq) < config["data"]["window_size"]:
                continue
            dataset = Keypoint_dataset(seq, config["data"]["window_size"])
            # print(len(dataset))
            dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
            for x_batch, label in dataloader:
                x_batch, label = x_batch.to(device), label.to(device)
                with torch.no_grad():
                    mu, _ = VAE.encode(x_batch)
                    
                logits = model(mu)
                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), f"Classifier_{config["training"]["num_epochs"]}.pt")
    return model
# model = train_vae()