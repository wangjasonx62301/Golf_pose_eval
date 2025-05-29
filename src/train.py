import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from src.model import Time_Series_VAE, Golf_Pose_Classifier
from src.data import load_json_to_dataform, Keypoint_dataset
from src.loss import vae_loss
import os
import math

def eval_vae(model, config):
    
    model.eval()
    total_loss = 0

    device = torch.device(config["device"])

    json_dir = config["data"]["eval_json_dir"]
    all_jsons = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
    # print(all_jsons)
    for path in all_jsons:
        seq, label = load_json_to_dataform(path)
        if len(seq) < config["data"]["window_size"]:
            continue
        dataset = Keypoint_dataset(seq, config["data"]["window_size"], label=label)
            # print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
        
        for batch, target, _ in dataloader:
                
            batch, target = batch.to(device), target.to(device)
            predicted_next_frame, mu, logvar = model(batch)
            recon, mu, logvar = model(batch)
            loss, recon_l, kl_l, smooth_l = vae_loss(
                    predicted_next_frame, target, mu, logvar, 
                    beta=0.0000001,
                    confidence_thresh=config["training"]["confidence_threshold"],
                    lambda_smooth=config["training"].get("lambda_smooth", 0.0) # 如果平滑損失不適用，可以設為 0
            )
            total_loss += loss.item()

    print(f"Eval Loss: {total_loss:.4f}")
    

def get_lr(config, iters=None):
    # print(config['training'])
    # 1) linear warmup for config['training']['warmup_iters'] steps
    if iters < config['training']['warmup_iters']:
        return config["training"]["learning_rate"] * (iters + 1) / (config['training']['warmup_iters'] + 1)
    # 2) if iters > config['training']['lr_decay_iters'], return min learning rate
    if iters > config['training']['lr_decay_iters']:
        return config["training"]["min_lr"]
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iters - config['training']['warmup_iters']) / (config['training']['lr_decay_iters'] - config['training']['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config["training"]["min_lr"] + coeff * (config["training"]["learning_rate"] - config["training"]["min_lr"])

# may need fix
def train_vae(cfg_path=None, config=None):

    if config is None:
        assert cfg_path is not None, "cfg_path or config must be provided"
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)

        
    device = torch.device(config["device"])

    json_dir = config["data"]["json_dir"]
    
    print(json_dir)
    all_jsons = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

    model = Time_Series_VAE(config=config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(config["training"]["num_epochs"]):
        
        if epoch % config["training"]["eval_interval"] == 0:
            print(f"Evaluating VAE at epoch {epoch}...")
            eval_vae(model, config)
        
        model.train()
        total_loss, recon_total, kl_total, smooth_total = 0, 0, 0, 0
        lr = get_lr(config, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for path in all_jsons:
            seq, label = load_json_to_dataform(path)
            if len(seq) < config["data"]["window_size"]:
                continue
            if label == False:
                continue
            dataset = Keypoint_dataset(seq, config["data"]["window_size"], label=label)
            # print(len(dataset))
            dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
            
            for batch, target, _ in dataloader:
                criterion = nn.MSELoss()
                batch, target = batch.to(device), target.to(device)
                predicted_next_frame, mu, logvar = model(batch)
                # print(predicted_next_frame.shape)
                beta_start = config["training"].get("beta_start", 0.0)
                beta_max = config["training"]["beta"]
                beta_growth = config["training"].get("beta_growth", 0.01) 
                beta = min(beta_start + epoch * beta_growth, beta_max)
                # loss, recon_l, kl_l, smooth_l = vae_loss(
                #     predicted_next_frame, target, mu, logvar, 
                #     beta=beta,
                #     confidence_thresh=config["training"]["confidence_threshold"],
                #     lambda_smooth=config["training"].get("lambda_smooth", 0.0) # 如果平滑損失不適用，可以設為 0
                # )
                loss = criterion(predicted_next_frame, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # recon_total += recon_l.item()
                # kl_total += kl_l.item()
                # smooth_total += smooth_l.item()

        print(f"[Epoch {epoch+1:5d}] Total: {total_loss:.4f} | Recon: {recon_total:.4f} | KL: {kl_total:.4f} | Smooth: {smooth_total:.4f}")

        # break  # For now, just break after one epoch for testing
    ckpt_path = config["training"]["ckpt_path"]
    total_epochs = config["training"]["num_epochs"]
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    torch.save(model.state_dict(), f"{ckpt_path}Time_Series_VAE_{total_loss}_epochs_{total_epochs}.pt")
    
    return model


        ##################################
        #     Need Fix Dataset First     #
        ##################################
        
def train_classifier(vae_ckpt=None, cfg_path=None, config=None):
    if config is None:
        assert cfg_path is not None, "cfg_path or config must be provided"
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
    
    device = torch.device(config["device"])

    
    if vae_ckpt is None:
        VAE = train_vae(cfg_path)
    else: 
        VAE = Time_Series_VAE(config)    
        VAE.load_state_dict(torch.load(vae_ckpt))
        print(f"Load VAE from {vae_ckpt}")
        
    VAE.eval().to(device)
    
    model = Golf_Pose_Classifier(config=config).to(device)
    
    device = torch.device(config["device"])
    json_dir = config["data"]["json_dir"]
    all_jsons = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    # dataset = Keypoint_dataset(seq, config["data"]["window_size"])
    # dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        total_loss = 0
        lr = get_lr(config, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for path in all_jsons:
            seq, label = load_json_to_dataform(path)
            if len(seq) < config["data"]["window_size"]:
                continue
            dataset = Keypoint_dataset(seq, config["data"]["window_size"], label=label)
            # print(len(dataset))
            # print(f'current label: {label}')
            dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
            for x_batch, label in dataloader:
                x_batch, label = x_batch.to(device), label.to(device)
                with torch.no_grad():
                    mu, _ = VAE.encode(x_batch)
                    
                logits = model(mu)
                # print(logits, label)
                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        print(f"[Epoch {epoch+1:5d}] Loss: {total_loss:.8f}")
        # break  # For now, just break after one epoch for testing

    ckpt_path = config["training"]["ckpt_path"]
    total_epochs = config["training"]["num_epochs"]
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    torch.save(model.state_dict(), f"{ckpt_path}Classifier_{total_loss}_epochs_{total_epochs}.pt")
    return model


# model = train_vae()