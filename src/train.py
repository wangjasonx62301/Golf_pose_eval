from pydoc import text
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from src.model import KeypointTransformer, Time_Series_VAE, Golf_Pose_Classifier, AdviceTransformer
from src.data import *
from src.loss import vae_loss
import os
import math
from tqdm import tqdm

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
    

def get_lr(config, iters=None, mode=None):
    if mode == "advice_decoder":
        config['training']['learning_rate'] = config['advice_model']['learning_rate']
        config['training']['warmup_iters'] = config['advice_model']['warmup_iters']
        config['training']['lr_decay_iters'] = config['advice_model']['lr_decay_iters']
        config['training']['min_lr'] = config['advice_model']['min_lr']
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

def eval_transformer(model, config=None):
    model.eval()
    device = torch.device(config["device"])
    criterion = nn.MSELoss()
    data_loader = DataLoader(
        MultiJSONKeypointDataset(config["data"]["eval_json_dir"], config["data"]["window_size"]),
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    pbar = tqdm(data_loader, desc="Evaluating Transformer")
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            # print(f"Output shape: {output.shape}, Target shape: {batch_y.shape}")

            # loss = criterion(output, batch_y)
            mask = (batch_y != 0.0).float()
            loss = ((output - batch_y) ** 2 * mask).sum() 

            
            total_loss += loss.item() * batch_x.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    print(f"Eval Loss: {avg_loss:.4f}, Total Loss: {total_loss:.4f}")
    return avg_loss

def train_transformer(ckpt=None, cfg_path=None, config=None, mode=1):
    if config is None:
        assert cfg_path is not None, "cfg_path or config must be provided"
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
    
    device = torch.device(config["device"])

    model = KeypointTransformer(config=config).to(device)
    
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
        print(f"Load Transformer from {ckpt}")
    else:
        print("Training Transformer from scratch")
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    best_loss = float('inf')
    model.train()
    
    data_loader = DataLoader(
        MultiJSONKeypointDataset(config["data"]["json_dir"], config["data"]["window_size"]),
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    for epoch in range(config["training"]["num_epochs"]):
        
        total_loss = 0
        
        lr = get_lr(config, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if (epoch % config["training"]["eval_interval"] == 0 and epoch > 0) or epoch == config["training"]["num_epochs"] - 1:
            print(f"Evaluating Transformer at epoch {epoch}...")
            eval_loss = eval_transformer(model, config)
            if eval_loss < best_loss:
                best_loss = eval_loss
                ckpt_path = config["training"]["ckpt_path"]
                total_epochs = config["training"]["num_epochs"]
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                torch.save(model.state_dict(), f"{ckpt_path}KeypointTransformer_{best_loss:.4f}_epochs_{total_epochs}_current_{epoch + 1}_NumLayers_{config['model']['num_layers']}_NumEmb_{config['model']['n_embd']}_NumHead_{config['model']['num_heads']}_Mode_{mode}.pt")
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)

            # loss = criterion(output, batch_y)
            mask = (batch_y != 0.0).float()
            # print(mask)
            if mask.sum() == 0:
                print("Warning: No valid data in batch, skipping loss calculation.")
                ones_mask = torch.ones_like(batch_y, device=device)  # Create a ones mask
                mask = ones_mask - mask
                # loss = torch.tensor(0.0, device=device)
                loss = criterion(output, batch_y)
            else:
                loss = ((output - batch_y) ** 2 * mask).sum() 


            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)  # Accumulate loss
            pbar.set_description(f"Epoch {epoch+1}/{config['training']['num_epochs']} - Loss: {loss.item():.4f}")
        # Average loss for the epoch
        pbar.close()
        avg_loss = total_loss / len(data_loader.dataset)
        print(f"[Epoch {epoch+1:5d}] Loss: {total_loss:.8f} | Avg Loss: {avg_loss:.4f}")
        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #     ckpt_path = config["training"]["ckpt_path"]
        #     total_epochs = config["training"]["num_epochs"]
        #     if not os.path.exists(ckpt_path):
        #         os.makedirs(ckpt_path)
    torch.save(model.state_dict(), f"{ckpt_path}KeypointTransformer_{best_loss:.4f}_epochs_{total_epochs}_current_{epoch + 1}_NumLayers_{config['model']['num_layers']}_NumEmb_{config['model']['n_embd']}_NumHead_{config['model']['num_heads']}_Mode_{mode}.pt")
    return model

def eval_transformer_AR(model, config=None):
    model.eval()
    device = torch.device(config["device"])
    criterion = nn.MSELoss()
    data_loader = DataLoader(
        AutoRegressiveKeypointDataset(config["data"]["json_dir"]),
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    pbar = tqdm(data_loader, desc="Evaluating Transformer")
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_mask, batch_y in pbar:
            batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
            output = model(batch_x, mask=batch_mask) 
            loss = criterion(output, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    print(f"Eval Loss: {avg_loss:.4f}, Total Loss: {total_loss:.4f}")
    return avg_loss

def train_transformer_AR(ckpt=None, cfg_path=None, config=None, mode=None):
   

    if config is None:
        assert cfg_path is not None, "cfg_path or config must be provided"
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)

    device = torch.device(config["device"])
    model = KeypointTransformer(config=config).to(device)

    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
        print(f"Loaded Transformer from {ckpt}")
    else:
        print("Training Transformer from scratch")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    best_loss = float('inf')
    model.train()

    dataset = AutoRegressiveKeypointDataset(
        json_paths=config["data"]["json_dir"],
    )
    data_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    for epoch in range(config["training"]["num_epochs"]):
        total_loss = 0

        lr = get_lr(config, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if (epoch % config["training"]["eval_interval"] == 0 and epoch > 0) or epoch == config["training"]["num_epochs"] - 1:
            print(f"Evaluating Transformer at epoch {epoch}...")
            eval_loss = eval_transformer_AR(model, config)
            if eval_loss < best_loss:
                best_loss = eval_loss
                ckpt_path = config["training"]["ckpt_path"]
                total_epochs = config["training"]["num_epochs"]
                os.makedirs(ckpt_path, exist_ok=True)
                torch.save(model.state_dict(), f"{ckpt_path}KeypointTransformerAR_{best_loss:.4f}_epochs_{total_epochs}_current_{epoch + 1}_NumLayers_{config['model']['num_layers']}_NumEmb_{config['model']['n_embd']}_NumHead_{config['model']['num_heads']}_Mode_{mode}.pt")

        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        for batch_x, batch_mask, batch_y in pbar:
            batch_x = batch_x.to(device)            # (B, T, D)
            batch_mask = batch_mask.to(device)      # (B, T)
            batch_y = batch_y.to(device)            # (B, D)

            optimizer.zero_grad()
            output = model(batch_x, mask=batch_mask)  # ➕ 加上 mask
            # loss = criterion(output, batch_y)
            mask = (batch_y != 0.0).float()
            if mask.sum() == 0:
                loss = torch.tensor(0.0, device=device)
            else:
                loss = ((output - batch_y) ** 2 * mask).sum() / mask.sum()
                
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            pbar.set_description(f"Epoch {epoch+1}/{config['training']['num_epochs']} - Loss: {loss.item():.4f}")

        pbar.close()
        avg_loss = total_loss / len(data_loader.dataset)
        print(f"[Epoch {epoch+1:3d}] Loss: {total_loss:.6f} | Avg: {avg_loss:.4f}")

    ckpt_path = config["training"]["ckpt_path"]
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), f"{ckpt_path}KeypointTransformerAR_{best_loss:.4f}_epochs_{total_epochs}_current_{epoch + 1}_NumLayers_{config['model']['num_layers']}_NumEmb_{config['model']['n_embd']}_NumHead_{config['model']['n_heads']}_Mode_{mode}.pt")
    return model

def eval_advice_decoder(model, config=None):
    # test generation
    model.eval()
    with torch.no_grad():
        device = torch.device(config["device"])
        text = model.generate_advice(max_new_tokens=32, input_seq=None)
        print(f"Generated Advice: {text}")
    


def train_advice_decoder(ckpt=None, cfg_path=None, config=None):
    if config is None:
        assert cfg_path is not None, "cfg_path or config must be provided"
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)

    device = torch.device(config["device"])
    model = AdviceTransformer(config=config).to(device)

    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
        print(f"Loaded AdviceDecoder from {ckpt}")
    else:
        print("Training AdviceDecoder from scratch")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    text_seq = get_text_token_sequence_from_csv(config["data"]["advice_csv_path"], get_tokenizer())
    df = AdviceDataset(config=config, tokenizer=get_tokenizer(), df=text_seq)
    
    for iter in range(config['training']['max_iters']):
        
        lr = get_lr(config, iter, mode='advice_decoder')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter % config['training']['eval_interval'] == 0 and iter > 0:
            print(f"Evaluating AdviceDecoder at iteration {iter}...")
            eval_advice_decoder(model, config)
            # eval_advice_decoder(model, config)  # Implement this function if needed
        model.train()
        
        xb, yb = get_advice_batch(df=df, target_idx=None, config=config)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Iter {iter}, Loss: {loss.item():.4f}")
        # break # For now, just break after one iteration for testing
        
    # save the model
    ckpt_path = config["training"]["ckpt_path"]
    total_iters = config["training"]["max_iters"]
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    torch.save(model.state_dict(), f"{ckpt_path}AdviceDecoder_{loss.item():.4f}_iters_{total_iters}.pt")