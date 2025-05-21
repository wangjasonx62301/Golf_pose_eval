import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Time_Series_VAE
from data import load_json_to_dataform, Keypoint_dataset
from loss import vae_loss
import os

with open("../cfg/time_series_vae.yaml", "r") as f:
    config = yaml.safe_load(f)
    
device = torch.device(config["device"])

json_dir = config["data"]["json_dir"]
all_jsons = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

model = Time_Series_VAE(config=config).to(device)

optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

for epoch in range(config["training"]["num_epochs"]):
    model.train()
    total_loss = 0

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

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")