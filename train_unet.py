import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from datetime import datetime

from data.dataset import OmniSightDataset
from models.autoencoder import UNet

def main(args):
    # 1. Standard Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hardware Check: Device set to {device}")

    # 2. Dataset Setup
    dataset = OmniSightDataset(bdd100k_dir=args.data_dir, real_rain_dir="")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset loaded with {len(dataset)} samples.")

    # 3. Load U-Net Architecture
    model = UNet(in_channels=6, out_channels=3).to(device)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] U-Net architecture loaded.")

    # 4. Training Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_loss = float('inf')

    # 5. Training Loop
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Standalone U-Net Training...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            input_6ch = batch['input_6ch'].to(device)
            clean_t = batch['clean_t'].to(device)  # Using direct ground truth, not a teacher!

            # --- TRAINING ---
            optimizer.zero_grad(set_to_none=True)
            
            output = model(input_6ch)
            loss = criterion(output, clean_t)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.5f}")

        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'unet_scratch_best.pt')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] * New Best Model Saved (Loss: {best_loss:.5f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default=r'D:\soft computing data\bdd100k\bdd100k\images\10k\train')
    main(parser.parse_args())
