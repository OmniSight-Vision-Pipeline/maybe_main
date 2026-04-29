import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import os

from data.dataset import OmniSightDataset
from models.restormer import Restormer

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. SETUP DUAL CITYSCAPES DATASETS ---
    dir_1 = "/home/jiit_992401210050/Documents/MyDatasets/Cityscapes/1/An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM"
    dir_2 = "/home/jiit_992401210050/Documents/MyDatasets/Cityscapes/2/An_o5cmHOsS1VbLdaKx_zfMdi0No5LUpL2htRxMwCjY_bophtOkM0-6yTKB2T2sa0yo1oP086sqiaCjmNEw5d_pofWyaE9LysYJagH8yXw_GZPzK2wfiQ9u4uAKrVcEIrkJiVuTn7JBumrA"

    print("Loading Cityscapes Dataset Part 1...")
    dataset_1 = OmniSightDataset(bdd100k_dir=dir_1, real_rain_dir="") 
    
    print("Loading Cityscapes Dataset Part 2...")
    dataset_2 = OmniSightDataset(bdd100k_dir=dir_2, real_rain_dir="")
    # Merge them together
    combined_dataset = ConcatDataset([dataset_1, dataset_2])
    print(f"Total combined Cityscapes images: {len(combined_dataset)}")

    # Update dataloader to use the combined dataset
    dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # --- 2. SETUP MODEL & LOAD V2 WEIGHTS ---
    model = Restormer(in_channels=3, out_channels=3, dim=32).to(device)
    
    print("Loading V2 Restormer weights from GitHub for Transfer Learning...")
    model.load_state_dict(torch.load('teacher_best.pt'))
    print("Weights loaded successfully!")

    criterion = CharbonnierLoss().to(device)
    
    # --- 3. FINE-TUNING OPTIMIZER ---
    # Lowered learning rate (1e-5) because we are fine-tuning pre-trained weights
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            noisy = batch['noisy_t'].to(device)
            clean = batch['clean_t'].to(device)

            optimizer.zero_grad()
            restored = model(noisy)
            loss = criterion(restored, clean)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # --- CRITICAL: SAVING WITH A NEW NAME ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'cityscapes_final.pt')
            print(f"Saved best model with loss {best_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30) # Default to 30 for fine-tuning
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()
    main(args)
