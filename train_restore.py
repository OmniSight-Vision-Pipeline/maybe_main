import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

    # Dummy datasets since real data is missing. Modify paths for production.
    dataset = OmniSightDataset(bdd100k_dir="data/bdd100k", real_rain_dir="data/real_rain")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = Restormer(in_channels=3, out_channels=3, dim=32).to(device)
    criterion = CharbonnierLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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

        # Basic plateau saving based on training loss for this scaffold
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'teacher_best.pt')
            print(f"Saved best model with loss {best_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()
    main(args)
