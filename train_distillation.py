import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os

from data.dataset import OmniSightDataset
from models.autoencoder import UNet
from models.restormer import Restormer

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = OmniSightDataset(bdd100k_dir="data/bdd100k", real_rain_dir="data/real_rain")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 1. Load Frozen Teacher
    teacher = Restormer(in_channels=3, out_channels=3, dim=32).to(device)
    if os.path.exists('teacher_best.pt'):
        teacher.load_state_dict(torch.load('teacher_best.pt', map_location=device))
        print("Loaded teacher_best.pt")
    else:
        print("WARNING: teacher_best.pt not found. Using untrained teacher.")
    teacher.eval() # Freeze teacher
    
    # Freeze teacher parameters explicitly
    for param in teacher.parameters():
        param.requires_grad = False

    # 2. Load Untrained Student (6-channel)
    student = UNet(in_channels=6, out_channels=3).to(device)
    
    criterion = nn.MSELoss() # Or CharbonnierLoss
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            input_6ch = batch['input_6ch'].to(device)
            noisy_t = batch['noisy_t'].to(device)

            # Pass noisy target frame (T) into Teacher -> T_clean
            with torch.no_grad():
                t_clean = teacher(noisy_t)

            # Pass noisy 6-channel frame pair into Student -> Output S
            optimizer.zero_grad()
            s_out = student(input_6ch)
            
            # The Golden Rule: Calculate loss between S and T_clean
            loss = criterion(s_out, t_clean)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Distill Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Distillation Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), 'distilled_unet_best.pt')
            print(f"Saved best student model with loss {best_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
