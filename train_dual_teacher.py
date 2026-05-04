import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from datetime import datetime

from data.dataset import OmniSightDataset
from models.autoencoder import UNet
from models.restormer import Restormer

def main(args):
    # 1. Standard Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hardware Check: Device set to {device}")

    # 2. Dataset Setup
    dataset = OmniSightDataset(bdd100k_dir=args.data_dir, real_rain_dir="")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset loaded with {len(dataset)} samples.")

    # 3. Load BOTH Teacher Models
    teacher1 = Restormer(in_channels=3, out_channels=3, dim=32).to(device)
    teacher2 = Restormer(in_channels=3, out_channels=3, dim=32).to(device)

    if os.path.exists(args.teacher1_weights) and os.path.exists(args.teacher2_weights):
        teacher1.load_state_dict(torch.load(args.teacher1_weights, map_location=device))
        teacher2.load_state_dict(torch.load(args.teacher2_weights, map_location=device))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully loaded both Teacher models")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Could not find one or both teacher weights.")
        return

    # Freeze both teachers
    teacher1.eval()
    teacher2.eval()
    for param in teacher1.parameters(): param.requires_grad = False
    for param in teacher2.parameters(): param.requires_grad = False

    # 4. Load Student Model
    student = UNet(in_channels=6, out_channels=3).to(device)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Student U-Net model loaded.")

    # 5. Training Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    
    best_loss = float('inf')

    # 6. Training Loop
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Dual-Teacher Distillation Training (CPU Mode)...")
    
    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            input_6ch = batch['input_6ch'].to(device)
            noisy_t = batch['noisy_t'].to(device)

            # --- DUAL TEACHER PREDICTION ---
            with torch.no_grad():
                t1_clean = teacher1(noisy_t)
                t2_clean = teacher2(noisy_t)
                
                # Average their knowledge (Ensemble)
                ensemble_clean_target = (t1_clean + t2_clean) / 2.0

            # --- STUDENT TRAINING ---
            optimizer.zero_grad(set_to_none=True)
            
            student_out = student(input_6ch)
            loss = criterion(student_out, ensemble_clean_target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.5f}")

        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), 'dual_distilled_unet_best.pt')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] * New Best Model Saved (Loss: {best_loss:.5f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default=r'D:\soft computing data\bdd100k\bdd100k\images\10k\train')
    parser.add_argument('--teacher1_weights', type=str, default='teacher_best.pt')
    parser.add_argument('--teacher2_weights', type=str, default='cityscapes_final.pt')
    main(parser.parse_args())
