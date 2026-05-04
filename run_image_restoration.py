import os
import sys
import cv2
import torch
import numpy as np
from models.restormer import Restormer

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    return tensor

def save_image(tensor, out_path):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_image_restoration.py <input_image_path> [weights_filename]")
        sys.exit(1)
    input_path = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Restormer(in_channels=3, out_channels=3).to(device)
    
    # Load weights (defaults to teacher_best.pt if not provided)
    weights_filename = sys.argv[2] if len(sys.argv) > 2 else 'teacher_best.pt'
    weight_path = os.path.join(os.path.dirname(__file__), weights_filename)
    if not os.path.exists(weight_path):
        print(f"Weight file not found: {weight_path}")
        sys.exit(1)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        img_tensor = load_image(input_path).to(device)
        
        # Pad image to be a multiple of 4 (since the model downsamples twice by factor of 2)
        _, _, h, w = img_tensor.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        restored = model(img_tensor)
        
        # Unpad
        if pad_h > 0 or pad_w > 0:
            restored = restored[:, :, :h, :w]
            
    base, ext = os.path.splitext(input_path)
    out_path = f"{base}_restored{ext}"
    save_image(restored, out_path)
    print(f"Restored image saved to: {out_path}")

if __name__ == "__main__":
    main()
