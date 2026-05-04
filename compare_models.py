import sys
import os
import cv2
import torch
import numpy as np
from models.restormer import Restormer
from run_image_restoration import load_image

def save_comparison(images, labels, out_path):
    # images is a list of numpy arrays (H,W,3) BGR
    labeled_images = []
    for img, label in zip(images, labels):
        img_copy = img.copy()
        # Add a black background for text to make it readable
        cv2.rectangle(img_copy, (5, 5), (250, 45), (0, 0, 0), -1)
        cv2.putText(img_copy, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        labeled_images.append(img_copy)
    
    # Concatenate all images horizontally side-by-side
    res = cv2.hconcat(labeled_images)
    cv2.imwrite(out_path, res)

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_models.py <input_image_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    models_to_test = {
        '1. Base Teacher': 'teacher_best.pt',
        '2. Cityscapes': 'cityscapes_final.pt',
        '3. Blended 50/50': 'blended_model.pt'
    }
    
    # Load original image
    orig_img = cv2.imread(input_path)
    if orig_img is None:
        print("Image not found.")
        sys.exit(1)
        
    # Resize to a width of 640 so the horizontal stack isn't way too massive to view
    h, w = orig_img.shape[:2]
    new_w = 640
    new_h = int(h * (new_w / w))
    orig_img = cv2.resize(orig_img, (new_w, new_h))
    
    results = [orig_img]
    labels = ['Original']
    
    # Initialize the Restormer architecture
    model = Restormer(in_channels=3, out_channels=3).to(device)
    
    for name, weight_file in models_to_test.items():
        if not os.path.exists(weight_file):
            print(f"Skipping {name}, {weight_file} not found.")
            continue
            
        print(f"Processing through {name}...")
        model.load_state_dict(torch.load(weight_file, map_location=device))
        model.eval()
        
        with torch.no_grad():
            img_tensor = load_image(input_path).to(device)
            # Pad image to be multiple of 4
            _, _, th, tw = img_tensor.shape
            pad_h = (4 - th % 4) % 4
            pad_w = (4 - tw % 4) % 4
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            
            restored = model(img_tensor)
            
            # Unpad
            if pad_h > 0 or pad_w > 0:
                restored = restored[:, :, :th, :tw]
                
        # Convert tensor back to numpy image
        out_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        out_np = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        
        # Resize to match the display size
        out_np = cv2.resize(out_np, (new_w, new_h))
        
        results.append(out_np)
        labels.append(name)
        
    base, ext = os.path.splitext(input_path)
    out_path = f"{base}_comparison{ext}"
    save_comparison(results, labels, out_path)
    print(f"\nDone! Comparison image saved to: {out_path}")

if __name__ == '__main__':
    main()
