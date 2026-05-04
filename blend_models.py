import torch
import sys
import os

def blend_weights(model1_path, model2_path, output_path, alpha=0.5):
    """
    Blends two PyTorch state dicts to create an ensemble model.
    alpha: weight given to model1. (1-alpha) is given to model2.
    """
    if not os.path.exists(model1_path):
        print(f"Error: Could not find {model1_path}")
        return
    if not os.path.exists(model2_path):
        print(f"Error: Could not find {model2_path}")
        return

    print(f"Loading {model1_path}...")
    state_dict1 = torch.load(model1_path, map_location='cpu')
    
    print(f"Loading {model2_path}...")
    state_dict2 = torch.load(model2_path, map_location='cpu')
    
    blended_state_dict = {}
    
    print(f"Blending weights with ratio {alpha*100}% Model 1 / {(1-alpha)*100}% Model 2...")
    for key in state_dict1.keys():
        if key in state_dict2:
            blended_state_dict[key] = (state_dict1[key] * alpha) + (state_dict2[key] * (1.0 - alpha))
        else:
            blended_state_dict[key] = state_dict1[key]
            
    print(f"Saving blended model to {output_path}...")
    torch.save(blended_state_dict, output_path)
    print("Done! You can now use this blended model for inference.")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python blend_models.py <model1.pt> <model2.pt> <output.pt> [alpha]")
        print("Example: python blend_models.py teacher_best.pt cityscapes_final.pt blended_model.pt 0.5")
        sys.exit(1)
        
    m1 = sys.argv[1]
    m2 = sys.argv[2]
    out = sys.argv[3]
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    
    blend_weights(m1, m2, out, alpha)
