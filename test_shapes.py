import torch
from models.restormer import Restormer
from models.autoencoder import UNet

def test_shapes():
    print("Testing Restormer shapes...")
    restormer = Restormer(in_channels=3, out_channels=3, dim=32)
    dummy_input = torch.randn(1, 3, 256, 256)
    out = restormer(dummy_input)
    assert out.shape == (1, 3, 256, 256), f"Restormer shape mismatch: {out.shape}"
    print("Restormer OK.")

    print("Testing UNet shapes...")
    unet = UNet(in_channels=6, out_channels=3)
    dummy_input_6ch = torch.randn(1, 6, 256, 256)
    out_unet = unet(dummy_input_6ch)
    assert out_unet.shape == (1, 3, 256, 256), f"UNet shape mismatch: {out_unet.shape}"
    print("UNet OK.")

if __name__ == "__main__":
    test_shapes()
