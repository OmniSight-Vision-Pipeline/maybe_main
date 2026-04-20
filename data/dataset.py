import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .corruption import ImageCorruptor

class OmniSightDataset(Dataset):
    def __init__(self, bdd100k_dir, real_rain_dir, transform=None):
        """
        Mixes BDD100k (synthetically corrupted) and Real-Rain datasets.
        """
        self.bdd100k_dir = bdd100k_dir
        self.real_rain_dir = real_rain_dir
        self.transform = transform
        self.corruptor = ImageCorruptor()
        
        self.bdd_files = []
        self.real_files = []
        
        if os.path.exists(bdd100k_dir):
            self.bdd_files = sorted([f for f in os.listdir(bdd100k_dir) if f.endswith(('.jpg', '.png'))])
        if os.path.exists(real_rain_dir):
            self.real_files = sorted([f for f in os.listdir(real_rain_dir) if f.endswith(('.jpg', '.png'))])
            
        self.total_samples = max(len(self.bdd_files), len(self.real_files)) * 2 
        if self.total_samples == 0:
            self.total_samples = 100 # Fallback for dummy testing without data

    def __len__(self):
        return self.total_samples

    def _get_frame_pair(self, file_list, dir_path, idx):
        actual_idx = idx % max(1, len(file_list))
        file_t = file_list[actual_idx]
        path_t = os.path.join(dir_path, file_t)
        
        prev_idx = max(0, actual_idx - 1)
        file_t_prev = file_list[prev_idx]
        path_t_prev = os.path.join(dir_path, file_t_prev)
        
        img_t = cv2.imread(path_t)
        img_t_prev = cv2.imread(path_t_prev)
        
        if img_t is None: img_t = np.zeros((256, 256, 3), dtype=np.uint8)
        if img_t_prev is None: img_t_prev = np.zeros((256, 256, 3), dtype=np.uint8)
        
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        img_t_prev = cv2.cvtColor(img_t_prev, cv2.COLOR_BGR2RGB)
        
        return img_t_prev, img_t

    def __getitem__(self, idx):
        # 80% BDD100k, 20% Real-Rain
        use_real = np.random.rand() < 0.2
        
        if use_real and len(self.real_files) > 0:
            img_t_prev, img_t_clean = self._get_frame_pair(self.real_files, self.real_rain_dir, idx)
            img_t_noisy = self.corruptor.apply_lens_distortion(img_t_clean)
            img_t_prev_noisy = self.corruptor.apply_lens_distortion(img_t_prev)
        else:
            if len(self.bdd_files) > 0:
                img_t_prev, img_t_clean = self._get_frame_pair(self.bdd_files, self.bdd100k_dir, idx)
            else:
                img_t_clean = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                img_t_prev = img_t_clean.copy()
                
            img_t_noisy = self.corruptor(img_t_clean)
            img_t_prev_noisy = self.corruptor(img_t_prev)

        # Normalize and convert to tensors (C, H, W)
        img_t_noisy = torch.from_numpy(img_t_noisy).float().permute(2, 0, 1) / 255.0
        img_t_prev_noisy = torch.from_numpy(img_t_prev_noisy).float().permute(2, 0, 1) / 255.0
        img_t_clean = torch.from_numpy(img_t_clean).float().permute(2, 0, 1) / 255.0
        
        # Concatenate T-1 and T along the channel dimension -> 6 channels
        input_6ch = torch.cat([img_t_prev_noisy, img_t_noisy], dim=0)

        return {
            'input_6ch': input_6ch,
            'noisy_t': img_t_noisy,
            'clean_t': img_t_clean
        }
