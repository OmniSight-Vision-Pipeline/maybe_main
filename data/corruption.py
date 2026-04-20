import cv2
import numpy as np
import albumentations as A

class ImageCorruptor:
    def __init__(self):
        # We use Albumentations for some basic corruptions
        # plus custom OpenCV logic for physical lens distortion.
        self.albu_transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ])

    def add_synthetic_rain(self, image):
        """Simulate synthetic rain streaks."""
        imshape = image.shape
        rain_drops = np.random.uniform(0, 255, (imshape[0], imshape[1]))
        
        # Blur the noise to create streaks
        rain_layer = cv2.blur(rain_drops, (3, 15))
        rain_layer = cv2.threshold(rain_layer, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Add rain to image
        rain_layer = np.expand_dims(rain_layer, axis=-1)
        rain_layer = np.repeat(rain_layer, 3, axis=-1)
        
        # Blend
        result = cv2.addWeighted(image, 0.8, rain_layer.astype(np.uint8), 0.2, 0)
        return result
        
    def add_fog(self, image):
        """Simulate fog using a transmission map."""
        row, col, ch = image.shape
        fog_layer = np.ones((row, col, ch), dtype=np.uint8) * 200
        result = cv2.addWeighted(image, 0.5, fog_layer, 0.5, 0)
        return result

    def apply_lens_distortion(self, image):
        """
        Simulate physical lens effects: Gaussian blur for out-of-focus water,
        light blooming/halos for headlights, and contrast reduction.
        """
        # 1. Out-of-focus water / dirt (Gaussian Blur)
        blurred = cv2.GaussianBlur(image, (7, 7), 0)
        
        # 2. Light blooming / halos for bright spots (e.g., headlights)
        # Convert to grayscale to find bright spots
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Dilate and heavily blur the bright areas to create halos
        kernel = np.ones((5, 5), np.uint8)
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=2)
        bloom = cv2.GaussianBlur(bright_mask, (41, 41), 0)
        
        # Add bloom back to the image
        bloom_color = cv2.cvtColor(bloom, cv2.COLOR_GRAY2BGR)
        image_with_bloom = cv2.addWeighted(blurred, 0.8, bloom_color, 0.4, 0)
        
        # 3. Contrast reduction (common in fog/rain)
        reduced_contrast = cv2.convertScaleAbs(image_with_bloom, alpha=0.7, beta=30)
        
        return reduced_contrast

    def __call__(self, image):
        # image is assumed to be a numpy array (H, W, C) in BGR or RGB
        img = image.copy()
        
        # Randomly apply effects
        if np.random.rand() > 0.5:
            img = self.add_synthetic_rain(img)
        if np.random.rand() > 0.5:
            img = self.add_fog(img)
            
        # Always apply some lens distortion to close the domain gap
        img = self.apply_lens_distortion(img)
        
        # Apply albumentations
        img = self.albu_transforms(image=img)['image']
        
        return img
