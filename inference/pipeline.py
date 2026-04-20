import torch
import cv2
import numpy as np
from ultralytics import YOLO
from models.autoencoder import UNet

class OmniSightPipeline:
    def __init__(self, unet_weights="distilled_unet_best.pt", yolo_weights="yolov8n.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Student U-Net
        self.unet = UNet(in_channels=6, out_channels=3).to(self.device)
        try:
            self.unet.load_state_dict(torch.load(unet_weights, map_location=self.device))
            print("Loaded distilled UNet successfully.")
        except Exception as e:
            print(f"Warning: Could not load UNet weights: {e}")
        self.unet.eval()
        
        # Load YOLOv8
        self.yolo = YOLO(yolo_weights)
        
        self.prev_frame_tensor = None

    def preprocess(self, frame):
        """Convert BGR numpy to RGB tensor and normalize."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device) # Add batch dim

    def postprocess(self, tensor):
        """Convert tensor back to BGR numpy."""
        tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def process_frame(self, current_frame):
        # 1. Preprocess
        curr_tensor = self.preprocess(current_frame)
        
        # Initialize prev_frame if it's the first frame
        if self.prev_frame_tensor is None:
            self.prev_frame_tensor = curr_tensor.clone()
            
        # Create 6-channel input
        input_6ch = torch.cat([self.prev_frame_tensor, curr_tensor], dim=1)
        
        # 2. U-Net Restoration with Mixed Precision
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                restored_tensor = self.unet(input_6ch)
                
        # Update prev frame
        self.prev_frame_tensor = curr_tensor.clone()
        
        # 3. Postprocess restored frame
        restored_frame = self.postprocess(restored_tensor)
        
        # 4. YOLOv8 + ByteTrack Detection
        # Pass the cleaned frame directly into YOLO's track method
        results = self.yolo.track(restored_frame, tracker="bytetrack.yaml", persist=True)
        
        return restored_frame, results

    def run_video(self, video_path, output_path="output.mp4"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run pipeline
            restored_frame, results = self.process_frame(frame)
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Combine original and restored/annotated for visualization
            # Resize annotated to half width, original to half width
            h, w = frame.shape[:2]
            half_w = w // 2
            
            orig_resized = cv2.resize(frame, (half_w, h))
            anno_resized = cv2.resize(annotated_frame, (half_w, h))
            
            combined = np.hstack((orig_resized, anno_resized))
            out.write(combined)
            
            cv2.imshow("OmniSight Pipeline (Left: Original Corrupted, Right: Restored+Tracked)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = OmniSightPipeline()
    # Dummy call - replace with actual path
    # pipeline.run_video("corrupted_test_video.mp4")
