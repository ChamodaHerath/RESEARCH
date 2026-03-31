"""
Deepfake Video Preprocessing Pipeline
Requirements:
    pip install opencv-python facenet-pytorch torch torchvision tqdm pillow

This script processes FaceForensics++ videos by:
1. Extracting every N-th frame for efficiency.
2. Detecting faces using MTCNN.
3. Cropping with a 20% margin.
4. Resizing to 224x224 (ViT standard).
5. Normalizing with ImageNet mean/std (when saving to .pth).
6. Saving as JPGs or a stacked .pth tensor.
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from tqdm import tqdm
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Deepfake Preprocessing Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to video file or directory of videos")
    parser.add_argument("--output", type=str, default="processed", help="Directory to save processed frames")
    parser.add_argument("--n_skip", type=int, default=10, help="Detect faces every N-th frame")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run MTCNN on")
    parser.add_argument("--margin", type=float, default=0.20, help="Margin for cropping (percentage of bounding box)")
    parser.add_argument("--save_tensor", action="store_true", help="Save as .pth tensor instead of .jpg images")
    return parser.parse_args()

class PreprocessingPipeline:
    def __init__(self, device='cpu', margin=0.20):
        self.device = device
        self.margin = margin
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=224,
            margin=0, # We handle custom margin ourselves for more control
            post_process=False,
            device=self.device,
            select_largest=True, # Usually only one face per video in FF++
            keep_all=False
        )
        
        # ImageNet normalization and ViT standard resizing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def crop_face(self, frame, box):
        """
        Crop face with custom margin around the bounding box.
        """
        h, w, _ = frame.shape
        x1, y1, x2, y2 = box
        
        # Calculate box width and height
        bw = x2 - x1
        bh = y2 - y1
        
        # Add margin (20% as requested)
        margin_x = bw * self.margin
        margin_y = bh * self.margin
        
        # New coordinates with padding
        nx1 = max(0, int(x1 - margin_x))
        ny1 = max(0, int(y1 - margin_y))
        nx2 = min(w, int(x2 + margin_x))
        ny2 = min(h, int(y2 + margin_y))
        
        return frame[ny1:ny2, nx1:nx2]

    def process_video(self, video_path, output_dir, n_skip=10, save_tensor=False):
        video_name = Path(video_path).stem
        video_output_path = Path(output_dir) / video_name
        video_output_path.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        processed_tensors = []
        
        with tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every N-th frame
                if frame_idx % n_skip == 0:
                    # MTCNN expects RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    # Detect face
                    boxes, _ = self.mtcnn.detect(pil_img)
                    
                    if boxes is not None:
                        # Take the first (largest) face
                        box = boxes[0]
                        cropped_face = self.crop_face(frame_rgb, box)
                        
                        # Apply Vit standard transform & normalization
                        pil_face = Image.fromarray(cropped_face)
                        face_tensor = self.transform(pil_face)
                        
                        if save_tensor:
                            processed_tensors.append(face_tensor)
                        else:
                            # Save as JPG (standardize for Vit as requested)
                            # We'll save the transformed image back to a file for JPG if needed, 
                            # but usually for detection we save the face image and then normalize on load.
                            # However, user asks to "Resize all cropped face images to exactly 224x224".
                            # I will save as JPG. Transform pipeline above is for model input.
                            # Let's resize and save.
                            final_face = pil_face.resize((224, 224), Image.LANCZOS)
                            final_face.save(video_output_path / f"frame_{frame_idx:04d}.jpg")
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        if save_tensor and processed_tensors:
            final_tensor = torch.stack(processed_tensors)
            torch.save(final_tensor, video_output_path / f"{video_name}.pth")
            print(f"Saved {len(processed_tensors)} faces as tensor to {video_output_path}")

def main():
    args = get_args()
    
    pipeline = PreprocessingPipeline(device=args.device, margin=args.margin)
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle single video or directory
    if input_path.is_file():
        pipeline.process_video(str(input_path), str(output_dir), n_skip=args.n_skip, save_tensor=args.save_tensor)
    elif input_path.is_dir():
        video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mkv"))
        print(f"Found {len(video_files)} videos in {input_path}")
        for vid in video_files:
            pipeline.process_video(str(vid), str(output_dir), n_skip=args.n_skip, save_tensor=args.save_tensor)
    else:
        print(f"Error: Path {args.input} does not exist.")

if __name__ == "__main__":
    main()
