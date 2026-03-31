import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# ---- BiSeNet Placeholder Implementation ----
# Note: Since BiSeNet is a custom architecture usually adopted from third-party 
# repos for CelebAMask-HQ, here is a structural mock for compilation.
# In production, import the real architecture: `from model import BiSeNet`
class DummyBiSeNet(nn.Module):
    """
    Simulated BiSeNet model for compilation.
    Replace with: https://github.com/zllrunning/face-parsing.PyTorch
    """
    def __init__(self, n_classes=19):
        super(DummyBiSeNet, self).__init__()
        self.n_classes = n_classes
        # Dummy convolutional layer to simulate outputting 19 class logits
        self.conv = nn.Conv2d(3, n_classes, kernel_size=1)

    def forward(self, x):
        # BiSeNet traditionally outputs the primary output and two auxiliary 
        # outputs for training. We only need the primary output for inference.
        out = self.conv(x)
        return out, out, out

# ---- Face Parsing Module ----
class FaceParsingModule:
    """
    Real-time Face Parsing module leveraging a BiSeNet backbone trained on 
    the CelebAMask-HQ dataset to extract precise facial landmarks.
    """
    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Initializing FaceParsingModule on {self.device}...")
        
        # 1. Model Selection
        # Instantiate BiSeNet model for 19 classes (CelebAMask-HQ format)
        self.net = DummyBiSeNet(n_classes=19) 
        
        if model_path:
            # Load pre-trained weights safely mapped to target device
            print(f"Loading weights from {model_path}...")
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.net.to(self.device)
        self.net.eval()
        
        # 4. Preprocessing Integration
        # ViT/ImageNet normalization bounds specifically expected by most pretrained BiSeNets
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Semantic Label Merging Mapping 
        # CelebAMask-HQ Index Guide:
        # l_brow:2, r_brow:3, l_eye:4, r_eye:5
        # nose:10, mouth_interior:11, u_lip:12, l_lip:13
        self.categories = {
            'Eyes': [2, 3, 4, 5],
            'Nose': [10],
            'Mouth': [11, 12, 13]
        }
        
    def _create_binary_masks(self, segmentation_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        3. Output Format conversion. Given a dense 0-18 segmentation array,
        merges classes into the 4 distinct binary outputs.
        """
        batch_size, h, w = segmentation_map.shape
        masks = {}
        
        # Keep track of pixels mapped to valid specific categories
        mapped_pixels = torch.zeros((batch_size, h, w), dtype=torch.bool, device=self.device)
        
        for category, labels in self.categories.items():
            # Create a boolean mask where any of the matching labels are present
            cat_mask = torch.zeros((batch_size, h, w), dtype=torch.bool, device=self.device)
            for label in labels:
                cat_mask = cat_mask | (segmentation_map == label)
            
            # Convert boolean to standard binary floats [0.0, 1.0]
            masks[category] = cat_mask.float() 
            mapped_pixels = mapped_pixels | cat_mask
            
        # 'Background/Skin' is fundamentally defined as exactly the inverse of the facial features
        masks['Background/Skin'] = (~mapped_pixels).float()
        
        return masks

    @torch.no_grad() # 6. Efficiency Check (Save Memory, No Backprop)
    def parse_batch(self, batch_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input Handling: Process a batch of face crops through the network.
        Expected input: (B, 3, 224, 224) with pixel value range [0, 1]
        """
        # Apply strict ImageNet normalization to the incoming batch
        batch_normalized = self.transform(batch_rgb).to(self.device)
        
        # Forward pass inference -> we unpack only the primary prediction tensor
        outputs, _, _ = self.net(batch_normalized)
        
        # Argmax over the channel dimension (19 channels -> 2D spatial prediction map)
        segmentation_map = outputs.argmax(dim=1) # Shape: (B, 224, 224)
        
        # Merge exactly into 4 explicitly separated dense binary mask dictionaries
        binary_masks = self._create_binary_masks(segmentation_map)
        
        return binary_masks

# 5. Visual Verification Function
def verify_parsing(face_tensor: torch.Tensor, category_masks: Dict[str, torch.Tensor], save_path='parsing_verification.png'):
    """
    Extracts the first sample in batch and creates a comparative matplotlib visualization
    of the Mouth and Eyes against the initial RGB image.
    """
    # Assuming batch size 1 for visualization
    orig_img = face_tensor[0].cpu().permute(1, 2, 0).numpy()
    
    # Clip for safe Matplotlib plotting boundaries
    orig_img = np.clip(orig_img, 0, 1)
    
    eyes_mask = category_masks['Eyes'][0].cpu().numpy()
    mouth_mask = category_masks['Mouth'][0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Crop (3x224x224)")
    axes[0].axis("off")
    
    axes[1].imshow(eyes_mask, cmap='viridis')
    axes[1].set_title("Binary Mask: Eyes & Brows")
    axes[1].axis("off")
    
    axes[2].imshow(mouth_mask, cmap='viridis')
    axes[2].set_title("Binary Mask: Lips & Mouth")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visual verification matrix to: '{save_path}'")
    
if __name__ == "__main__":
    print("-- Testing BiSeNet FaceParsingModule Sandbox --\n")
    # Initialize implementation
    parser = FaceParsingModule()
    
    # Randomly initialized Dummy batch representing valid RGB image float parameters
    dummy_input = torch.rand(2, 3, 224, 224)
    print(f"Sample Input Loaded Shape: {dummy_input.shape}")
    
    # Run parsing loop
    predicted_masks = parser.parse_batch(dummy_input)
    
    # Validate final returned formats mathematically
    print(f"\nExtracted Output Mask Keys: {list(predicted_masks.keys())}")
    for feature_cat, b_mask in predicted_masks.items():
        print(f" > {feature_cat} Tensor Shape: {b_mask.shape} | Expected Binary Range: {b_mask.unique().tolist()}")
        
    # Run the verification graph dump
    verify_parsing(dummy_input, predicted_masks)
