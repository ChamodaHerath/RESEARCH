import torch
import torch.nn as nn
from transformers import ViTModel

class DeepfakeViT(nn.Module):
    """
    Vision Transformer for Deepfake Detection (Binary Classification: Real vs. Fake).
    Includes attention map extraction for explainability.
    """
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super(DeepfakeViT, self).__init__()
        
        # 1. Model Loading: Load the pre-trained ViT base model
        # We set output_attentions=True to extract attention weights for explainability
        # We don't need the add_pooling_layer since we will use the CLS token directly
        self.vit = ViTModel.from_pretrained(model_name, add_pooling_layer=False, output_attentions=True)
        
        # 2. Binary Classification Head: Replace the 1,000-class ImageNet head
        # The hidden size for vit-base is 768. We output 2 logits (Real, Fake)
        self.classifier = nn.Linear(self.vit.config.hidden_size, 2)
        
        # 3. Transfer Learning: Freeze the first 6 layers of the transformer
        # This preserves general ImageNet-trained feature extraction
        # while allowing the later layers to fine-tune on deeper, deepfake-specific artifacts.
        for layer_idx in range(6):
            for param in self.vit.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    def extract_cls_attention_map(self, attentions):
        """
        Helper function to extract the self-attention weights from the last transformer block,
        specifically capturing how the [CLS] token attends to the image patches.
        """
        # attentions is a tuple of length num_layers (12 for vit-base).
        # We want the attention from the very last layer:
        # Shape: (batch_size, num_heads, sequence_length, sequence_length)
        last_layer_attention = attentions[-1] 
        
        # Average the attention weights across all multi-head attention heads
        # Shape becomes: (batch_size, sequence_length, sequence_length)
        mean_attention = torch.mean(last_layer_attention, dim=1) 
        
        # The sequence is length 197: index 0 is the [CLS] token, indices 1-196 are the patch tokens.
        # We want the attention of the [CLS] token (row 0) to all patch tokens (cols 1 to 196)
        # Shape becomes: (batch_size, 196)
        cls_attention = mean_attention[:, 0, 1:] 
        
        # 4. Patch-to-Grid Transformation: Reshape the 196 linear elements back to a 14x14 grid
        batch_size = cls_attention.size(0)
        attention_map_14x14 = cls_attention.view(batch_size, 14, 14) 
        
        return attention_map_14x14

    def forward(self, pixel_values):
        """
        5. Forward Pass: Takes a (batch_size, 3, 224, 224) pixel_values tensor and returns 
        the classification Logits and a 2D Attention Map for interpretability.
        """
        # Forward pass through the ViT encoder
        outputs = self.vit(pixel_values=pixel_values)
        
        # Extract the sequence embeddings: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs.last_hidden_state
        
        # Get the representation for the [CLS] token (the 0th token in the sequence)
        cls_token_state = sequence_output[:, 0, :]
        
        # Pass the [CLS] representation through our custom Binary Classification head
        logits = self.classifier(cls_token_state)
        
        # Retrieve the attention map from the [CLS] token to the patches
        attention_map_2d = self.extract_cls_attention_map(outputs.attentions)
        
        return logits, attention_map_2d

if __name__ == "__main__":
    print("Initializing Deepfake ViT Model...")
    model = DeepfakeViT()
    
    # Verify parameter freezing
    frozen_layers = sum(1 for param in model.vit.encoder.layer[:6].parameters() if not param.requires_grad)
    trainable_layers = sum(1 for param in model.vit.encoder.layer[6:].parameters() if param.requires_grad)
    print(f"Number of frozen parameters in first 6 layers: {frozen_layers}")
    print(f"Number of trainable parameters in last 6 layers: {trainable_layers}")
    
    # Create a dummy input tensor: Batch size=1, Channels=3, Height=224, Width=224
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"\nInput Tensor Shape: {dummy_input.shape}")
    
    # Perform a forward pass
    print("Running forward pass...")
    with torch.no_grad(): # Disable gradient graph for inference
        logits, attention_map = model(dummy_input)
    
    # Ouputs
    print(f"\n-- Output Shapes --")
    print(f"Logits Shape: {logits.shape} -> (Batch, Classes)")
    print(f"Attention Map Shape: {attention_map.shape} -> (Batch, Grid_H, Grid_W)")
    
    print(f"\n-- Output Values --")
    print(f"Logits: {logits}")
    print(f"Attention Map (Top-Left 3x3 patch grid):")
    print(attention_map[0, :3, :3])
