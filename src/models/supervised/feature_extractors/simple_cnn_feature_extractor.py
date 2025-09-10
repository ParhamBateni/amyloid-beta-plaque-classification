import torch
import torch.nn as nn
from .base_feature_extractor import BaseFeatureExtractor

class SimpleCNNFeatureExtractor(BaseFeatureExtractor):
    """
    Simple CNN feature extractor.
    """
    
    def __init__(self, input_dim: int, freeze_feature_extractor: bool = False,  output_dim: int = 28):
        super().__init__(input_dim, freeze_feature_extractor)
        self.output_dim = output_dim
        # Match notebook architecture exactly: 3 MaxPool2d layers on 224x224 → 28x28
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 → 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 → 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 → 28
            
            # No adaptive pooling - use exact notebook structure
            nn.Flatten()  # Results in 128 * 28 * 28 = 100,352 features
        )
        # Ensure all parameters are float32
        self.float()
    
    def get_output_dim(self) -> int:
        """Return the output dimension of the feature extractor."""
        # 128 channels * 28 * 28 = 100,352 features (matches notebook)
        return 128 * 28 * 28
    
    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.
        
        Args:
            x_image: Image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Combined features tensor
        """
        image_features = self.feature_extractor(x_image)
        return image_features