from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple

class BaseFeatureExtractor(ABC, nn.Module):
    """
    Base class for all feature extractors.
    All feature extractors should inherit from this class.
    """
    
    def __init__(self, input_dim: int,freeze_feature_extractor: bool = False):
        super().__init__()
        self.freeze_feature_extractor = freeze_feature_extractor
        self.input_dim = input_dim
        self.feature_extractor = None

        # Freeze feature extractor if requested
        if self.freeze_feature_extractor:
            for param in self.parameters():
                param.requires_grad = False
    
    @abstractmethod
    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.
        
        Args:
            x_image: Image tensor of shape (batch_size, channels, height, width)    
            
        Returns:
            Extracted features tensor
        """
        pass
    
    def get_output_dim(self) -> int:
        """
        Get the flattened output feature dimension of the extractor.
        
        Returns:
            Integer number of features after the extractor's forward pass.
        """
        with torch.no_grad():
            dummy = torch.zeros(size=(1, 3, self.input_dim, self.input_dim), dtype=torch.float32)
            output = self.forward(dummy)
            if output.dim() > 2:
                output = output.view(output.size(0), -1)
            return int(output.shape[1])
