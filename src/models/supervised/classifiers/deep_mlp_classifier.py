import torch
import torch.nn as nn
import torch.optim as optim
from .base_classifier import BaseClassifier

class DeepMLPClassifier(BaseClassifier):
    """
    Deep MLP classifier with multiple hidden layers.
    """
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dims: list = [512, 256, 128], dropout_rate: float = 0.5, 
                 learning_rate: float = 0.001, num_epochs: int = 100, **kwargs):
        super().__init__(input_dim, num_classes, **kwargs)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Build deep MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=learning_rate)
        
        # Ensure all parameters are float32
        self.mlp.float()
