import torch
import torch.nn as nn
import torch.optim as optim
from .base_classifier import BaseClassifier

class SimpleMLPClassifier(BaseClassifier):
    """
    Simple MLP classifier with minimal layers.
    """
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dim: int = 256, dropout_rate: float = 0.5, 
                 learning_rate: float = 0.001, num_epochs: int = 100, **kwargs):
        super().__init__(input_dim, num_classes, **kwargs)
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Simple MLP: input -> hidden -> output
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=learning_rate)
        
        # Ensure all parameters are float32
        self.mlp.float()
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the MLP classifier to the training data.
        
        Args:
            X: Training features of shape (n_samples, input_dim)
            y: Training labels of shape (n_samples,)
        """
        self.mlp.train()
        
        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.mlp(X)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Features of shape (n_samples, input_dim)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        self.mlp.eval()
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.mlp(X)
            _, predicted = torch.max(outputs, 1)
            return predicted
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for samples in X.
        
        Args:
            X: Features of shape (n_samples, input_dim)
            
        Returns:
            Predicted class probabilities of shape (n_samples, num_classes)
        """
        self.mlp.eval()
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.mlp(X)
            return torch.softmax(outputs, dim=1)
