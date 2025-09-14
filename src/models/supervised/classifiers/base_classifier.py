from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseClassifier(ABC, nn.Module):
    """
    Base class for all classifiers.
    All classifiers should inherit from this class.
    """
    
    def __init__(self, input_dim: int, num_classes: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.kwargs = kwargs
    
    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing logits.
        
        Args:
            X: Features of shape (n_samples, input_dim)
        Returns:
            Logits tensor of shape (n_samples, num_classes)
        """
        pass
    
    
    def save(self, path: str) -> None:
        """
        Save the classifier to a file.
        
        Args:
            path: Path to save the classifier
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseClassifier':
        """
        Load a classifier from a file.
        
        Args:
            path: Path to load the classifier from
            
        Returns:
            Loaded classifier
        """
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

def create_classifier(classifier_name: str, input_dim: int, num_classes: int, classifier_config: dict) -> BaseClassifier:
    full_cfg = {"input_dim": input_dim, "num_classes": num_classes, **classifier_config}
    if classifier_name == "custom_mlp":
        from .custom_mlp_classifier import CustomMLPClassifier
        return CustomMLPClassifier(**full_cfg)
    elif classifier_name == "simple_mlp":
        from .simple_mlp_classifier import SimpleMLPClassifier
        return SimpleMLPClassifier(**full_cfg)
    elif classifier_name == "deep_mlp":
        from .deep_mlp_classifier import DeepMLPClassifier
        return DeepMLPClassifier(**full_cfg)
    # TODO: Add more classifiers
    else:
        raise ValueError(f"Classifier {classifier_name} not found")
