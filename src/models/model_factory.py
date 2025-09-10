from models.supervised.feature_extractors.base_feature_extractor import BaseFeatureExtractor
from models.supervised.feature_extractors.simple_cnn_feature_extractor import SimpleCNNFeatureExtractor
from models.supervised.feature_extractors.resnet_feature_extractor import ResNetFeatureExtractor
from models.supervised.classifiers.base_classifier import BaseClassifier
from models.supervised.classifiers.custom_mlp_classifier import CustomMLPClassifier
from models.supervised.classifiers.simple_mlp_classifier import SimpleMLPClassifier
from models.supervised.classifiers.deep_mlp_classifier import DeepMLPClassifier

class ModelFactory:
    @staticmethod
    def create_feature_extractor(feature_extractor_name: str, feature_extractor_config: dict) -> BaseFeatureExtractor:
        if feature_extractor_name == "simple_cnn":
            return SimpleCNNFeatureExtractor(**feature_extractor_config)
        elif feature_extractor_name.startswith("resnet"):
            return ResNetFeatureExtractor(**feature_extractor_config)
        # TODO: Add more feature extractors
        else:
            raise ValueError(f"Feature extractor {feature_extractor_name} not found")
    
    @staticmethod
    def create_classifier(classifier_name: str, input_dim: int, num_classes: int, classifier_config: dict) -> BaseClassifier:
        full_cfg = {"input_dim": input_dim, "num_classes": num_classes, **classifier_config}
        if classifier_name == "custom_mlp":
            return CustomMLPClassifier(**full_cfg)
        elif classifier_name == "simple_mlp":
            return SimpleMLPClassifier(**full_cfg)
        elif classifier_name == "deep_mlp":
            return DeepMLPClassifier(**full_cfg)
        # TODO: Add more classifiers
        else:
            raise ValueError(f"Classifier {classifier_name} not found")