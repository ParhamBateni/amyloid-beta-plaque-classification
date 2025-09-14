import torch
import torch.nn as nn
from models.config import Config
from models.supervised.feature_extractors.base_feature_extractor import (
    create_feature_extractor,
)
from models.supervised.classifiers.base_classifier import create_classifier
from tqdm import tqdm
from utils import print_log
from models.base_model import BaseModel
import numpy as np


class SupervisedModel(BaseModel):
    """
    Supervised learning model that combines a feature extractor and a classifier.
    Supports freezing the feature extractor for transfer learning scenarios.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # Parse names and device from config
        self.feature_extractor_name = (
            config.supervised.supervised_config.feature_extractor_name
        )
        self.feature_extractor_config = (
            config.supervised.supervised_config.feature_extractor
        )
        self.classifier_name = config.supervised.supervised_config.classifier_name
        self.classifier_config = config.supervised.supervised_config.classifier

        self.num_classes = len(config.name_to_label)
        self.num_epochs = config.supervised.supervised_config.training.num_epochs
        self.learning_rate = config.supervised.supervised_config.training.learning_rate
        self.weight_decay = config.supervised.supervised_config.training.weight_decay
        self.early_stop = config.supervised.supervised_config.training.early_stop
        self.log_mode = config.general_config.system.log_mode
        self.device = config.general_config.system.device

        # Create feature extractor using nested config under 'feature_extractor'
        self.feature_extractor = create_feature_extractor(
            feature_extractor_name=self.feature_extractor_name,
            input_dim=config.general_config.data.downscaled_image_size[0],
            feature_extractor_config=config.supervised.feature_extractors_config[
                self.feature_extractor_name
            ].to_dict(),
        )

        # Flags for optional extra features (ignored in end-to-end path)
        self.use_extra_features = config.general_config.data.use_extra_features
        self.image_feature_dim = self.feature_extractor.get_output_dim()
        self.extra_feature_dim = config.general_config.data.extra_feature_dim

        # Create classifier using nested config under 'classifier'
        self.classifier = create_classifier(
            classifier_name=self.classifier_name,
            input_dim=self.image_feature_dim + self.extra_feature_dim
            if self.use_extra_features
            else self.image_feature_dim,
            num_classes=self.num_classes,
            classifier_config=config.supervised.classifiers_config[
                self.classifier_name
            ].to_dict(),
        )

        # Move to device
        self.to(self.device)

    def fit(
        self,
        labeled_train_dataloader: torch.utils.data.DataLoader,
        labeled_val_dataloader: torch.utils.data.DataLoader,
        unlabeled_dataloader: torch.utils.data.DataLoader = None,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Train the supervised model with early stopping and logging."""

        print_log(
            f"Starting supervised training for {self.num_epochs} epochs",
            log_mode=self.log_mode,
        )

        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(self.num_epochs):
            # Training phase
            self.train()
            # Confirm we're in training mode and inspect final layer bias
            train_loss = 0.0
            train_num_samples = 0
            train_num_correct = 0
            for batch_idx, (
                _,
                _,
                _,
                _,
                normalized_transformed_images,
                extra_features,
                labels,
            ) in enumerate(
                tqdm(
                    labeled_train_dataloader,
                    desc=f"Epoch {epoch + 1}/{self.num_epochs} - Training",
                )
            ):
                normalized_transformed_images = normalized_transformed_images.to(
                    self.device
                )
                labels = labels.to(self.device)

                # Only move extra_features to device if we're using them
                if self.use_extra_features:
                    extra_features = extra_features.to(self.device)

                optimizer.zero_grad()
                # Use raw images for training (matches notebook approach)
                outputs = self(normalized_transformed_images, None)
                predicted = torch.max(outputs.data, 1)[1]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_num_samples += labels.size(0)
                train_num_correct += (predicted == labels).sum().item()
                _, predicted = torch.max(outputs.data, 1)
                print_log(
                    f"Actual labels: {labels}, Predicted labels: {predicted}",
                    log_mode=self.log_mode,
                )

            # Validation phase
            self.eval()
            val_loss = 0.0

            val_num_samples = 0
            val_num_correct = 0
            with torch.no_grad():
                for (
                    _,
                    _,
                    _,
                    _,
                    normalized_transformed_images,
                    extra_features,
                    labels,
                ) in tqdm(
                    labeled_val_dataloader,
                    desc=f"Epoch {epoch + 1}/{self.num_epochs} - Validation",
                ):
                    normalized_transformed_images = normalized_transformed_images.to(
                        self.device
                    )
                    labels = labels.to(self.device)

                    # Only move extra_features to device if we're using them
                    if self.use_extra_features:
                        extra_features = extra_features.to(self.device)

                    # Use raw images for validation (matches notebook approach)
                    outputs = self(normalized_transformed_images, None)
                    predicted = torch.max(outputs.data, 1)[1]
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_loss += loss.item()
                    val_num_samples += labels.size(0)
                    val_num_correct += (predicted == labels).sum().item()
            # Calculate metrics
            train_acc = 100 * train_num_correct / train_num_samples
            val_acc = 100 * val_num_correct / val_num_samples

            train_loss = train_loss / len(labeled_train_dataloader)
            val_loss = val_loss / len(labeled_val_dataloader)
            print_log(
                f"Epoch {epoch + 1}/{self.num_epochs}:",
                log_mode=self.log_mode,
            )
            print_log(
                f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%",
                log_mode=self.log_mode,
            )
            print_log(
                f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%",
                log_mode=self.log_mode,
            )
            train_losses.append(np.round(train_loss, 3))
            val_losses.append(np.round(val_loss, 3))
            train_accuracies.append(np.round(train_acc, 3))
            val_accuracies.append(np.round(val_acc, 3))

            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop:
                    print_log(
                        f"Early stopping triggered after {epoch + 1} epochs",
                        log_mode=self.log_mode,
                    )
                    self.load_state_dict(best_model_state)
                    break

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(
        self,
        labeled_test_dataloader: torch.utils.data.DataLoader,
    ):
        """Predict the labels of the supervised model and return labels and predictions."""

        print_log("Predicting labels of the supervised model", log_mode=self.log_mode)

        self.eval()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for (
                _,
                _,
                _,
                _,
                normalized_transformed_images,
                extra_features,
                labels,
            ) in tqdm(labeled_test_dataloader, desc="Predicting labels"):
                normalized_transformed_images = normalized_transformed_images.to(
                    self.device
                )
                labels = labels.to(self.device)

                # Only move extra_features to device if we're using them
                if self.use_extra_features:
                    extra_features = extra_features.to(self.device)

                # Use raw images for testing (matches notebook approach)
                outputs = self(normalized_transformed_images, None)
                predicted = torch.max(outputs.data, 1)[1]

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        return all_labels, all_preds

    def forward(
        self, x_image: torch.Tensor, x_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the supervised model.

        Args:
            x_image: Image tensor of shape (batch_size, channels, height, width)
            x_features: Additional features tensor of shape (batch_size, feature_dim)

        Returns:
            Model output
        """
        # Notebook-style architecture
        x = self.feature_extractor(x_image)
        x = self.classifier(x)
        return x

    def extract_features(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the feature extractor.

        Args:
            x_image: Image tensor

        Returns:
            Extracted features
        """
        return self.feature_extractor(x_image)

    def get_name(self):
        """String representation of the model."""
        return f"{self.__class__.__name__}_{self.feature_extractor_name}_{self.classifier_name}"


if __name__ == "__main__":
    from models.data.plaque_dataset import load_dataloaders

    config = Config.load_config("configs", "supervised")
    model = SupervisedModel(config)
    train_dataloader, val_dataloader, test_dataloader, unlabeled_dataloader = (
        load_dataloaders(config)
    )
    (
        image_path,
        raw_image,
        transformed_image,
        normalized_raw_image,
        normalized_transformed_image,
        extra_features,
        labels,
    ) = next(iter(train_dataloader))
    model_output = model(normalized_transformed_image, None)
    print("Model input: ", normalized_transformed_image)
    print("Model input shape: ", normalized_transformed_image.shape)
    print("Model input mean: ", normalized_transformed_image.mean())
    print("Model input std: ", normalized_transformed_image.std())
    print("Model input min: ", normalized_transformed_image.min())
    print("Model input max: ", normalized_transformed_image.max())
    print("Model output: ", model_output)
    print("Model output shape: ", model_output.shape)
    print("Model output mean: ", model_output.mean())
    print("Model output std: ", model_output.std())
    print("Model output min: ", model_output.min())
    print("Model output max: ", model_output.max())
