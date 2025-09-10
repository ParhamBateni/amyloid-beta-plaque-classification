import torch
import torch.nn as nn
from models.config import Config
from models.model_factory import ModelFactory
from typing import Optional, Iterable
from tqdm import tqdm
from utils import print_log
class SupervisedModel(nn.Module):
    """
    Supervised learning model that combines a feature extractor and a classifier.
    Supports freezing the feature extractor for transfer learning scenarios.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        # Parse names and device from config
        self.feature_extractor_name = config.supervised.supervised_config.feature_extractor_name
        self.feature_extractor_config = config.supervised.supervised_config.feature_extractor
        self.classifier_name = config.supervised.supervised_config.classifier_name
        self.classifier_config = config.supervised.supervised_config.classifier
        
        self.num_classes = len(config.name_to_label)
        self.num_epochs = config.supervised.supervised_config.training.num_epochs
        self.learning_rate = config.supervised.supervised_config.training.learning_rate
        self.weight_decay = config.supervised.supervised_config.training.weight_decay
        self.early_stop = config.supervised.supervised_config.training.early_stop
        self.log_mode = config.general_config.system.log_mode
        self.device = config.general_config.system.device

        # End-to-end CNN matching notebook architecture exactly
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28 (this was missing!)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),  # Match notebook: 128*28*28 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Match notebook dropout
            nn.Linear(256, self.num_classes)
        )
        
        # Flags for optional extra features (ignored in end-to-end path)
        self.use_extra_features = False
        
        # Move to device
        self.to(self.device)

    def train_model(
        self,
        train_dataloader: Iterable,
        val_dataloader: Iterable,
    ) -> None:
        """Train the supervised model with early stopping and logging."""

        print_log(
            f"Starting supervised training for {self.num_epochs} epochs",
            log_mode=self.log_mode,
        )

        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            self.train()
            # Confirm we're in training mode and inspect final layer bias
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_idx, (_, _, _, _, normalized_transformed_images, extra_features, labels) in enumerate(
                tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.num_epochs} - Training",
                )
            ):
                normalized_transformed_images = normalized_transformed_images.to(self.device)
                labels = labels.to(self.device)
                
                # Only move extra_features to device if we're using them
                if self.use_extra_features:
                    extra_features = extra_features.to(self.device)

                optimizer.zero_grad()
                # Use raw images for training (matches notebook approach)
                outputs = self(normalized_transformed_images, None)
                # Log input and output stats for first few batches
                # if batch_idx < 3:
                #     x = normalized_transformed_images
                #     print_log(
                #         f"Batch {batch_idx} input stats (mean,std,min,max): "
                #         f"({x.mean().item():.4f}, {x.std().item():.4f}, {x.min().item():.4f}, {x.max().item():.4f})",
                #         log_mode=self.log_mode,
                #     )
                #     print_log(
                #         f"Batch {batch_idx} logits stats (mean,std,min,max): "
                #         f"({outputs.mean().item():.4f}, {outputs.std().item():.4f}, {outputs.min().item():.4f}, {outputs.max().item():.4f})",
                #         log_mode=self.log_mode,
                #     )
                loss = criterion(outputs, labels)
                loss.backward()
                # # Gradient diagnostics (first few batches only)
                # if batch_idx < 3:
                #     def _avg_grad(named_params):
                #         total = 0.0
                #         count = 0
                #         for name, p in named_params:
                #             if p.requires_grad and p.grad is not None:
                #                 total += p.grad.detach().abs().mean().item()
                #                 count += 1
                #         return (total / count) if count > 0 else None


                #     fe_grad = _avg_grad(self.features.named_parameters()) if any(p.requires_grad for p in self.features.parameters()) else None
                #     cl_grad = _avg_grad(self.classifier.named_parameters()) if any(p.requires_grad for p in self.classifier.parameters()) else None
                #     print_log(f"Batch {batch_idx} avg grad | features: {fe_grad} | classifier: {cl_grad}", log_mode=self.log_mode)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                print("Actual labels: ", labels)
                print("Predicted labels: ", predicted)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation phase
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for _, _, _, _, normalized_transformed_images, extra_features, labels in tqdm(
                    val_dataloader,
                    desc=f"Epoch {epoch+1}/{self.num_epochs} - Validation",
                ):
                    normalized_transformed_images = normalized_transformed_images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Only move extra_features to device if we're using them
                    if self.use_extra_features:
                        extra_features = extra_features.to(self.device)

                    # Use raw images for validation (matches notebook approach)
                    outputs = self(normalized_transformed_images, None)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_dataloader)
            avg_val_loss = val_loss / len(val_dataloader)

            print_log(
                f"Epoch {epoch+1}/{self.num_epochs}:",
                log_mode=self.log_mode,
            )
            print_log(
                f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%",
                log_mode=self.log_mode,
            )
            print_log(
                f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%",
                log_mode=self.log_mode,
            )

            # Early stopping and checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop:
                    print_log(
                        f"Early stopping triggered after {epoch+1} epochs",
                        log_mode=self.log_mode,
                    )
                    break

    def test_model(
        self,
        test_dataloader: Iterable,
    ):
        """Test the supervised model and return labels and predictions."""

        print_log(
            "Testing supervised model", log_mode=self.log_mode
        )

        self.eval()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for _, _, _, _, normalized_transformed_images, extra_features, labels in tqdm(test_dataloader, desc="Testing"):
                normalized_transformed_images = normalized_transformed_images.to(self.device)
                labels = labels.to(self.device)
                
                # Only move extra_features to device if we're using them
                if self.use_extra_features:
                    extra_features = extra_features.to(self.device)

                # Use raw images for testing (matches notebook approach)
                outputs = self(normalized_transformed_images, None)
                _, predicted = torch.max(outputs.data, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        return all_labels, all_preds

    def forward(self, x_image: torch.Tensor, x_features: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the supervised model.
        
        Args:
            x_image: Image tensor of shape (batch_size, channels, height, width)
            x_features: Additional features tensor of shape (batch_size, feature_dim)
            
        Returns:
            Model output
        """
        # Notebook-style architecture
        x = self.features(x_image)
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
        return self.features(x_image)
    


if __name__ == "__main__":
    from utils import load_config
    from models.data.plaque_dataset import load_dataloaders
    config = load_config("configs", "supervised")
    model = SupervisedModel(config)
    train_dataloader, val_dataloader, test_dataloader, unlabeled_dataloader = load_dataloaders(config)
    image_path, raw_image, transformed_image, normalized_raw_image, normalized_transformed_image, extra_features, labels = next(iter(train_dataloader))
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

