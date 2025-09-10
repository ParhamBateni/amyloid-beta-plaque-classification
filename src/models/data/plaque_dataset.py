import torch
import pandas as pd
import os
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys
from sklearn.model_selection import train_test_split
from models.config import Config
from typing import List
import matplotlib.pyplot as plt
from utils import print_log
import numpy as np

from torchvision.transforms.functional import to_pil_image


# PlaqueDataset
class PlaqueDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_df: pd.DataFrame,
        data_folder_path: str,
        name_to_label: Dict[str, int] = None,
        transform: transforms.Compose = None,
        preload: bool = False,
        apply_transforms_on_the_fly: bool = True,
        description: str = "Plaque images",
        normalize_data: bool = True,
        normalize_mean: Optional[torch.Tensor] = None,
        normalize_std: Optional[torch.Tensor] = None,
        use_extra_features: bool = False,
        pixels_scale: int = 255,
        downscaled_image_size: Tuple[int, int] = (224, 224),
    ):
        self.data_df = data_df
        self.data_folder_path = data_folder_path
        # build the name_to_label dictionary if it is not provided impute the labels using scikit-learn
        self.name_to_label = (
            name_to_label
            if name_to_label is not None
            else {
                k: v
                for k, v in enumerate(
                    LabelEncoder().fit(data_df["Label"]).transform(data_df["Label"])
                )
            }
        )
        self.label_to_name = {v: k for k, v in self.name_to_label.items()}
        self.transform = transform
        self.preload = preload
        # If preload is True, apply_transforms_on_the_fly determines whether to apply the transform on the fly or not
        self.apply_transforms_on_the_fly = apply_transforms_on_the_fly
        # store normalization stats (expected shape [C])
        self.normalize_data = normalize_data
        self.normalize_mean = (
            normalize_mean.clone().detach() if normalize_mean is not None else None
        )
        self.normalize_std = (
            normalize_std.clone().detach() if normalize_std is not None else None
        )
        self.downscaled_image_size = downscaled_image_size
        self.use_extra_features = use_extra_features
        self.pixels_scale = pixels_scale
        self._preloaded_data = None
        if self.preload:
            self._preloaded_data = []
            for idx in tqdm(
                range(len(self.data_df)),
                desc=f"Preloading {description}...",
                file=sys.stdout,
            ):
                # When preloading, apply transforms only if not applying on the fly
                self._preloaded_data.append(
                    self._process_row(
                        idx, apply_transform=not self.apply_transforms_on_the_fly
                    )
                )

    def __len__(self):
        return len(self.data_df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
    ]:
        if self.preload and self._preloaded_data is not None:
            (
                image_path,
                scaled_raw_image_tensor,
                scaled_transformed_image_tensor,
                scaled_normalized_raw_image_tensor,
                scaled_normalized_transformed_image_tensor,
                extra_features,
                label,
            ) = self._preloaded_data[idx]
            # If we apply transforms on the fly, recompute transformed and its normalized variant now
            if self.transform and self.apply_transforms_on_the_fly:
                transformed_image_tensor = self.transform(to_pil_image(scaled_raw_image_tensor / self.pixels_scale))
                scaled_transformed_image_tensor = (
                    transformed_image_tensor * self.pixels_scale
                )
                scaled_normalized_transformed_image_tensor = (
                    self._normalize_tensor(transformed_image_tensor) * self.pixels_scale
                )
            return (
                image_path,
                scaled_raw_image_tensor,
                scaled_transformed_image_tensor,
                scaled_normalized_raw_image_tensor,
                scaled_normalized_transformed_image_tensor,
                extra_features,
                label,
            )
        return self._process_row(idx)

    def _process_row(self, idx: int, apply_transform: bool = True) -> Tuple[
        str,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        row = self.data_df.iloc[idx]
        image_path = os.path.join(
            self.data_folder_path,
            row["Label"] if pd.notna(row["Label"]) else "",
            f"{row['Image'].replace('.hdf5', '')}_index_{row['Index']}.png",
        )
        raw_image_pil = Image.open(image_path).convert("RGB").resize(self.downscaled_image_size, Image.BILINEAR)
        raw_image_tensor = transforms.ToTensor()(raw_image_pil)
        # Ensure transform receives the correct input type (Tensor or PIL as expected)
        if self.transform and apply_transform:
            transformed_image_tensor = self.transform(raw_image_pil)
        else:
            transformed_image_tensor = raw_image_tensor
        normalized_raw_image_tensor = self._normalize_tensor(raw_image_tensor)
        normalized_transformed_image_tensor = self._normalize_tensor(
            transformed_image_tensor
        )
        return (
            image_path,
            raw_image_tensor * self.pixels_scale,
            transformed_image_tensor * self.pixels_scale,
            normalized_raw_image_tensor * self.pixels_scale,
            normalized_transformed_image_tensor * self.pixels_scale,
            (
                torch.tensor([row["Roundness"], row["Area"]], dtype=torch.float32)
                if self.use_extra_features
                else torch.empty(0, dtype=torch.float32)
            ),
            self.name_to_label.get(row["Label"], -1),
        )

    def _normalize_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if (
            not self.normalize_data
            or self.normalize_mean is None
            or self.normalize_std is None
        ):
            return image_tensor
        # reshape mean/std to [C,1,1]
        mean = self.normalize_mean.view(-1, 1, 1).to(image_tensor.dtype)
        std = self.normalize_std.view(-1, 1, 1).to(image_tensor.dtype)
        return (image_tensor - mean) / std


# Load dataloaders
def load_dataloaders(
    config: Config,
) -> List[torch.utils.data.DataLoader]:
    data_df = pd.read_csv(
        os.path.join(
            config.general_config.data.data_folder,
            config.general_config.data.data_table_file_name,
        )
    )
    labeled_data_df = data_df[data_df["Label"].notna()]
    labeled_data_df = labeled_data_df.sample(
        n=min(config.general_config.data.labeled_sample_size, len(labeled_data_df)),
        random_state=config.general_config.system.random_seed,
        replace=False,
    )
    unlabeled_data_df = data_df[data_df["Label"].isna()]
    unlabeled_data_df = unlabeled_data_df.sample(
        n=min(
            config.general_config.data.unlabeled_sample_size,
            len(unlabeled_data_df),
        ),
        random_state=config.general_config.system.random_seed,
        replace=False,
    )

    normalize_mean = torch.tensor(
        config.general_config.data.normalize_mean, dtype=torch.float32
    )
    normalize_std = torch.tensor(
        config.general_config.data.normalize_std, dtype=torch.float32
    )

    train_labeled_data_df, test_labeled_data_df = train_test_split(
        labeled_data_df,
        test_size=config.general_config.data.test_size,
        random_state=config.general_config.system.random_seed,
        stratify=labeled_data_df["Label"],
    )
    train_labeled_data_df, val_labeled_data_df = train_test_split(
        train_labeled_data_df,
        test_size=config.general_config.data.val_size,
        random_state=config.general_config.system.random_seed,
        stratify=train_labeled_data_df["Label"],
    )

    # Augmentations only; normalization is applied inside the dataset and returned separately
    labeled_train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 90)),  # RandomRotate90
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2
            ),  # RandomBrightnessContrast
            transforms.ToTensor(),
        ]
    )
    labeled_val_transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    labeled_test_transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    labeled_data_folder_path = os.path.join(
        config.general_config.data.data_folder,
        config.general_config.data.labeled_data_folder,
    )
    train_labeled_dataset = PlaqueDataset(
        train_labeled_data_df,
        labeled_data_folder_path,
        name_to_label=config.name_to_label,
        transform=labeled_train_transform if config.general_config.data.transform_data else None,
        preload=config.general_config.data.preload_data,
        apply_transforms_on_the_fly=config.general_config.data.apply_transforms_on_the_fly,
        description="labeled plaque images",
        normalize_data=config.general_config.data.normalize_data,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        use_extra_features=config.general_config.data.use_extra_features,
        pixels_scale=config.general_config.data.pixels_scale,
        downscaled_image_size=config.general_config.data.downscaled_image_size,
    )
    test_labeled_dataset = PlaqueDataset(
        test_labeled_data_df,
        labeled_data_folder_path,
        name_to_label=config.name_to_label,
        transform=labeled_test_transform if config.general_config.data.transform_data else None,
        preload=config.general_config.data.preload_data,
        apply_transforms_on_the_fly=config.general_config.data.apply_transforms_on_the_fly,
        description="test labeled plaque images",
        normalize_data=config.general_config.data.normalize_data,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        use_extra_features=config.general_config.data.use_extra_features,
        pixels_scale=config.general_config.data.pixels_scale,
        downscaled_image_size=config.general_config.data.downscaled_image_size,
    )
    val_labeled_dataset = PlaqueDataset(
        val_labeled_data_df,
        labeled_data_folder_path,
        name_to_label=config.name_to_label,
        transform=labeled_val_transform if config.general_config.data.transform_data else None,
        preload=config.general_config.data.preload_data,
        apply_transforms_on_the_fly=config.general_config.data.apply_transforms_on_the_fly,
        description="val labeled plaque images",
        normalize_data=config.general_config.data.normalize_data,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        use_extra_features=config.general_config.data.use_extra_features,
        pixels_scale=config.general_config.data.pixels_scale,
        downscaled_image_size=config.general_config.data.downscaled_image_size,
    )

    train_labeled_dataloader = torch.utils.data.DataLoader(
        train_labeled_dataset,
        batch_size=config.general_config.data.batch_size,
        shuffle=False,
        num_workers=config.general_config.data.num_workers,
        pin_memory=config.general_config.data.pin_memory,
        persistent_workers=config.general_config.data.persistent_workers,
    )
    test_labeled_dataloader = torch.utils.data.DataLoader(
        test_labeled_dataset,
        batch_size=config.general_config.data.batch_size,
        shuffle=False,
        num_workers=config.general_config.data.num_workers,
        pin_memory=config.general_config.data.pin_memory,
        persistent_workers=config.general_config.data.persistent_workers,
    )
    val_labeled_dataloader = torch.utils.data.DataLoader(
        val_labeled_dataset,
        batch_size=config.general_config.data.batch_size,
        shuffle=False,
        num_workers=config.general_config.data.num_workers,
        pin_memory=config.general_config.data.pin_memory,
        persistent_workers=config.general_config.data.persistent_workers,
    )

    unlabeled_data_folder_path = os.path.join(
        config.general_config.data.data_folder,
        config.general_config.data.unlabeled_data_folder,
    )
    unlabeled_dataset = PlaqueDataset(
        unlabeled_data_df,
        unlabeled_data_folder_path,
        name_to_label=config.name_to_label,
        transform=labeled_val_transform if config.general_config.data.transform_data else None,
        preload=config.general_config.data.preload_data,
        apply_transforms_on_the_fly=config.general_config.data.apply_transforms_on_the_fly,
        description="unlabeled plaque images",
        normalize_data=config.general_config.data.normalize_data,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        use_extra_features=config.general_config.data.use_extra_features,
        pixels_scale=config.general_config.data.pixels_scale,
        downscaled_image_size=config.general_config.data.downscaled_image_size,
    )
    unlabeled_dataloader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=config.general_config.data.batch_size,
        shuffle=True,
        num_workers=config.general_config.data.num_workers,
        pin_memory=config.general_config.data.pin_memory,
        persistent_workers=config.general_config.data.persistent_workers,
    )

    return (
        train_labeled_dataloader,
        val_labeled_dataloader,
        test_labeled_dataloader,
        unlabeled_dataloader,
    )


if __name__ == "__main__":
    from utils import load_config

    config = load_config("configs", "supervised")
    (
        train_labeled_dataloader,
        val_labeled_dataloader,
        test_labeled_dataloader,
        unlabeled_dataloader,
    ) = load_dataloaders(config)
    print("train_labeled_dataloader number of batches: ", len(train_labeled_dataloader))
    print("val_labeled_dataloader number of batches: ", len(val_labeled_dataloader))
    print("test_labeled_dataloader number of batches: ", len(test_labeled_dataloader))
    print("unlabeled_dataloader number of batches: ", len(unlabeled_dataloader))

    # Show the first batch pictures and labels
    (
        image_paths,
        scaled_raw_images,
        scaled_transformed_images,
        scaled_normalized_raw_images,
        scaled_normalized_transformed_images,
        extra_features,
        labels,
    ) = next(iter(train_labeled_dataloader))
    print("scaled_raw_images shape: ", scaled_raw_images.shape)
    print("scaled_transformed_images shape: ", scaled_transformed_images.shape)
    print("scaled_normalized_raw_images shape: ", scaled_normalized_raw_images.shape)
    print("scaled_normalized_transformed_images shape: ", scaled_normalized_transformed_images.shape)
    print("extra_features shape: ", extra_features.shape)
    print("labels shape: ", labels.shape)
    print("example scaled_raw_image: ", scaled_raw_images[0])
    print("example scaled_transformed_image: ", scaled_transformed_images[0])
    print("average pixel value of scaled_transformed_image: ", scaled_transformed_images[0].mean())
    print("example scaled_normalized_raw_image: ", scaled_normalized_raw_images[0])
    print("example scaled_normalized_transformed_image: ", scaled_normalized_transformed_images[0])
    print("example extra_features: ", extra_features[0])
    print("example labels: ", labels[0])

    LIMIT = 8
    fig, ax = plt.subplots(min(LIMIT, len(scaled_raw_images)), 4, figsize=(20, 20))
    for i in range(min(LIMIT, len(scaled_raw_images))):
        print(f"image_paths {i}: {image_paths[i]}")
        # If pixel values are in [0, 255], convert to [0, 1] for imshow
        ax[i, 0].imshow((scaled_raw_images[i].permute(1, 2, 0) / config.general_config.data.pixels_scale).clip(0, 1))
        ax[i, 1].imshow((scaled_transformed_images[i].permute(1, 2, 0) / config.general_config.data.pixels_scale).clip(0, 1))
        ax[i, 2].imshow((scaled_normalized_raw_images[i].permute(1, 2, 0) / config.general_config.data.pixels_scale).clip(0, 1))
        ax[i, 3].imshow((scaled_normalized_transformed_images[i].permute(1, 2, 0) / config.general_config.data.pixels_scale).clip(0, 1))
        ax[i, 0].set_ylabel(f"{labels[i]}")
        ax[i, 1].set_ylabel(f"{labels[i]}")
        ax[i, 2].set_ylabel(f"{labels[i]}")
        ax[i, 3].set_ylabel(f"{labels[i]}")
        ax[i, 0].tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax[i, 1].tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax[i, 2].tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax[i, 3].tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        if i == min(LIMIT, len(scaled_raw_images)) - 1:
            ax[i, 0].set_xlabel("Raw Image")
            ax[i, 1].set_xlabel("Transformed Image")
            ax[i, 2].set_xlabel("Normalized Raw Image")
            ax[i, 3].set_xlabel("Normalized Transformed Image")
        plt.tight_layout()
    plt.show()
