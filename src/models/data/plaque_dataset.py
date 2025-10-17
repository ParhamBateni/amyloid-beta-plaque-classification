import torch
import pandas as pd
import os
from typing import Dict, Optional, Tuple, Union

from torchvision import transforms as trf
from PIL import Image
from tqdm import tqdm
import sys
from models.config import Config
from typing import List
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

class PlaqueDatasetAugmented(torch.utils.data.Dataset):
    def __init__(self, data_df: pd.DataFrame, data_folder_path: str, name_to_label: Dict[str, int] = {}, transforms: Union[trf.Compose, List[trf.Compose]] = None, preload: bool = False, description: str = "Plaque images", normalize_data: bool = True, normalize_mean: Optional[torch.Tensor] = None, normalize_std: Optional[torch.Tensor] = None, use_extra_features: bool = False, downscaled_image_size: Tuple[int, int] = (224, 224), downscaling_method: str = "bilinear", number_of_augmentations: int = 1):
        self.transforms = transforms
        self.number_of_augmentations = number_of_augmentations
        self.plaque_datasets = [PlaqueDataset(data_df=data_df, data_folder_path=data_folder_path, name_to_label=name_to_label, transforms=transforms, preload=preload, apply_transforms_on_the_fly=False, description=description, normalize_data=normalize_data, normalize_mean=normalize_mean, normalize_std=normalize_std, use_extra_features=use_extra_features, downscaled_image_size=downscaled_image_size, downscaling_method=downscaling_method) for _ in range(number_of_augmentations)]
        self.plaque_datasets.append(PlaqueDataset(data_df=data_df, data_folder_path=data_folder_path, name_to_label=name_to_label, transforms=trf.ToTensor(), preload=preload, apply_transforms_on_the_fly=True, description=description, normalize_data=normalize_data, normalize_mean=normalize_mean, normalize_std=normalize_std, use_extra_features=use_extra_features, downscaled_image_size=downscaled_image_size, downscaling_method=downscaling_method))
    
    def __len__(self):
        return len(self.plaque_datasets[0])*(self.number_of_augmentations+1)
    
    def __getitem__(self, idx: int):
        dataset_idx = idx//(len(self.plaque_datasets[0]))
        transform_idx = idx%(len(self.plaque_datasets[0]))
        image_path, _, normalized_transformed_image_tensors, extra_features, label = self.plaque_datasets[dataset_idx][transform_idx]
        return image_path, normalized_transformed_image_tensors[0], extra_features, label



# PlaqueDataset
class PlaqueDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_df: pd.DataFrame,
        data_folder_path: str,
        name_to_label: Dict[str, int] = {},
        transforms: Union[trf.Compose, List[trf.Compose]] = None,
        preload: bool = False,
        apply_transforms_on_the_fly: bool = False,
        description: str = "Plaque images",
        normalize_data: bool = True,
        normalize_mean: Optional[torch.Tensor] = None,
        normalize_std: Optional[torch.Tensor] = None,
        use_extra_features: bool = False,
        downscaled_image_size: Tuple[int, int] = (224, 224),
        downscaling_method: str = "bilinear",
    ):
        self.data_df = data_df
        self.data_folder_path = data_folder_path
        # build the name_to_label dictionary if it is not provided impute the labels using scikit-learn
        self.name_to_label = name_to_label if name_to_label is not None else {}
        self.label_to_name = {v: k for k, v in self.name_to_label.items()}
        if isinstance(transforms, List):
            self.transforms = transforms
        else:
            self.transforms = [transforms]

        self.preload = preload
        # If preload is True, apply_transforms_on_the_fly determines whether to apply the transform on the fly or not
        self.apply_transforms_on_the_fly = apply_transforms_on_the_fly
        # store normalization stats (expected shape [C])
        self.normalize_data = normalize_data
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.downscaled_image_size = downscaled_image_size
        self.downscaling_method = downscaling_method
        self.use_extra_features = use_extra_features
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
            image_path, raw_image_tensor, normalized_raw_image_tensor, normalized_transformed_image_tensors, extra_features, label = self._preloaded_data[idx]
            # If we apply transforms on the fly, recompute transformed and its normalized variant now
            if self.transforms and self.apply_transforms_on_the_fly:
                normalized_transformed_image_tensors = torch.stack([self._normalize_tensor(transform(to_pil_image(raw_image_tensor))) for transform in self.transforms])
        else:
            image_path, _, normalized_raw_image_tensor, normalized_transformed_image_tensors, extra_features, label = self._process_row(idx)
        return (image_path, normalized_raw_image_tensor, normalized_transformed_image_tensors, extra_features, label)

    def _process_row(self, idx: int, apply_transform: bool = True) -> Tuple[
        str,
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

        if self.downscaling_method == "bilinear":
            raw_image_pil = Image.open(image_path).convert("RGB").resize(self.downscaled_image_size, Image.BILINEAR)
        elif self.downscaling_method == "nearest":
            raw_image_pil = Image.open(image_path).convert("RGB").resize(self.downscaled_image_size, Image.NEAREST)
        else:
            raise ValueError(f"Invalid downscaling method: {self.downscaling_method}. It should be either 'bilinear' or 'nearest'.")
        raw_image_tensor = trf.ToTensor()(raw_image_pil)
        # Ensure transform receives the correct input type (Tensor or PIL as expected)
        if self.transforms and apply_transform:
            normalized_transformed_image_tensors = torch.stack([self._normalize_tensor(transform(raw_image_pil)) for transform in self.transforms])
        else:
            normalized_transformed_image_tensors = torch.empty(0, dtype=torch.float32)
        normalized_raw_image_tensor = self._normalize_tensor(raw_image_tensor)
        return (
            image_path,
            raw_image_tensor,
            normalized_raw_image_tensor,
            normalized_transformed_image_tensors,
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


# def load_labeled_dataloaders(train_labeled_data_df: pd.DataFrame, test_labeled_data_df: pd.DataFrame, val_labeled_data_df: pd.DataFrame, train_transforms: List[trf.Compose], val_transforms: List[trf.Compose], test_transforms: List[trf.Compose], config: Config) -> List[torch.utils.data.DataLoader]:
#     normalize_mean = torch.tensor(
#         config.general_config.data.normalize_mean, dtype=torch.float32
#     )
#     normalize_std = torch.tensor(
#         config.general_config.data.normalize_std, dtype=torch.float32
#     )
#     # Augmentations only; normalization is applied inside the dataset and returned separately
#     labeled_data_folder_path = os.path.join(
#         config.general_config.data.data_folder,
#         config.general_config.data.labeled_data_folder,
#     ) 
#     train_labeled_dataset = PlaqueDataset(
#         train_labeled_data_df,
#         labeled_data_folder_path,
#         name_to_label=config.name_to_label,
#         transforms=train_transforms,
#         preload=config.general_config.data.preload_data,
#         apply_transforms_on_the_fly=config.general_config.data.apply_transforms_on_the_fly,
#         description="train labeled plaque images",
#         normalize_data=config.general_config.data.normalize_data,
#         normalize_mean=normalize_mean,
#         normalize_std=normalize_std,
#         use_extra_features=config.general_config.data.use_extra_features,
#         downscaled_image_size=config.general_config.data.downscaled_image_size,
#         downscaling_method=config.general_config.data.downscaling_method,
#     )
#     test_labeled_dataset = PlaqueDataset(
#         test_labeled_data_df,
#         labeled_data_folder_path,
#         name_to_label=config.name_to_label,
#         transforms=test_transforms,
#         preload=config.general_config.data.preload_data,
#         apply_transforms_on_the_fly=config.general_config.data.apply_transforms_on_the_fly,
#         description="test labeled plaque images",
#         normalize_data=config.general_config.data.normalize_data,
#         normalize_mean=normalize_mean,
#         normalize_std=normalize_std,
#         use_extra_features=config.general_config.data.use_extra_features,
#         downscaled_image_size=config.general_config.data.downscaled_image_size,
#         downscaling_method=config.general_config.data.downscaling_method,
#     )
#     val_labeled_dataset = PlaqueDataset(
#         val_labeled_data_df,
#         labeled_data_folder_path,
#         name_to_label=config.name_to_label,
#         transforms=val_transforms,
#         preload=config.general_config.data.preload_data,
#         apply_transforms_on_the_fly=config.general_config.data.apply_transforms_on_the_fly,
#         description="val labeled plaque images",
#         normalize_data=config.general_config.data.normalize_data,
#         normalize_mean=normalize_mean,
#         normalize_std=normalize_std,
#         use_extra_features=config.general_config.data.use_extra_features,
#         downscaled_image_size=config.general_config.data.downscaled_image_size,
#         downscaling_method=config.general_config.data.downscaling_method,
#     )

#     train_labeled_dataloader = torch.utils.data.DataLoader(
#         train_labeled_dataset,
#         batch_size=config.general_config.data.batch_size,
#         shuffle=False,
#         num_workers=config.general_config.data.num_workers,
#         pin_memory=config.general_config.data.pin_memory,
#         persistent_workers=config.general_config.data.persistent_workers,
#     )
#     test_labeled_dataloader = torch.utils.data.DataLoader(
#         test_labeled_dataset,
#         batch_size=config.general_config.data.batch_size,
#         shuffle=False,
#         num_workers=config.general_config.data.num_workers,
#         pin_memory=config.general_config.data.pin_memory,
#         persistent_workers=config.general_config.data.persistent_workers,
#     )
#     val_labeled_dataloader = torch.utils.data.DataLoader(
#         val_labeled_dataset,
#         batch_size=config.general_config.data.batch_size,
#         shuffle=False,
#         num_workers=config.general_config.data.num_workers,
#         pin_memory=config.general_config.data.pin_memory,
#         persistent_workers=config.general_config.data.persistent_workers,
#     )
#     return (
#         train_labeled_dataloader,
#         val_labeled_dataloader,
#         test_labeled_dataloader,
#     )

# def load_unlabeled_dataloader(unlabeled_data_df: pd.DataFrame, unlabeled_transform: trf.Compose, config: Config) -> torch.utils.data.DataLoader:
#     if len(unlabeled_data_df) == 0:
#         return torch.utils.data.DataLoader([])
#     normalize_mean = torch.tensor(
#         config.general_config.data.normalize_mean, dtype=torch.float32
#     )
#     normalize_std = torch.tensor(
#         config.general_config.data.normalize_std, dtype=torch.float32
#     )
#     unlabeled_data_folder_path = os.path.join(
#         config.general_config.data.data_folder,
#         config.general_config.data.unlabeled_data_folder,
#     )
#     unlabeled_dataset = PlaqueDataset(
#         unlabeled_data_df,
#         unlabeled_data_folder_path,
#         name_to_label=config.name_to_label,
#         transforms=unlabeled_transform,
#         preload=config.general_config.data.preload_data,
#         apply_transforms_on_the_fly=config.general_config.data.apply_transforms_on_the_fly,
#         description="unlabeled plaque images",
#         normalize_data=config.general_config.data.normalize_data,
#         normalize_mean=normalize_mean,
#         normalize_std=normalize_std,
#         use_extra_features=config.general_config.data.use_extra_features,
#         downscaled_image_size=config.general_config.data.downscaled_image_size,
#         downscaling_method=config.general_config.data.downscaling_method,
#     )
#     unlabeled_dataloader = torch.utils.data.DataLoader(
#         unlabeled_dataset,
#         batch_size=config.general_config.data.batch_size,
#         shuffle=True,
#         num_workers=config.general_config.data.num_workers,
#         pin_memory=config.general_config.data.pin_memory,
#         persistent_workers=config.general_config.data.persistent_workers,
#     )
#     return unlabeled_dataloader



if __name__ == "__main__":
    print("Running plaque_dataset.py visualization sample")
    from utils import load_data_df
    config = Config.load_config("configs")

    # Load data and create splits (using the config to locate paths and parameters)
    labeled_data_df, unlabeled_data_df = load_data_df("supervised", config)
    print("Loaded labeled_data_df shape: ", labeled_data_df.shape)

    # Location of image data folder
    labeled_data_folder_path = os.path.join(
        config.general_config.data.data_folder, 
        config.general_config.data.labeled_data_folder
    )

    # Just sample from the labeled dataset for visualization
    sample_indices = list(range(min(8, len(labeled_data_df))))
    sample_df = labeled_data_df.iloc[sample_indices]

    aug_transform = trf.Compose([
        trf.RandomHorizontalFlip(p=0.5),
        trf.RandomVerticalFlip(p=0.5),
        trf.RandomRotation(degrees=(0, 90)),
        trf.ColorJitter(brightness=0.2, contrast=0.2),
        trf.ToTensor(),
    ])

    # ds = PlaqueDatasetAugmented(
    #     sample_df,
    #     labeled_data_folder_path,
    #     name_to_label=config.name_to_label,
    #     transforms=aug_transform,
    #     description="labeled images (aug)",
    #     normalize_data = False
    # )
    # i = 0
    # while i < len(ds):
    #     image_path, normalized_transformed_image_tensor, extra_features, label = ds[i]
    #     normalized_transformed_image_tensor = normalized_transformed_image_tensor.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
    #     plt.imshow(normalized_transformed_image_tensor)
    #     plt.show()
    #     i+=8

    ds = PlaqueDataset(
        sample_df,
        labeled_data_folder_path,
        name_to_label=config.name_to_label,
        transforms=aug_transform,
        description="labeled images (aug)",
        normalize_data = False
    )

    import numpy as np

    LIMIT = min(8, len(ds))
    fig, axes = plt.subplots(LIMIT, 2, figsize=(8, LIMIT * 3))
    if LIMIT == 1:  # special case if only 1 image
        axes = np.expand_dims(axes, axis=0)
    for i in range(LIMIT):
        # Get raw (unaugmented) and augmented samples from the datasets
        image_path, normalized_raw_image_tensor, normalized_transformed_image_tensors, extra_features, label = ds[i]
        normalized_transformed_image_tensor = normalized_transformed_image_tensors[0]
        # Move channel to last dimension for imshow
        normalized_raw_image_tensor = normalized_raw_image_tensor.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        normalized_transformed_image_tensor = normalized_transformed_image_tensor.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        axes[i, 0].imshow(normalized_raw_image_tensor)
        axes[i, 0].set_yticks([112], [f"Label: {label}"])
        axes[i, 0].set_title(f"Raw")
        axes[i, 1].imshow(normalized_transformed_image_tensor)
        axes[i, 1].set_yticks([112], [f"Label: {label}"])
        axes[i, 1].set_title(f"Transformed")
        for ax in axes[i]:
            ax.set_xticks([])
    plt.tight_layout()
    plt.show()
