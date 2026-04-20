"""PyTorch ``Dataset`` classes for plaque crops, transforms, normalization, and optional preloading."""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as trf
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


class PlaqueDatasetAugmented(torch.utils.data.Dataset):
    """Wraps several :class:`PlaqueDataset` clones plus one raw-tensor dataset for augmented indexing."""

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
        number_of_augmentations: int = 1,
        exclude_raw_images: bool = False,
    ):
        """
        1. Build ``number_of_augmentations`` dataset clones sharing the same transforms pipeline.
        2. Append one extra dataset using only ``ToTensor`` for a non-augmented branch.

        Args:
            data_df: Metadata table with image keys and labels.
            data_folder_path: Root directory containing class subfolders of PNGs.
            name_to_label: Mapping from string label to integer class id.
            transforms: Single ``Compose`` or list of compose pipelines (one clone per augmentation).
            preload: Whether underlying datasets preload rows into memory.
            description: Progress-bar label for preloading.
            normalize_data: If True, apply mean/std normalization in children.
            normalize_mean, normalize_std: Per-channel stats tensors (shape ``[C]``).
            use_extra_features: Pass through to children for Roundness/Area z-scores.
            downscaled_image_size: Resize target ``(H, W)``.
            downscaling_method: ``bilinear`` or ``nearest``.
            number_of_augmentations: How many augmented dataset instances to stack.

        Returns:
            None.
        """
        self.transforms = transforms
        self.number_of_augmentations = number_of_augmentations
        self.exclude_raw_images = exclude_raw_images
        self.plaque_datasets = [
            PlaqueDataset(
                data_df=data_df,
                data_folder_path=data_folder_path,
                name_to_label=name_to_label,
                transforms=transforms,
                preload=preload,
                apply_transforms_on_the_fly=apply_transforms_on_the_fly,
                description=description,
                normalize_data=normalize_data,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
                use_extra_features=use_extra_features,
                downscaled_image_size=downscaled_image_size,
                downscaling_method=downscaling_method,
            )
            for _ in range(number_of_augmentations)
        ]
        if not exclude_raw_images:
            self.plaque_datasets.append(
                PlaqueDataset(
                    data_df=data_df,
                    data_folder_path=data_folder_path,
                    name_to_label=name_to_label,
                    transforms=trf.ToTensor(),
                    preload=preload,
                    apply_transforms_on_the_fly=apply_transforms_on_the_fly,
                    description=description,
                    normalize_data=normalize_data,
                    normalize_mean=normalize_mean,
                    normalize_std=normalize_std,
                    use_extra_features=use_extra_features,
                    downscaled_image_size=downscaled_image_size,
                    downscaling_method=downscaling_method,
                )
        )

    def __len__(self):
        """
        Returns:
            ``(number_of_augmentations + 1)`` times the length of one underlying dataset.
        """
        return len(self.plaque_datasets[0]) * (self.number_of_augmentations + (1 if not self.exclude_raw_images else 0))

    def __getitem__(self, idx: int):
        """
        1. Map flat ``idx`` to a dataset index and within-dataset sample index.
        2. Return path, first transformed view tensor, extras, and label.

        Args:
            idx: Linear index over the virtual concatenation of child datasets.

        Returns:
            Tuple ``(image_path, tensor, extra_features, label)``.
        """
        dataset_idx = idx // (len(self.plaque_datasets[0]))
        transform_idx = idx % (len(self.plaque_datasets[0]))
        image_path, _, normalized_transformed_image_tensors, extra_features, label = (
            self.plaque_datasets[dataset_idx][transform_idx]
        )
        is_transformed = dataset_idx < self.number_of_augmentations
        return (
            image_path,
            is_transformed,
            normalized_transformed_image_tensors[0],
            extra_features,
            label,
        )


class PlaqueDataset(torch.utils.data.Dataset):
    """Loads plaque PNGs from disk, resizes, optionally preloads, and returns multi-view tensors."""

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
        """
        1. Store dataframe path, label maps, transform list, and normalization settings.
        2. If ``preload``, iterate all rows and cache :meth:`_process_row` outputs.

        Args:
            data_df: Table with ``Label``, ``Image``, ``Index``, and optional shape features.
            data_folder_path: Directory containing one subfolder per label name.
            name_to_label: Label string to class index; may be empty if filled externally.
            transforms: Single ``Compose`` or list of composes (multi-view).
            preload: Cache processed tensors in RAM.
            apply_transforms_on_the_fly: When preloaded, re-apply transforms each ``__getitem__`` if True.
            description: tqdm description during preload.
            normalize_data: Enable :meth:`_normalize_tensor` when stats are set.
            normalize_mean, normalize_std: Per-channel stats; if any missing, normalization is skipped.
            use_extra_features: Standardize ``Roundness`` and ``Area`` into a 2-D vector.
            downscaled_image_size: Target ``(H, W)`` after resize.
            downscaling_method: PIL resize filter name.

        Returns:
            None.
        """
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
        self.normalize_mean = (
            torch.tensor(normalize_mean) if normalize_mean is not None else None
        )
        self.normalize_std = (
            torch.tensor(normalize_std) if normalize_std is not None else None
        )
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
        """
        Returns:
            Number of rows in ``data_df``.
        """
        return len(self.data_df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        1. Load from cache or call :meth:`_process_row` for index ``idx``.
        2. If preloaded and ``apply_transforms_on_the_fly``, re-stack transformed views from the raw tensor.

        Args:
            idx: Row index into ``data_df``.

        Returns:
            ``(image_path, normalized_raw_image_tensor, normalized_transformed_image_tensors, extra_features, label)``.
        """
        if self.preload and self._preloaded_data is not None:
            (
                image_path,
                raw_image_tensor,
                normalized_raw_image_tensor,
                normalized_transformed_image_tensors,
                extra_features,
                label,
            ) = self._preloaded_data[idx]
            # If we apply transforms on the fly, recompute transformed and its normalized variant now
            if self.transforms and self.apply_transforms_on_the_fly:
                normalized_transformed_image_tensors = torch.stack(
                    [
                        self._normalize_tensor(
                            transform(to_pil_image(raw_image_tensor))
                        )
                        for transform in self.transforms
                    ]
                )
        else:
            (
                image_path,
                _,
                normalized_raw_image_tensor,
                normalized_transformed_image_tensors,
                extra_features,
                label,
            ) = self._process_row(idx)
        return (
            image_path,
            normalized_raw_image_tensor,
            normalized_transformed_image_tensors,
            extra_features,
            label,
        )

    def _process_row(
        self, idx: int, apply_transform: bool = True
    ) -> Tuple[
        str,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """
        1. Resolve the PNG path from the dataframe row.
        2. Load, resize, and tensorize the image; optionally apply each transform and stack.
        3. Normalize the raw tensor; build standardized extra features when enabled.
        4. Map the string label to an integer (or ``-1`` if unknown).

        Args:
            idx: Row index in ``data_df``.
            apply_transform: If False, return an empty transform stack tensor.

        Returns:
            ``(path, raw_tensor, normalized_raw, transformed_stack, extra_features, label)``.
        """
        row = self.data_df.iloc[idx]
        image_path = os.path.join(
            self.data_folder_path,
            row["Label"] if pd.notna(row["Label"]) else "",
            f"{row['Image'].replace('.hdf5', '')}_index_{row['Index']}.png",
        )

        if self.downscaling_method == "bilinear":
            raw_image_pil = (
                Image.open(image_path)
                .convert("RGB")
                .resize(self.downscaled_image_size, Image.BILINEAR)
            )
        elif self.downscaling_method == "nearest":
            raw_image_pil = (
                Image.open(image_path)
                .convert("RGB")
                .resize(self.downscaled_image_size, Image.NEAREST)
            )
        else:
            raise ValueError(
                f"Invalid downscaling method: {self.downscaling_method}. It should be either 'bilinear' or 'nearest'."
            )
        raw_image_tensor = trf.ToTensor()(raw_image_pil)
        # Ensure transform receives the correct input type (Tensor or PIL as expected)
        if self.transforms and apply_transform:
            normalized_transformed_image_tensors = torch.stack(
                [
                    self._normalize_tensor(transform(raw_image_pil))
                    for transform in self.transforms
                ]
            )
        else:
            normalized_transformed_image_tensors = torch.empty(0, dtype=torch.float32)
        normalized_raw_image_tensor = self._normalize_tensor(raw_image_tensor)
        if self.use_extra_features:
            extra_features = torch.tensor(
                [row["Roundness"], row["Area"]], dtype=torch.float32
            )
            extra_features_mean = torch.tensor(
                [self.data_df["Roundness"].mean(), self.data_df["Area"].mean()],
                dtype=torch.float32,
            )
            extra_features_std = torch.tensor(
                [self.data_df["Roundness"].std(), self.data_df["Area"].std()],
                dtype=torch.float32,
            )
            extra_features = (extra_features - extra_features_mean) / extra_features_std
        else:
            extra_features = torch.empty(0, dtype=torch.float32)
        return (
            image_path,
            raw_image_tensor,
            normalized_raw_image_tensor,
            normalized_transformed_image_tensors,
            extra_features,
            self.name_to_label.get(row["Label"], -1),
        )

    def _normalize_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        1. If normalization is disabled or stats are missing, return ``image_tensor`` unchanged.
        2. Otherwise subtract ``mean`` and divide by ``std`` with broadcast shape ``[C,1,1]``.

        Args:
            image_tensor: ``(C, H, W)`` float tensor.

        Returns:
            Normalized tensor with same shape and dtype-promoted stats.
        """
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
    # Add src directory to path when running directly
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    print("Running plaque_dataset.py visualization sample")
    from models.config import Config
    from utils.data_utils import load_data_df

    config = Config.load_config("configs", "supervised")

    # Load data and create splits (using the config to locate paths and parameters)
    data_df_path = os.path.join(
        config.general_config.data.data_folder,
        config.general_config.data.data_table_file_name,
    )
    labeled_data_df, unlabeled_data_df = load_data_df(
        data_df_path=data_df_path,
        labeled_sample_size=config.general_config.data.labeled_sample_size,
        unlabeled_sample_size=config.general_config.data.unlabeled_sample_size,
        train_mode="supervised",
    )
    print("Loaded labeled_data_df shape: ", labeled_data_df.shape)

    # Location of image data folder
    labeled_data_folder_path = os.path.join(
        config.general_config.data.data_folder,
        config.general_config.data.labeled_data_folder,
    )

    # Just sample from the labeled dataset for visualization
    sample_indices = list(range(min(8, len(labeled_data_df))))
    sample_df = labeled_data_df.iloc[sample_indices]

    aug_transform = trf.Compose(
        [
            trf.RandomHorizontalFlip(p=0.5),
            trf.RandomVerticalFlip(p=0.5),
            trf.RandomRotation(degrees=(0, 90)),
            trf.ColorJitter(brightness=0.2, contrast=0.2),
            trf.ToTensor(),
        ]
    )

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
    import time

    t0 = time.time()
    ds = PlaqueDataset(
        sample_df,
        labeled_data_folder_path,
        name_to_label=config.name_to_label,
        transforms=aug_transform,
        preload=True,
        description="labeled images (aug)",
        normalize_data=False,
    )
    print(f"Time taken to load dataset: {time.time() - t0} seconds")
    import numpy as np

    LIMIT = min(8, len(ds))
    fig, axes = plt.subplots(LIMIT, 2, figsize=(8, LIMIT * 3))
    if LIMIT == 1:  # special case if only 1 image
        axes = np.expand_dims(axes, axis=0)
    for i in range(LIMIT):
        # Get raw (unaugmented) and augmented samples from the datasets
        t0 = time.time()
        (
            image_path,
            normalized_raw_image_tensor,
            normalized_transformed_image_tensors,
            extra_features,
            label,
        ) = ds[i]
        print(f"Time taken to get item: {time.time() - t0} seconds")
        normalized_transformed_image_tensor = normalized_transformed_image_tensors[0]
        # Move channel to last dimension for imshow
        normalized_raw_image_tensor = (
            normalized_raw_image_tensor.permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .clip(0, 1)
        )
        normalized_transformed_image_tensor = (
            normalized_transformed_image_tensor.permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .clip(0, 1)
        )
        axes[i, 0].imshow(normalized_raw_image_tensor)
        axes[i, 0].set_yticks([112], [f"Label: {label}"])
        axes[i, 0].set_title("Raw")
        axes[i, 1].imshow(normalized_transformed_image_tensor)
        axes[i, 1].set_yticks([112], [f"Label: {label}"])
        axes[i, 1].set_title("Transformed")
        for ax in axes[i]:
            ax.set_xticks([])
    plt.tight_layout()
    plt.show()
