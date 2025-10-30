# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
ADNI 3D Volume Dataset with Slice-wise Aggregation

This dataset handles 3D MRI volumes by:
1. Extracting all 2D slices along a specified axis
2. Converting each grayscale slice to 3-channel RGB
3. Processing each slice through a 2D encoder (like DINOv3)
4. Aggregating slice features via mean pooling

This allows 2D-native encoders to process 3D volumetric data.
"""

import logging
from typing import Callable, Optional, Tuple, List
import numpy as np
import torch
from PIL import Image
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
)

logger = logging.getLogger("dinov3")


class SliceAggregationDataset:
    """
    Wrapper dataset that extracts all slices from a 3D volume dataset.
    
    Instead of returning a single image-label pair, this returns:
    - A list of 2D slices (each converted to 3-channel RGB)
    - The volume label
    
    During feature extraction, each slice is processed independently through
    the encoder, and the resulting features are aggregated via mean pooling.
    
    Args:
        base_dataset: The underlying 3D volume dataset (e.g., ADNI)
        slice_axis: Which axis to slice along (0=sagittal, 1=coronal, 2=axial)
        stride: Slice stride (e.g., 2 = every other slice). Default: 1 (all slices)
        transform: Transform to apply to each 2D slice
    """
    
    def __init__(
        self,
        base_dataset,
        slice_axis: int = 0,
        stride: int = 1,
        transform: Optional[Callable] = None,
    ):
        self.base_dataset = base_dataset
        self.slice_axis = slice_axis
        self.stride = stride
        self.transform = transform

        # MONAI pipeline for 3D volumes: load -> orient -> normalize to [0,1]
        self._monai = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"],
                    lower=1.0,
                    upper=99.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )

        logger.info(f"SliceAggregationDataset: axis={slice_axis}, stride={stride}")
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def get_target(self, index: int) -> int:
        """Get label for the volume"""
        return self.base_dataset.get_target(index)
    
    def get_targets(self) -> np.ndarray:
        """Get all labels"""
        return self.base_dataset.get_targets()
    
    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], int]:
        """
        Returns:
            slices: List of 2D slice tensors (each processed through transform)
            target: Volume label
        """
        # Resolve image path and label from base dataset
        import os

        image_relpath = self.base_dataset.get_image_relpath(index)
        target = self.base_dataset.get_target(index)
        image_path = os.path.join(self.base_dataset.root, image_relpath)

        # Use MONAI to load and normalize the 3D volume to [0,1]
        data = {"image": image_path}
        data = self._monai(data)
        volume = data["image"]  # (C, D, H, W)

        # Remove channel dimension
        if volume.ndim == 4 and volume.shape[0] == 1:
            volume = volume[0]  # (D, H, W)
        
        # Extract slices along specified axis
        num_slices = volume.shape[self.slice_axis]
        slice_indices = range(0, num_slices, self.stride)
        
        slices = []
        for slice_idx in slice_indices:
            # Extract 2D slice
            if self.slice_axis == 0:
                slice_2d = volume[slice_idx, :, :]
            elif self.slice_axis == 1:
                slice_2d = volume[:, slice_idx, :]
            else:  # axis == 2
                slice_2d = volume[:, :, slice_idx]
            
            # Convert grayscale (float in [0,1]) to 3-channel RGB uint8
            slice_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)
            slice_rgb = np.clip(slice_rgb * 255.0, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            slice_pil = Image.fromarray(slice_rgb, mode='RGB')
            
            # Apply DINOv3 2D transform if provided; otherwise return CHW tensor
            if self.transform is not None:
                slice_tensor = self.transform(slice_pil)
            else:
                slice_tensor = torch.from_numpy(np.array(slice_pil)).permute(2, 0, 1)
            
            slices.append(slice_tensor)
        
        return slices, target


def extract_features_with_slice_aggregation(
    model: torch.nn.Module,
    dataset,  # SliceAggregationDataset instance
    batch_size: int = 1,  # unused; kept for API compatibility
    num_workers: int = 0,  # unused; kept for API compatibility
    device: str = "cuda",
    slice_axis: int = 0,  # unused; slicing handled by dataset
    stride: int = 2,  # unused; slicing handled by dataset
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from a 3D volume dataset using slice-wise aggregation.
    
    For each 3D volume:
    1. Extract all 2D slices along specified axis
    2. Convert each grayscale slice to 3-channel RGB
    3. Process each slice through the 2D encoder
    4. Aggregate slice features via mean pooling
    
    Args:
        model: The 2D encoder (e.g., DINOv3)
        dataset: ADNI dataset containing 3D volumes
        batch_size: Batch size for processing volumes
        num_workers: Number of data loading workers
        device: Device to use for inference
        slice_axis: Which axis to slice along (0=sagittal, 1=coronal, 2=axial)
        stride: Slice stride (e.g., 2 = every other slice)
    
    Returns:
        features: Aggregated features per volume (N, feature_dim)
        labels: Volume labels (N,)
    """
    model.eval()
    model = model.to(device)

    logger.info(f"Extracting features with slice aggregation using dataset pipeline")
    logger.info(f"Processing {len(dataset)} volumes...")

    all_features = []
    all_labels = []

    for idx in range(len(dataset)):
        slices, label = dataset[idx]

        if len(slices) == 0:
            continue

        slice_batch = torch.stack(slices).to(device)

        with torch.no_grad():
            slice_features = model(slice_batch)

        volume_feature = slice_features.mean(dim=0)
        all_features.append(volume_feature.cpu())
        all_labels.append(label)

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(dataset)} volumes")

    features = torch.stack(all_features)
    labels = torch.tensor(all_labels)

    logger.info(f"Extracted aggregated features: {features.shape}, labels: {labels.shape}")

    return features, labels

