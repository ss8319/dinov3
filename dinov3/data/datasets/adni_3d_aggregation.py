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
        # Get the 3D volume and label from base dataset
        # We need to access the volume directly before transforms
        import nibabel as nib
        import os
        
        # Get image path and label directly from base dataset
        image_relpath = self.base_dataset.get_image_relpath(index)
        target = self.base_dataset.get_target(index)
        image_path = os.path.join(self.base_dataset.root, image_relpath)
        
        # Load NIfTI volume
        nii_img = nib.load(image_path)
        volume = nii_img.get_fdata().astype(np.float32)
        
        # If 4D, take first volume
        if volume.ndim == 4:
            volume = volume[..., 0]
        
        # Normalize volume to [0, 255] range
        volume_min, volume_max = volume.min(), volume.max()
        if volume_max > volume_min:
            volume = 255.0 * (volume - volume_min) / (volume_max - volume_min)
        
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
            
            # Convert grayscale to 3-channel RGB by replication
            slice_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1).astype(np.uint8)
            
            # Convert to PIL Image
            slice_pil = Image.fromarray(slice_rgb, mode='RGB')
            
            # Apply transform if provided
            if self.transform is not None:
                slice_tensor = self.transform(slice_pil)
            else:
                slice_tensor = torch.from_numpy(np.array(slice_pil)).permute(2, 0, 1)
            
            slices.append(slice_tensor)
        
        return slices, target


def extract_features_with_slice_aggregation(
    model: torch.nn.Module,
    dataset,  # ADNI dataset
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = "cuda",
    slice_axis: int = 0,
    stride: int = 2,  # Use every 2nd slice to reduce computation
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
    import nibabel as nib
    import os
    from torch.utils.data import DataLoader
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    model.eval()
    model = model.to(device)
    
    # Get transform from dataset if available
    transform = dataset.transform if hasattr(dataset, 'transform') else None
    
    logger.info(f"Extracting features with slice aggregation: axis={slice_axis}, stride={stride}")
    logger.info(f"Processing {len(dataset)} volumes...")
    
    # Pre-load all volumes into memory to avoid repeated I/O
    logger.info("Pre-loading volumes into memory...")
    volumes_data = []
    labels_data = []
    
    # Use ThreadPoolExecutor for parallel I/O
    def load_volume(idx):
        image_relpath = dataset.get_image_relpath(idx)
        label = dataset.get_target(idx)
        image_path = os.path.join(dataset.root, image_relpath)
        nii_img = nib.load(image_path)
        volume = nii_img.get_fdata().astype(np.float32)
        return (volume, label)
    
    with ThreadPoolExecutor(max_workers=max(1, num_workers * 2)) as executor:
        results = list(executor.map(load_volume, range(len(dataset))))
    
    volumes_data, labels_data = zip(*results)
    logger.info(f"Loaded {len(volumes_data)} volumes into memory")
    
    all_features = []
    all_labels = []
    
    # Process volumes
    for vol_idx, (volume, label) in enumerate(zip(volumes_data, labels_data)):
        # If 4D, take first volume
        if volume.ndim == 4:
            volume = volume[..., 0]
        
        # Normalize volume to [0, 255] range
        volume_min, volume_max = volume.min(), volume.max()
        if volume_max > volume_min:
            volume = 255.0 * (volume - volume_min) / (volume_max - volume_min)
        
        # Extract slices along specified axis
        num_slices = volume.shape[slice_axis]
        slice_indices = range(0, num_slices, stride)
        
        slice_tensors = []
        
        # Process slices
        for slice_idx in slice_indices:
            # Extract 2D slice
            if slice_axis == 0:
                slice_2d = volume[slice_idx, :, :]
            elif slice_axis == 1:
                slice_2d = volume[:, slice_idx, :]
            else:  # axis == 2
                slice_2d = volume[:, :, slice_idx]
            
            # Convert grayscale to 3-channel RGB
            slice_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1).astype(np.uint8)
            
            # Convert to PIL Image
            slice_pil = Image.fromarray(slice_rgb, mode='RGB')
            
            # Apply transform
            if transform is not None:
                slice_tensor = transform(slice_pil)
            else:
                slice_tensor = torch.from_numpy(np.array(slice_pil)).permute(2, 0, 1).float() / 255.0
            
            slice_tensors.append(slice_tensor)
        
        # Stack all slices and process through model
        if len(slice_tensors) > 0:
            slice_batch = torch.stack(slice_tensors).to(device)  # (num_slices, C, H, W)
            
            with torch.no_grad():
                slice_features = model(slice_batch)  # (num_slices, feature_dim)
            
            # Aggregate via mean pooling across slices
            volume_feature = slice_features.mean(dim=0)  # (feature_dim,)
            
            all_features.append(volume_feature.cpu())
            all_labels.append(label)
        
        if (vol_idx + 1) % 10 == 0:
            logger.info(f"Processed {vol_idx + 1}/{len(dataset)} volumes")
    
    features = torch.stack(all_features)
    labels = torch.tensor(all_labels)
    
    logger.info(f"Extracted aggregated features: {features.shape}, labels: {labels.shape}")
    
    return features, labels

