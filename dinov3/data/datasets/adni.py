# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
ADNI Dataset for Alzheimer's Disease Classification

Dataset structure:
- CSV files: pat_id, label, Sex, Age
- Images: {root}/{pat_id}.nii.gz
- Labels: 0 = CN (Cognitively Normal), 1 = AD (Alzheimer's Disease)
"""

import logging
import os
from enum import Enum
from typing import Callable, Optional

import pandas as pd
import numpy as np

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def csv_filename(self) -> str:
        """Return the CSV filename for this split"""
        return f"ad_{self.value}.csv"


class NiftiImageDecoder:
    """Deprecated: image decoding is handled by the slice aggregation dataset."""
    def __init__(self):
        pass
    def decode(self, image_path: str) -> np.ndarray:
        raise RuntimeError("NiftiImageDecoder is deprecated in favor of MONAI pipeline")


class ADNI(ExtendedVisionDataset):
    """
    ADNI Dataset for Alzheimer's Disease binary classification.
    
    Args:
        split: Dataset split (TRAIN, VAL, or TEST)
        root: Root directory containing NIfTI images
        extra: Directory containing CSV files (ad_train.csv, ad_val.csv, ad_test.csv)
               Alias for csv_dir to match dinov3 loader's expected key
        csv_dir: Directory containing CSV files (ad_train.csv, ad_val.csv, ad_test.csv)
        transforms: Optional transforms to apply to both image and target
        transform: Optional transform to apply to image only
        target_transform: Optional transform to apply to target only
    """
    
    Target = _Target
    Split = _Split
    
    def __init__(
        self,
        *,
        split: "ADNI.Split",
        root: str,
        extra: Optional[str] = None,
        csv_dir: Optional[str] = None,
        csv_filename: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        
        self._split = split
        # Accept either csv_dir or extra (dinov3 loader provides "extra")
        csv_root = csv_dir if csv_dir is not None else extra
        if csv_root is None:
            raise ValueError("ADNI requires 'csv_dir' or 'extra' to point to the CSV directory")
        self._csv_dir = csv_root
        
        # Allow custom CSV filename, otherwise use default from split
        csv_file = csv_filename if csv_filename is not None else split.csv_filename
        
        # Load CSV file
        csv_path = os.path.join(self._csv_dir, csv_file)
        logger.info(f"Loading ADNI {split.value} split from {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self._df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self._df)} samples for {split.value} split")
        
        # Validate required columns
        required_cols = ['pat_id', 'label']
        missing_cols = [col for col in required_cols if col not in self._df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        # Log class distribution
        class_counts = self._df['label'].value_counts().to_dict()
        logger.info(f"Class distribution: CN (0)={class_counts.get(0, 0)}, AD (1)={class_counts.get(1, 0)}")
        
        self._entries = None
    
    @property
    def split(self) -> "ADNI.Split":
        return self._split
    
    def _get_entries(self):
        """Lazy load entries"""
        if self._entries is None:
            self._entries = []
            for idx, row in self._df.iterrows():
                pat_id = str(row['pat_id'])
                label = int(row['label'])
                
                # Construct image path
                image_relpath = f"{pat_id}.nii.gz"
                
                self._entries.append((image_relpath, label))
            
            logger.info(f"Created {len(self._entries)} entries for {self.split.value} split")
        
        return self._entries
    
    def get_image_relpath(self, index: int) -> str:
        """Get relative path to image file"""
        entries = self._get_entries()
        image_relpath, _ = entries[index]
        return image_relpath
    
    def get_target(self, index: int) -> _Target:
        """Get label for sample"""
        entries = self._get_entries()
        _, label = entries[index]
        return label
    
    def get_targets(self) -> np.ndarray:
        """Get all labels as numpy array"""
        entries = self._get_entries()
        return np.array([label for _, label in entries], dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self._get_entries())
    
    def __getitem__(self, index: int):
        """This dataset now serves as an index provider; use SliceAggregationDataset for I/O."""
        raise RuntimeError("ADNI.__getitem__ is not supported; use SliceAggregationDataset for loading")

