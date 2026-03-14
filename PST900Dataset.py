import os
from typing import Tuple, List
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- PST900 Statistics (Computed from training set) ---
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet stats
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THERMAL_MEAN = np.array([0.2266], dtype=np.float32)  # Computed from PST900 train
THERMAL_STD  = np.array([0.2804], dtype=np.float32)

class PST900Dataset(Dataset):
    """
    PST900 RGB-Thermal Dataset for Indoor Semantic Segmentation
    
    5 Classes:
        0: Background
        1: Fire Extinguisher
        2: Backpack
        3: Hand Drill 
        4: Rescue Randy (mannequin)
    
    Dataset Structure:
        PST900_RGBT_Dataset/
        ├── train/
        │   ├── rgb/       # 1280x720 PNG (RGB)
        │   ├── thermal/   # 1280x720 PNG (grayscale)
        │   └── labels/    # 1280x720 PNG (uint8, values 0-4)
        └── test/
            └── (same structure)
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        have_label: bool = True,
        rgb_size: Tuple[int, int] = (720, 1280),  # Native PST900 resolution
        thermal_size: Tuple[int, int] = (720, 1280),
        ignore_index: int = 255,
        use_augmentation: bool = True,
    ):
        super().__init__()
        assert split in ["train", "test"], f'Invalid split: {split}. Use "train" or "test".'
        self.data_dir = data_dir
        self.split = split
        self.have_label = have_label
        self.rgb_size = rgb_size
        self.thermal_size = thermal_size
        self.ignore_index = ignore_index
        self.use_augmentation = use_augmentation and (split == "train")
        
        # --- Get File Names ---
        rgb_dir = os.path.join(data_dir, split, "rgb")
        if not os.path.exists(rgb_dir):
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
        
        # Extract base names (without extension)
        self.names: List[str] = [
            os.path.splitext(f)[0] for f in sorted(os.listdir(rgb_dir))
            if f.endswith('.png')
        ]
        self.n_data = len(self.names)
        print(f"[PST900Dataset] Loaded {self.n_data} samples from '{split}' split.")
        
        # ============================================================
        # 1. SHARED GEOMETRIC AUGMENTATION (RGB + Thermal + Mask)
        # ============================================================
        if self.use_augmentation:
            self.shared_geom = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                    scale=(0.7, 1.3),
                    rotate=(-10, 10),
                    mode=cv2.BORDER_CONSTANT,
                    cval=0,
                    cval_mask=self.ignore_index,
                    p=0.5,
                ),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                ], p=0.2),
                A.CoarseDropout(
                    num_holes_range=(1, 5),
                    hole_height_range=(20, 50),
                    hole_width_range=(20, 50),
                    fill_value=0,
                    p=0.2
                ),
            ], additional_targets={"thermal": "image"})
        else:
            self.shared_geom = A.Compose([], additional_targets={"thermal": "image"})
        
        # ============================================================
        # 2. RESIZE & FORMAT
        # ============================================================
        self.resize_rgb = A.Resize(self.rgb_size[0], self.rgb_size[1], interpolation=cv2.INTER_LINEAR)
        self.resize_thermal = A.Resize(self.thermal_size[0], self.thermal_size[1], interpolation=cv2.INTER_LINEAR)
        self.resize_mask = A.Resize(self.thermal_size[0], self.thermal_size[1], interpolation=cv2.INTER_NEAREST)
        
        # ============================================================
        # 3. COLOR AUGMENTATION (RGB only)
        # ============================================================
        rgb_pipeline = []
        if self.use_augmentation:
            rgb_pipeline.extend([
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), noise_scale_factor=1.0, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.2),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
            ])
        
        rgb_pipeline.extend([
            A.Normalize(mean=RGB_MEAN, std=RGB_STD),
            ToTensorV2()
        ])
        self.rgb_transform = A.Compose(rgb_pipeline)
        
        # ============================================================
        # 4. THERMAL AUGMENTATION
        # ============================================================
        thermal_pipeline = []
        if self.use_augmentation:
            thermal_pipeline.extend([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(var_limit=(5.0, 30.0), noise_scale_factor=1.0, p=0.2),
            ])
        
        thermal_pipeline.extend([
            A.Normalize(mean=THERMAL_MEAN, std=THERMAL_STD),
            ToTensorV2()
        ])
        self.thermal_transform = A.Compose(thermal_pipeline)
    
    def _read_rgb(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, self.split, "rgb", f"{name}.png")
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        return img
    
    def _read_thermal(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, self.split, "thermal", f"{name}.png")
        thermal = np.array(Image.open(path).convert('L'), dtype=np.uint8)  # Grayscale
        return thermal
    
    def _read_mask(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, self.split, "labels", f"{name}.png")
        mask = np.array(Image.open(path).convert('L'), dtype=np.int32)
        return mask
    
    def __getitem__(self, index: int):
        name = self.names[index]
        rgb = self._read_rgb(name)
        thermal = self._read_thermal(name)
        
        if self.have_label:
            mask = self._read_mask(name)
        else:
            mask = None
        
        # --- 1. Shared Geometric Augmentation ---
        if mask is not None:
            augmented = self.shared_geom(image=rgb, thermal=thermal, mask=mask)
            rgb = augmented['image']
            thermal = augmented['thermal']
            mask = augmented['mask']
        else:
            augmented = self.shared_geom(image=rgb, thermal=thermal)
            rgb = augmented['image']
            thermal = augmented['thermal']
        
        # --- 2. Resize ---
        rgb = self.resize_rgb(image=rgb)['image']
        thermal = self.resize_thermal(image=thermal)['image']
        if mask is not None:
            mask = self.resize_mask(image=mask)['image']
        
        # --- 3. RGB Color Augmentation + Normalization ---
        rgb = self.rgb_transform(image=rgb)['image']  # -> Tensor [3, H, W]
        
        # --- 4. Thermal Augmentation + Normalization ---
        # Thermal is grayscale [H, W], need to add channel dim for Albumentations
        thermal = thermal[..., np.newaxis]  # [H, W, 1]
        thermal = self.thermal_transform(image=thermal)['image']  # -> Tensor [1, H, W]
        
        # Keep thermal as 1-channel (model expects 1-channel IR input)
        # thermal stays as [1, H, W]
        
        # --- 5. Mask to Tensor ---
        if mask is not None:
            mask = torch.from_numpy(mask.copy()).long()
        else:
            mask = torch.zeros((self.rgb_size[0], self.rgb_size[1]), dtype=torch.long)
        
        return rgb, thermal, mask, name
    
    def __len__(self):
        return self.n_data


# ============================================================
# Utility Functions
# ============================================================
def get_pst900_palette():
    """
    Returns color palette for PST900 5-class segmentation.
    """
    return [
        [0, 0, 0],        # Class 0: Background (Black)
        [255, 0, 0],      # Class 1: Fire Extinguisher (Red)
        [0, 255, 0],      # Class 2: Backpack (Green)
        [0, 0, 255],      # Class 3: Hand Drill (Blue)
        [255, 255, 0],    # Class 4: Rescue Randy (Yellow)
    ]


if __name__ == "__main__":
    # Quick test
    print("Testing PST900Dataset...")
    print("Testing with native resolution (720, 1280)...")
    dataset = PST900Dataset(
        data_dir="/datavolume/data/emrecanitez/Datasets/PST900_RGBT_Dataset",
        split="train",
        rgb_size=(720, 1280),  # Native resolution
        thermal_size=(720, 1280),
    )
    
    print(f"Dataset length: {len(dataset)}")
    rgb, thermal, mask, name = dataset[0]
    print(f"Sample 0: {name}")
    print(f"  RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"  Thermal shape: {thermal.shape}, dtype: {thermal.dtype} (1-channel)")
    print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}, unique: {torch.unique(mask).tolist()}")
    assert thermal.shape[0] == 1, "Thermal should be 1-channel!"
    print("✅ PST900Dataset test passed!")
