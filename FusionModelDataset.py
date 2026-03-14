import os
from typing import Tuple, List
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Statistics ---
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IR_MEAN  = np.array([0.3616], dtype=np.float32)
IR_STD   = np.array([0.0765], dtype=np.float32)

class FusionModelDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        have_label: bool = True,
        rgb_size: Tuple[int, int] = (480, 640),
        ir_size: Tuple[int, int] = (480, 640),
        ignore_index: int = 255,
        use_weather_aug: bool = False, # Bunu True yapmayı düşünebilirsiniz
    ):
        super().__init__()
        assert split in ["train","train_" ,"train_balanced", "val", "test", "test_day", "test_night","train_balanced_2to1","train_mixed","train_boost",'train_guardrail_night_only','train_guardrail_augmented','train_guardrail_copypaste'], 'Invalid split'
        self.data_dir = data_dir
        self.split = split
        self.have_label = have_label
        self.rgb_size = rgb_size
        self.ir_size = ir_size
        self.ignore_index = ignore_index

        with open(os.path.join(data_dir, f"{split}.txt"), "r") as f:
            self.names: List[str] = [line.strip() for line in f if line.strip()]
        self.n_data = len(self.names)

        # ============================================================
        # 1. ORTAK GEOMETRİK DÖNÜŞÜMLER (Spatial Augmentation)
        # ============================================================
        # RGB, IR ve Maske aynı şekilde dönmeli/kırpılmalı
        if split in ["train", "train_", "train_balanced", "train_balanced_2to1", "train_mixed", "train_boost", "train_guardrail_night_only", "train_guardrail_augmented", "train_guardrail_copypaste"]:
            self.shared_geom = A.Compose([
                A.HorizontalFlip(p=0.5),
                
                # --- GÜÇLENDİRME 1: Daha Agresif Scale ve Rotate ---
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.5, # 0.1 -> 0.5 yaptık (Çok önemli!)
                    rotate_limit=15, # 10 -> 15
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=self.ignore_index,
                    p=0.5, # Olasılık arttı
                ),
                
                # --- GÜÇLENDİRME 2: Geometrik Bozulmalar (Yol şekli için) ---
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    A.Perspective(scale=(0.05, 0.1), p=1.0),
                ], p=0.3),

                # --- GÜÇLENDİRME 3: CoarseDropout (Cutout etkisi) ---
                # Modelin eksik veriye karşı dirençli olmasını sağlar.
                # RGB ve IR üzerinde delikler açar.
                A.CoarseDropout(
                    max_holes=8, 
                    max_height=64, 
                    max_width=64, 
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0, 
                    mask_fill_value=self.ignore_index, 
                    p=0.3
                ),
            ], additional_targets={"ir": "image"})
        else:
            self.shared_geom = A.Compose([], additional_targets={"ir": "image"})

        # ============================================================
        # 2. RESIZE & FORMAT
        # ============================================================
        self.resize_rgb = A.Resize(self.rgb_size[0], self.rgb_size[1], interpolation=cv2.INTER_LINEAR)
        self.resize_ir  = A.Resize(self.ir_size[0], self.ir_size[1], interpolation=cv2.INTER_LINEAR)
        self.resize_mask = A.Resize(self.ir_size[0], self.ir_size[1], interpolation=cv2.INTER_NEAREST)

        # ============================================================
        # 3. RENK/PIKSEL DÖNÜŞÜMLERİ (Color Augmentation)
        # ============================================================
        
        # --- RGB İÇİN ---
        rgb_pipeline = []
        if split in ["train", "train_", "train_balanced", "train_balanced_2to1", "train_mixed", "train_boost", "train_guardrail_night_only", "train_guardrail_augmented", "train_guardrail_copypaste"]:
            rgb_pipeline.extend([
                # Gürültü ve Bulanıklık
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),  # 50 -> 80 (daha güçlü)
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.3),  # 0.2 -> 0.3 (daha sık)
                
                
                # Renk Oynamaları (GÜÇLENDİRİLDİ - Day/Night için)
                A.ColorJitter(
                    brightness=0.4,  # 0.2 -> 0.4 (daha agresif)
                    contrast=0.4,    # 0.2 -> 0.4
                    saturation=0.3,  # 0.2 -> 0.3
                    hue=0.15,        # 0.1 -> 0.15
                    p=0.6            # 0.4 -> 0.6 (daha sık uygula)
                ),
                
                # ENHANCED: Daha Agresif Gece Simülasyonu (test %100 gece!)
                A.OneOf([
                    # Option 1: Very strong darkening
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.7, 0.0),  # Çok karanlık
                        contrast_limit=(-0.4, 0.2),
                        p=1.0
                    ),
                    # Option 2: Strong darkening
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.5, 0.1),  # Karanlık bias
                        contrast_limit=(-0.2, 0.3),
                        p=1.0
                    ),
                    # Option 3: Moderate night
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.3, 0.05),
                        contrast_limit=0.2,
                        p=1.0
                    ),
                ], p=0.5),  # %50 şans ile bir gece simülasyonu uygula

                
                # CLAHE (Kontrast Dengeleme - Gece sahneleri için iyi)
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            ])
            
            if use_weather_aug:
                rgb_pipeline.append(
                    A.OneOf([
                        A.RandomRain(p=1),
                        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1),
                        A.RandomShadow(p=1), # MFNet gündüz/gece karışık, gölge iyidir
                    ], p=0.2)
                )

        # Normalize ve Tensor
        rgb_pipeline.extend([
            A.Normalize(mean=RGB_MEAN, std=RGB_STD),
            ToTensorV2()
        ])
        self.rgb_transform = A.Compose(rgb_pipeline)

        # --- IR İÇİN (YENİLİK!) ---
        # IR görüntüleri de bozulabilir, sadece normalize yetmez.
        ir_pipeline = []
        if split == "train" or split == "train_" or split == "train_balanced" or split == "train_balanced_2to1":
            ir_pipeline.extend([
                # ENHANCED: IR channel da gece için güçlendirildi
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),  # 0.2->0.3, 0.4->0.5
                A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),  # 50->60, p=0.2->0.3, IR sensör gürültüsü
            ])
        
        ir_pipeline.extend([
            A.Normalize(mean=IR_MEAN, std=IR_STD),
            ToTensorV2()
        ])
        self.ir_transform = A.Compose(ir_pipeline)

    def _read_image_4ch(self, name: str) -> np.ndarray:
        # Check original folder first
        path = os.path.join(self.data_dir, "images", f"{name}.png")
        if not os.path.exists(path):
            # Check augmented folder
            path = os.path.join(self.data_dir, "images_guardrail_aug", f"{name}.png")
        if not os.path.exists(path):
            # Check copypaste folder
            path = os.path.join(self.data_dir, "images_guardrail_copypaste", f"{name}.png")
        img = np.asarray(Image.open(path))
        if img.ndim == 2: raise ValueError(f"Image {path} is single-channel")
        return img[:, :, :4].copy() if img.shape[2] >= 4 else img

    def _read_mask(self, name: str) -> np.ndarray:
        # Check original folder first
        path = os.path.join(self.data_dir, "labels", f"{name}.png")
        if not os.path.exists(path):
            # Check augmented folder
            path = os.path.join(self.data_dir, "labels_guardrail_aug", f"{name}.png")
        if not os.path.exists(path):
            # Check copypaste folder
            path = os.path.join(self.data_dir, "labels_guardrail_copypaste", f"{name}.png")
        mask = np.asarray(Image.open(path))
        return mask.astype(np.int32).copy()

    def __getitem__(self, index: int):
        name = self.names[index]
        img4 = self._read_image_4ch(name)
        rgb  = img4[..., :3].astype(np.uint8)
        ir   = img4[..., 3:]

        if self.have_label and self.split in ["train", "train_", "train_balanced", "train_balanced_2to1", "train_mixed", "train_boost", "train_guardrail_night_only", "train_guardrail_augmented", "train_guardrail_copypaste", "val", "test", "test_day", "test_night"]:
            mask = self._read_mask(name)
        else:
            mask = None

        # 1. Ortak Geometrik Dönüşüm (Shared Geom)
        # Resize'dan ÖNCE yapılmalı ki kayıp olmasın
        if mask is not None:
            aug = self.shared_geom(image=rgb, ir=ir, mask=mask)
            rgb, ir, mask = aug["image"], aug["ir"], aug["mask"]
        else:
            aug = self.shared_geom(image=rgb, ir=ir)
            rgb, ir = aug["image"], aug["ir"]

        # 2. Resize (Her biri kendi boyutuna)
        rgb = self.resize_rgb(image=rgb)["image"]
        ir  = self.resize_ir(image=ir)["image"]
        if mask is not None:
            mask = self.resize_mask(image=mask)["image"]

        # 3. Renk ve Tensör Dönüşümü (Bağımsız)
        rgb = self.rgb_transform(image=rgb)["image"]
        ir  = self.ir_transform(image=ir)["image"]

        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.long)
            return rgb, ir, mask, name
        else:
            return rgb, ir, name

    def __len__(self) -> int:
        return self.n_data