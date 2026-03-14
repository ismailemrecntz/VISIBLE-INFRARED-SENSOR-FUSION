from html import parser
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
import argparse
import time
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
import wandb  # Optional: for experiment tracking

# Import your models
from FusionModel import FusionModel
from FusionModelDataset import FusionModelDataset
from PST900Dataset import PST900Dataset
from FusionModelUtils import compute_results
from sklearn.metrics import confusion_matrix


# ---- EMA (Exponential Moving Average) ----
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # unwrap DP
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.detach().clone().to(p.device)
    def update(self, model):
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if name in self.shadow and p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}
    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state, device):
        self.shadow = {}
        for k, v in state.items():
            self.shadow[k] = v.clone().to(device)

# ---- Lovasz + Dice + CE (label smoothing) ----
def lovasz_softmax(logits, labels, classes='present', eps=1e-6):
    # logits: [B,C,H,W] (ham), labels: [B,H,W]
    probs = torch.softmax(logits.float(), dim=1)
    B, C, H, W = probs.shape
    losses = []
    for c in range(C):
        if classes == 'present' and (labels == c).sum() == 0:
            continue
        fg = (labels == c).float()
        pc = probs[:, c, ...]
        errors = (fg - pc).abs().reshape(-1)
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = fg.reshape(-1)[perm]
        grad = torch.cumsum(gt_sorted, 0) / (gt_sorted.sum() + eps)
        grad = 1.0 - grad
        losses.append(torch.dot(errors_sorted, grad) / (H * W))
    return torch.stack(losses).mean() if len(losses) else probs.new_tensor(0.)

# ---- Focal Loss for Hard Examples ----
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Focuses training on hard, misclassified examples.
    """
    def __init__(self, gamma=2.0, alpha=None, target_class=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.target_class = target_class
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        p = torch.exp(-ce_loss)  # probability
        
        # Focal weight: (1 - p)^gamma - focuses on hard examples
        focal_weight = (1 - p) ** self.gamma
        
        # If target_class specified (>= 0), only apply to that class
        if self.target_class is not None and self.target_class >= 0:
            mask = (targets == self.target_class).float()
            focal_loss = focal_weight * ce_loss * mask
            return focal_loss.sum() / (mask.sum() + 1e-8)
        else:
            # Apply to all classes (target_class is None or -1)
            focal_loss = focal_weight * ce_loss
            if self.alpha is not None:
                if isinstance(self.alpha, (list, np.ndarray)):
                    alpha_t = torch.tensor(self.alpha, device=logits.device)[targets]
                    focal_loss = alpha_t * focal_loss
            return focal_loss.mean()

class ComboLoss3(nn.Module):
    def __init__(self, ce_w=0.6, dice_w=0.2, lovasz_w=0.2,
                 class_weights=None, ignore_index=None, label_smoothing=0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.ce_w, self.dice_w, self.lovasz_w = ce_w, dice_w, lovasz_w

    def dice(self, logits, y, eps=1e-6):
        logits = logits.float()
        C = logits.shape[1]
        y1h = F.one_hot(y.clamp_min(0), C).permute(0,3,1,2).float()
        p = torch.softmax(logits, dim=1)

        inter = (p * y1h).sum(dim=(0,2,3))
        union = p.sum(dim=(0,2,3)) + y1h.sum(dim=(0,2,3))

        dice = (2*inter + eps) / (union + eps)

        if self.ce.weight is not None:
            dice = dice * self.ce.weight.to(dice.device)

        return 1 - dice.mean()

    def forward(self, logits, y):
        # CE + Dice + Lovasz (kaybı fp32'de hesaplayacağız)
        return (self.ce_w * self.ce(logits, y)
              + self.dice_w * self.dice(logits, y)
              + self.lovasz_w * lovasz_softmax(logits, y))


# ---- OHEM (Online Hard Example Mining) Loss ----
class OHEMCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining - focuses on hardest pixels.
    Improves rare class detection by mining difficult examples.
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
    
    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W]
        B, C, H, W = logits.shape
        
        # Calculate per-pixel loss
        loss = self.ce(logits, targets)  # [B, H, W]
        
        # Flatten
        loss_flat = loss.view(-1)
        
        # Get valid mask (not ignore_index)
        valid_mask = (targets != self.ignore_index).view(-1)
        loss_flat = loss_flat[valid_mask]
        
        if loss_flat.numel() == 0:
            return loss.mean()
        
        # Sort losses descending
        loss_sorted, _ = torch.sort(loss_flat, descending=True)
        
        # Find threshold: keep pixels with loss > thresh OR top min_kept
        num_pixels = loss_flat.numel()
        min_kept = min(self.min_kept, num_pixels)
        
        # Use probability threshold
        probs = torch.softmax(logits, dim=1)
        max_probs, _ = probs.max(dim=1)  # [B, H, W]
        max_probs_flat = max_probs.view(-1)[valid_mask]
        
        # Pixels where max_prob < thresh are "hard"
        hard_mask = max_probs_flat < self.thresh
        num_hard = hard_mask.sum().item()
        
        # Keep at least min_kept or all hard examples
        k = max(min_kept, num_hard)
        k = min(k, num_pixels)
        
        # Get top-k losses
        threshold_loss = loss_sorted[int(k) - 1] if k > 0 else loss_sorted[0]
        
        # Select hard examples
        hard_loss = loss_flat[loss_flat >= threshold_loss]
        
        return hard_loss.mean()


# ---- Boundary Loss for Edge Detection ----
class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss to improve edge detection.
    Critical for thin objects like guardrail.
    """
    def __init__(self, num_classes=9, ignore_index=255, kernel_size=3):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.kernel_size = kernel_size
        
        # Laplacian kernel for edge detection
        laplacian = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian', laplacian)
    
    def get_boundary(self, mask):
        """Extract boundary from segmentation mask"""
        B, H, W = mask.shape
        
        # One-hot encode
        one_hot = F.one_hot(mask.clamp(0, self.num_classes - 1), self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        boundaries = []
        for c in range(self.num_classes):
            class_mask = one_hot[:, c:c+1, :, :]  # [B, 1, H, W]
            # Apply Laplacian to detect edges
            edges = F.conv2d(class_mask, self.laplacian, padding=1)
            edges = (edges.abs() > 0.1).float()
            boundaries.append(edges)
        
        # Combine all class boundaries
        boundary = torch.cat(boundaries, dim=1).max(dim=1, keepdim=True)[0]
        return boundary.squeeze(1)  # [B, H, W]
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] model predictions
            targets: [B, H, W] ground truth
        """
        # Get ground truth boundaries
        gt_boundary = self.get_boundary(targets)  # [B, H, W]
        
        # Get predicted boundaries from softmax
        probs = torch.softmax(logits, dim=1)
        
        # Compute boundary from predictions
        pred_boundary = torch.zeros_like(gt_boundary)
        for c in range(self.num_classes):
            class_prob = probs[:, c:c+1, :, :]
            edges = F.conv2d(class_prob, self.laplacian.to(class_prob.device), padding=1)
            pred_boundary = torch.max(pred_boundary, edges.abs().squeeze(1))
        
        # Scale logits (don't apply sigmoid - will be done in BCE with logits)
        pred_boundary = pred_boundary * 5.0
        
        # Binary cross entropy on boundaries
        valid_mask = (targets != self.ignore_index).float()
        
        # Focus more on GT boundary regions
        boundary_weight = 1.0 + 2.0 * gt_boundary  # 3x weight on boundaries
        
        # Use BCE with logits (autocast-safe)
        loss = F.binary_cross_entropy_with_logits(
            pred_boundary,
            gt_boundary,
            weight=boundary_weight * valid_mask,
            reduction='sum'
        )
        
        return loss / (valid_mask.sum() + 1e-6)


# ---- Enhanced ComboLoss with OHEM and Boundary ----
class ComboLossOHEM(nn.Module):
    """
    Enhanced loss combining: OHEM CE + Dice + Lovasz + Boundary
    Optimized for rare class detection (guardrail, car_stop, bump)
    """
    def __init__(self, ce_w=0.4, dice_w=0.2, lovasz_w=0.2, ohem_w=0.1, boundary_w=0.1,
                 class_weights=None, ignore_index=255, num_classes=9,
                 ohem_thresh=0.7, ohem_min_kept=100000,
                 focal_weight=0.0, focal_gamma=2.0, focal_target_class=6):
        super().__init__()
        
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.ohem = OHEMCrossEntropyLoss(
            ignore_index=ignore_index, 
            thresh=ohem_thresh, 
            min_kept=ohem_min_kept,
            weight=class_weights
        )
        self.boundary = BoundaryLoss(num_classes=num_classes, ignore_index=ignore_index)
        
        self.ce_w = ce_w
        self.dice_w = dice_w
        self.lovasz_w = lovasz_w
        self.ohem_w = ohem_w
        self.boundary_w = boundary_w
        self.focal_w = focal_weight
        
        # Initialize Focal Loss if enabled
        if focal_weight > 0:
            self.focal_loss = FocalLoss(gamma=focal_gamma, target_class=focal_target_class, ignore_index=ignore_index)
            print(f"[ComboLossOHEM] Weights: CE={ce_w}, Dice={dice_w}, Lovasz={lovasz_w}, OHEM={ohem_w}, Boundary={boundary_w}, Focal={focal_weight}")
            print(f"[ComboLossOHEM] Focal: gamma={focal_gamma}, target_class={focal_target_class} (guardrail)")
        else:
            self.focal_loss = None
            print(f"[ComboLossOHEM] Weights: CE={ce_w}, Dice={dice_w}, Lovasz={lovasz_w}, OHEM={ohem_w}, Boundary={boundary_w}")
    
    def dice(self, logits, y, eps=1e-6):
        logits = logits.float()
        C = logits.shape[1]
        y1h = F.one_hot(y.clamp_min(0), C).permute(0,3,1,2).float()
        p = torch.softmax(logits, dim=1)
        
        inter = (p * y1h).sum(dim=(0,2,3))
        union = p.sum(dim=(0,2,3)) + y1h.sum(dim=(0,2,3))
        
        dice = (2*inter + eps) / (union + eps)
        return 1 - dice.mean()
    
    def forward(self, logits, y):
        loss = (self.ce_w * self.ce(logits, y)
              + self.dice_w * self.dice(logits, y)
              + self.lovasz_w * lovasz_softmax(logits, y)
              + self.ohem_w * self.ohem(logits, y)
              + self.boundary_w * self.boundary(logits, y))
        
        # Add Focal Loss if enabled (for guardrail targeting)
        if self.focal_loss is not None:
            loss = loss + self.focal_w * self.focal_loss(logits, y)
            
        return loss

# =============================================================================
# Training Components
# =============================================================================

class FusionTrainer:
    """
    Trainer class for RGB-IR Fusion model.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds
        self.set_seed(config.seed)
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.setup_model()
        
        # Setup data
        self.train_loader, self.val_loader = self.setup_data()
        
        # Setup training components
        self.criterion_main, self.criterion_aux = self.setup_loss()
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        self.ema = ModelEMA(self.model, decay=self.config.ema_decay) if self.config.ema else None       
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.start_epoch = 1
        self.best_miou = 0.0
        self.best_loss = float('inf')
        
        # Load checkpoint if resuming
        if config.resume == True and config.resume_with_reset == False:
            self.load_checkpoint()

        elif config.resume == True and config.resume_with_reset == True:
            self.resume_from_best_with_reset()
    
    def set_seed(self, seed):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_directories(self):
        """Create necessary directories"""
        self.exp_dir = Path(self.config.exp_dir) / self.config.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.exp_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # Save config
        config_path = self.exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=4)
        
        print(f"Experiment directory: {self.exp_dir}")
    
    def setup_model(self):
        """Initialize the fusion model"""
        ctx_dim = [int(x.strip()) for x in self.config.context_dim.strip('[]').split(',')]
        resolution = (self.config.img_height, self.config.img_width)
        
        # Get decoder configuration
        decoder_type = getattr(self.config, 'decoder_type', 'fpn')
        deep_supervision = getattr(self.config, 'deep_supervision', True)
        
        # Get fusion configuration
        enhanced_fusion = getattr(self.config, 'enhanced_fusion', False)
        use_safd = getattr(self.config, 'use_safd', False)
        use_cafg = getattr(self.config, 'use_cafg', False)
        use_tpsw = getattr(self.config, 'use_tpsw', False)
        
        model = FusionModel(
            rgb_arch=self.config.rgb_arch,
            ir_arch=self.config.ir_arch,
            num_classes=self.config.num_classes,
            input_resolution=resolution,
            rgb_backbone_resolution=resolution,  # Match input resolution
            ir_backbone_resolution=resolution,   # Match input resolution
            context_dim=ctx_dim,
            output_resolution=resolution,  # Output at same resolution as input
            decoder_type=decoder_type,
            deep_supervision=deep_supervision,
            enhanced_fusion=enhanced_fusion,
            use_safd=use_safd,
            use_cafg=use_cafg,
            use_tpsw=use_tpsw
        )
        
        model = model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1 and self.config.multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        if self.config.freeze_backbones:
            for name, p in model.named_parameters():
                if ('rgb_encoder' in name) or ('ir_encoder' in name):
                    p.requires_grad = False

        return model
    
    def setup_data(self):
        """Setup data loaders"""
        # Create datasets based on selected dataset type
        if self.config.dataset == 'pst900':
            # PST900 native resolution is 720x1280, but can be configured
            resolution = (self.config.img_height, self.config.img_width)
            print(f"[INFO] Using PST900 RGB-Thermal Dataset (5 classes)")
            print(f"[INFO] Resolution: {resolution[0]}x{resolution[1]} (HxW)")
            train_dataset = PST900Dataset(
                data_dir=self.config.data_root,
                split='train',
                rgb_size=resolution,
                thermal_size=resolution,
                use_augmentation=True
            )
            val_dataset = PST900Dataset(
                data_dir=self.config.data_root,
                split='test',
                rgb_size=resolution,
                thermal_size=resolution,
                use_augmentation=False
            )
        elif self.config.dataset == 'mfnet':
            print(f"[INFO] Using MFNet RGB-IR Dataset (9 classes)")
            train_dataset = FusionModelDataset(self.config.data_root, self.config.train_source, have_label=True)
            val_dataset  = FusionModelDataset(self.config.data_root, 'test', have_label=True)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}. Choose 'mfnet' or 'pst900'.")
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Train iterations per epoch: {len(train_loader)}")
        
        return train_loader, val_loader
    
    def setup_loss(self):
        """Setup loss functions"""
        # Main loss
        if self.config.class_weights:
            # Calculate or load class weights
            class_weights = self.calculate_class_weights()
        else:
            class_weights = None
        
        ignore_idx = 0 if self.config.ignore_unlabeled else -100

        if self.config.loss_type == 'combo3':
            criterion_main = ComboLoss3(
                ce_w=self.config.ce_weight,
                dice_w=self.config.dice_weight,
                lovasz_w=self.config.lovasz_weight,
                class_weights=class_weights,
                ignore_index=ignore_idx,
                label_smoothing=self.config.label_smoothing
            ).to(self.device)
        elif self.config.loss_type == 'combo_ohem':
            # Auto-detect target class for Focal Loss if not specified
            focal_target = getattr(self.config, 'focal_target_class', -1)
            if focal_target == -1:
                # Auto-select based on dataset
                if self.config.dataset == 'pst900':
                    focal_target = 3  # Hand Drill
                elif self.config.dataset == 'mfnet':
                    focal_target = 6  # Guardrail
                else:
                    focal_target = None  # Apply to all classes
            elif focal_target == -2:
                focal_target = None  # Explicitly apply to all classes
                
            criterion_main = ComboLossOHEM(
                ce_w=self.config.ce_weight,
                dice_w=self.config.dice_weight,
                lovasz_w=self.config.lovasz_weight,
                ohem_w=self.config.ohem_weight,
                boundary_w=self.config.boundary_weight,
                class_weights=class_weights,
                ignore_index=ignore_idx,
                num_classes=self.config.num_classes,
                ohem_thresh=self.config.ohem_thresh,
                ohem_min_kept=self.config.ohem_min_kept,
                focal_weight=getattr(self.config, 'focal_weight', 0.0),
                focal_gamma=getattr(self.config, 'focal_gamma', 2.0),
                focal_target_class=focal_target
            ).to(self.device)
            print(f"[INFO] Using ComboLossOHEM with OHEM threshold={self.config.ohem_thresh}")
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Auxiliary loss
        criterion_aux = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index =ignore_idx
        ).to(self.device)
        
        return criterion_main, criterion_aux
    
    def calculate_class_weights(self):
        print("Calculating class weights...")
        class_counts = torch.zeros(self.config.num_classes, device=self.device)

        for _, _, mask, _ in tqdm(self.train_loader, desc="Computing class weights"):
            mask = mask.to(self.device)
            valid = (mask >= 0) & (mask < self.config.num_classes)
            mask = mask[valid]
            class_counts += torch.bincount(mask, minlength=self.config.num_classes)

        freq = class_counts / class_counts.sum()

        print("Class pixel frequencies:")
        for i,f in enumerate(freq):
            print(f"  Class {i}: {f:.6f}")

        # --- BALANCED INVERSE FREQUENCY WEIGHTING ---
        # This gives high weights to rare classes but keeps it balanced
        
        # Step 1: Calculate inverse frequency weights
        # weight[i] = total_pixels / (num_classes * class_pixels[i])
        total_pixels = class_counts.sum()
        num_classes = self.config.num_classes
        weights = total_pixels / (num_classes * class_counts.clamp(min=1))
        
        # Step 2: Normalize so max weight = 1.0
        weights = weights / weights.max()
        
        # Step 3: Scale up (configurable via --class_weight_multiplier)
        # Default: 10.0 for balanced, 5.0 for full diversity
        weight_multiplier = getattr(self.config, 'class_weight_multiplier', 10.0)
        weights = weights * weight_multiplier
        
        # Step 4: Clamp to reasonable range
        # Min 0.5 (for very common classes), Max 10.0 (for very rare)
        weights = torch.clamp(weights, 0.5, 15.0)

        # PST900-specific: Boost Hand Drill (Class 3) - rarest and hardest class
        if self.config.dataset == 'pst900' and self.config.num_classes == 5:
            print("[INFO] Applying PST900-specific boost for Hand Drill (Class 3)...")
            weights[3] *= 1.3  # Hand Drill boost (reduced from 1.5 to 1.3 for stability)
            print(f"  Hand Drill weight boosted: {weights[3]:.2f}")

        # MFNet-specific: Boost weak classes (Car Stop, Guardrail, Bump)
        if self.config.dataset == 'mfnet' and self.config.num_classes == 9:
            print("[INFO] Applying MFNet-specific boosts for weak classes...")
            weights[5] *= 1.4  # Car Stop boost
            weights[6] *= 2.0  # Guardrail boost (very rare)
            weights[8] *= 1.3  # Bump boost
            print(f"  Car Stop (5) weight boosted: {weights[5]:.2f}")
            print(f"  Guardrail (6) weight boosted: {weights[6]:.2f}")
            print(f"  Bump (8) weight boosted: {weights[8]:.2f}")

        # Note: Manual weight overrides removed for multi-dataset compatibility
        # If needed for specific datasets, add conditional logic based on num_classes

        print("Final Class Weights:")
        for i,w in enumerate(weights):
            print(f"  Class {i}: {w:.4f}")

        return weights
    
    def setup_optimizer(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        base_lr = self.config.lr_backbone
        decay = 0.90

        param_groups = {}

        def get_rgb_layer_id(name):
            if "stem" in name: return 0
            if "stages_0" in name: return 1
            if "stages_1" in name: return 2
            if "stages_2" in name: return 3
            if "stages_3" in name: return 4
            return 5

        def get_ir_layer_id(name):
            if "conv_stem" in name: return 0
            if "blocks.0" in name: return 1
            if "blocks.1" in name: return 2
            if "blocks.2" in name: return 3
            if "blocks.3" in name: return 4
            if "blocks.4" in name: return 5
            return 6

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith("rgb_encoder"):
                layer_id = get_rgb_layer_id(name)
                lr = base_lr * (decay ** (5 - layer_id))
                wd = 0.05
                key = f"rgb_{layer_id}"

            elif name.startswith("ir_encoder"):
                layer_id = get_ir_layer_id(name)
                lr = base_lr * (decay ** (5 - layer_id))
                wd = 0.03
                key = f"ir_{layer_id}"

            elif "fusion_stage" in name:
                lr = self.config.lr_fusion
                wd = 0.01
                key = "fusion"

            elif "thermal_prior" in name:
                lr = self.config.lr_fusion
                wd = 0.01
                key = "thermal_prior"

            elif "decoder" in name:
                lr = self.config.lr_decoder
                wd = 0.005
                key = "decoder"

            else:
                lr = base_lr * 2.0
                wd = 0.005
                key = "other"

            if key not in param_groups:
                param_groups[key] = {
                    "params": [],
                    "lr": lr,
                    "weight_decay": wd,
                    "name": key
                }

            param_groups[key]["params"].append(param)


        # Create optimizer
        if self.config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                list(param_groups.values()),
                betas=(0.9, 0.999), 
        )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def setup_scheduler(self):
        """Setup learning rate scheduler - MORE STABLE"""
        if self.config.scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=1e-7)
        elif self.config.scheduler == 'cosine_restart':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=self.config.t0, T_mult=2, eta_min=1e-7)    
        elif self.config.scheduler == 'poly':
            from torch.optim.lr_scheduler import PolynomialLR
            scheduler = PolynomialLR(self.optimizer, total_iters=self.config.epochs, power=0.9)
        elif self.config.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  
                factor=0.7,
                patience=20,
                min_lr=1e-7,
                verbose=True
            )
        elif self.config.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )
        elif self.config.scheduler == 'cosine_warmup':
            warmup_epochs = self.config.warmup_epochs
            flat_epochs = self.config.flat_epochs

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                elif epoch < warmup_epochs + flat_epochs:
                    return 1.0
                else:
                    return 0.5 ** ((epoch - warmup_epochs - flat_epochs) / 15)

            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
        else:
            scheduler = None
        
        return scheduler
    
    def setup_logging(self):
        """Setup tensorboard and wandb logging"""
        # Tensorboard
        self.writer = SummaryWriter(self.log_dir)
        # Plain text logging
        self.log_path = self.exp_dir / 'log.txt'
        # Open in append mode so resume runs don't overwrite
        self._log_file = open(self.log_path, 'a', buffering=1)
        # Weights & Biases (optional)
        if self.config.use_wandb:
            wandb.init(
                project="rgb-ir-fusion",
                name=self.config.exp_name,
                config=self.config
            )
            wandb.watch(self.model)

        # initial log
        # use helper below (safe because it's a class method)
        # but call after writer/file are set
        try:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._log_file.write(f'[{ts}] Experiment started: {self.exp_dir}\n')
        except Exception:
            pass

    def log(self, msg, print_console=True):
        """Helper to write a timestamped message to log.txt and optionally print."""
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}\n'
        try:
            self._log_file.write(line)
        except Exception:
            # fallback: print to console if file write fails
            print('Log write failed; message:', line)
        if print_console:
            print(line, end='')
    
    def get_aux_weight(self, epoch):
        """Calculate auxiliary loss weight based on epoch"""
        if epoch < self.config.aux_weight_decay_epoch:
            alpha = epoch / self.config.aux_weight_decay_epoch
            return (1 - alpha) * self.config.aux_weight_start + alpha * self.config.aux_weight_end
        return self.config.aux_weight_end


    def train_epoch(self, epoch):
        """Train with gradient accumulation for stability"""
        self.model.train()
        accumulation_steps = max(1, int(self.config.grad_accum_steps))
        
        # Metrics tracking
        running_loss = 0.0
        running_main_loss = 0.0
        running_aux_loss = 0.0
        num_batches = 0
        
        # Get aux weight
        aux_weight = self.get_aux_weight(epoch)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.epochs}')
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (rgb, ir, masks,_) in enumerate(pbar):
            # Forward pass
            if masks.max() >= self.config.num_classes:
                print(f"\nWarning: Found invalid labels in batch {batch_idx}")
                print(f"Label range before clamping: min={masks.min()}, max={masks.max()}")
                # Print samples with invalid labels
                invalid_mask = masks >= self.config.num_classes

                max_label = self.config.num_classes - 1
                masks = masks.clamp_(0, max_label)
                print(f"Labels clamped to range [0, {max_label}]")

            rgb = rgb.to(device=self.device)
            ir = ir.to(device=self.device)
            masks = masks.to(device=self.device)
            
            with autocast('cuda', enabled=True): #with autocast('cuda',dtype=torch.bfloat16):#
                main_out, aux_out = self.model(rgb, ir)
                
                # Calculate losses
                loss_main = self.criterion_main(main_out, masks)
                
                # Handle deep supervision (PANet decoder returns dict during training)
                if isinstance(aux_out, dict):
                    # PANet deep supervision format: {'aux': tensor, 'deep': [t0, t1, t2, t3]}
                    loss_aux = self.criterion_aux(aux_out['aux'], masks)
                    
                    # Add deep supervision losses (weighted by level, deeper = lower weight)
                    deep_outputs = aux_out.get('deep', [])
                    deep_weights = [0.1, 0.2, 0.3, 0.4]  # Level 0->3: increasing importance
                    for i, deep_out in enumerate(deep_outputs):
                        weight = deep_weights[i] if i < len(deep_weights) else 0.1
                        loss_aux = loss_aux + weight * self.criterion_aux(deep_out, masks)
                else:
                    # Standard FPN format: aux_out is a tensor
                    loss_aux = self.criterion_aux(aux_out, masks) if aux_out is not None else 0
                
                # Combined loss
                loss = loss_main + aux_weight * loss_aux
                loss = loss / accumulation_steps  # Scale loss
            
            # Track metrics
            running_loss += loss.item() * accumulation_steps  # Unscale for tracking
            running_main_loss += loss_main.item()
            if aux_out is not None and (isinstance(aux_out, dict) or loss_aux != 0):
                running_aux_loss += loss_aux.item() if isinstance(loss_aux, torch.Tensor) else loss_aux
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                
                # Check for NaN gradients
                for param in self.model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        param.grad.zero_()
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # EMA güncelle
                if self.ema is not None:
                    self.ema.update(self.model)
            
            # Update progress bar
            num_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'main': f'{loss_main.item():.4f}',
                'aux': f'{loss_aux.item() if loss_aux != 0 else 0:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_main_loss = running_main_loss / num_batches
        epoch_aux_loss = running_aux_loss / num_batches
        
        return epoch_loss, epoch_main_loss, epoch_aux_loss
    
    def validate(self, epoch):
        """Optimized validation"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_masks = []
        aux_weight = self.get_aux_weight(epoch)
        conf_total = np.zeros((self.config.num_classes, self.config.num_classes))
        if self.ema is not None and self.config.eval_use_ema:
            self.ema.apply_shadow(self.model)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for rgb, ir, masks,_ in pbar:
                rgb = rgb.to(self.device)
                ir = ir.to(self.device) 
                masks = masks.to(self.device)
                
                with autocast(device_type='cuda', enabled=True):
                    main_out, aux_out = self.model(rgb, ir)
                
                loss_main = self.criterion_main(main_out, masks)
                
                # Handle deep supervision output format
                if isinstance(aux_out, dict):
                    loss_aux = self.criterion_aux(aux_out['aux'], masks)
                else:
                    loss_aux = self.criterion_aux(aux_out, masks) if aux_out is not None else 0
                
                loss = loss_main + aux_weight * loss_aux
                running_loss += loss.item()
                
                preds = main_out.argmax(dim=1).cpu().numpy()
                labels = masks.cpu().numpy()

                for pred, label in zip(preds, labels):
                    label = label.flatten()
                    pred = pred.flatten()
                    conf = confusion_matrix(y_true=label, y_pred=pred, labels=list(range(self.config.num_classes)))
                    conf_total += conf
                    
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # EMA restore
        if self.ema is not None and self.config.eval_use_ema:
            self.ema.restore(self.model)

        precision_per_class, recall_per_class, iou_per_class, f1score = compute_results(conf_total)
        miou = np.mean(iou_per_class)
        pixel_acc = np.trace(conf_total) / np.sum(conf_total)


        torch.cuda.empty_cache()
        return running_loss / len(self.val_loader), miou, pixel_acc, iou_per_class

    def _unwrap_model(self):
        m = self.model
        return m.module if hasattr(m, "module") else m

    def save_checkpoint(self, epoch, is_best=False):
        model_to_save = self._unwrap_model().state_dict()  
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_to_save,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if getattr(self, "scheduler", None) else None,
            "scaler_state_dict": self.scaler.state_dict() if getattr(self, "scaler", None) else None,
            "ema_state_dict": (self.ema.state_dict() if hasattr(self, "ema") and self.ema is not None else None),
            "best_miou": float(getattr(self, "best_miou", 0.0)),
            "config": vars(self.config) if hasattr(self.config, "__dict__") else self.config,
        }

        latest_path = self.checkpoint_dir / "latest.pth"
        tmp_path = self.checkpoint_dir / "latest.tmp"
        #torch.save(ckpt, tmp_path)
        #os.replace(tmp_path, latest_path)

        if is_best:
            torch.save(ckpt, self.checkpoint_dir / "best.pth")
            # --- Save EMA full model separately ---
            if self.ema is not None:
                # 1) Apply EMA weights to model 
                self.ema.apply_shadow(self.model)
                # 2) Get full state dict
                full_state = self._unwrap_model().state_dict()
                # 3) Save
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": full_state,
                    "config": vars(self.config),
                    "best_miou": float(self.best_miou)
                }, self.checkpoint_dir / "best_model.ema.pth")
                print("[INFO] EMA FULL model saved as best_model.ema.pth")
                # 4) Restore original weights
                self.ema.restore(self.model)

        if getattr(self.config, "save_interval", 0) and epoch % self.config.save_interval == 0:
            torch.save(ckpt, self.checkpoint_dir / f"epoch_{epoch}.pth")  

    def resume_from_best_with_reset(self):
        # Use custom path if provided, otherwise
        if self.config.resume_from:
            checkpoint_path = Path(self.config.resume_from)
        else:
            checkpoint_path = self.checkpoint_dir / "best.pth"
            
        if not checkpoint_path.exists():
            print(f"[WARN] No checkpoint found at: {checkpoint_path}")
            return

        print(f"[INFO] Resuming from best model with LR reset: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        #OPTIMIZER'I SIFIRDAN KUR
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        # EMA yükle
        if self.ema and checkpoint.get("ema_state_dict"):
            self.ema.load_state_dict(checkpoint["ema_state_dict"], self.device)

        self.start_epoch = 1 #checkpoint["epoch"] + 1
        self.best_miou = checkpoint["best_miou"]
        print(f"[INFO] Resuming from best model with best miou: {self.best_miou}")
        print(f"[INFO] Resume from BEST with optimizer reset @ epoch {self.start_epoch}")

    def load_checkpoint(self):
        """Load checkpoint for resuming training"""
        # Check if resume_from is specified (custom checkpoint path)
        if self.config.resume_from:
            checkpoint_path = Path(self.config.resume_from)
        else:
            # Use default checkpoint from experiment directory
            checkpoint_path = self.checkpoint_dir / 'latest.pth'
            if not checkpoint_path.exists():
                checkpoint_path = self.checkpoint_dir / 'best.pth'
        
        if not checkpoint_path.exists():
            print(f"[WARN] No checkpoint found at: {checkpoint_path}")
            return
        
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)

        # Always load model weights
        model_state = checkpoint.get('model_state_dict')
        if model_state:
            missing_keys, unexpected_keys = self.model.load_state_dict(model_state, strict=False)
            if missing_keys:
                print(f"[WARN] Missing keys in model state: {missing_keys}")
            if unexpected_keys:
                print(f"[WARN] Unexpected keys in model state: {unexpected_keys}")
            print("[INFO] Model weights loaded.")

        # Attempt to load optimizer
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("[INFO] Optimizer state loaded.")

        except Exception as e:
            print(f"[WARN] Optimizer state not loaded: {e}")

        # Attempt to load scheduler
        try:
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("[INFO] Scheduler state loaded.")
        except Exception as e:
            print(f"[WARN] Scheduler state not loaded: {e}")
        
        # Attempt to load scaler
        try:
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("[INFO] Scaler state loaded.")
        except Exception as e:
            print(f"[WARN] Scaler state not loaded: {e}")

        # Load EMA if available
        if self.ema is not None and checkpoint.get("ema_state_dict") is not None:
            try:
                self.ema.load_state_dict(checkpoint["ema_state_dict"],device=self.device)
                print("[INFO] EMA state loaded.")
            except Exception as e:
                print(f"[WARN] EMA state not loaded: {e}")
        
        self.start_epoch = 1 #checkpoint["epoch"] + 1
        self.best_miou = checkpoint['best_miou']
        
        print(f"Resumed from epoch {self.start_epoch} with best mIoU: {self.best_miou:.4f}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            torch.cuda.empty_cache()
            # Training
            train_loss, train_main_loss, train_aux_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, miou, pixel_acc, class_ious = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(miou)
                else:
                    self.scheduler.step()
            
            # Check if best model
            is_best = miou > self.best_miou
            if is_best:
                self.best_miou = miou
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Log results
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f} (Main: {train_main_loss:.4f}, Aux: {train_aux_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"mIoU: {miou:.4f} (Best: {self.best_miou:.4f})")
            print(f"Pixel Acc: {pixel_acc:.4f}")
            
            # Log class-wise IoU
            if self.config.verbose:
                class_names = ['unlabeled', 'car', 'person', 'bike', 'curve', 
                              'car_stop', 'guardrail', 'color_cone', 'bump']
                for i, iou in enumerate(class_ious):
                    if i < len(class_names):
                        print(f"  {class_names[i]}: {iou:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/mIoU', miou, epoch)
            self.writer.add_scalar('Val/PixelAcc', pixel_acc, epoch)
            
            # Log learning rates
            for param_group in self.optimizer.param_groups:
                name = param_group.get('name', 'default')
                self.writer.add_scalar(f'LR/{name}', param_group['lr'], epoch)
            try:
                lrs = ','.join([f"{pg.get('name','default')}:{pg['lr']:.2e}" for pg in self.optimizer.param_groups])
            except Exception:
                lrs = ','.join([f"{pg.get('lr',0):.2e}" for pg in self.optimizer.param_groups])

            self.log(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"TrainLoss={train_loss:.4f} (Main={train_main_loss:.4f}, Aux={train_aux_loss:.4f}) | "
                f"ValLoss={val_loss:.4f} | mIoU={miou:.4f} | PixelAcc={pixel_acc:.4f} | LR={lrs}"
            )
            
            # Log to wandb
            # if self.config.use_wandb:
            #     wandb.log({
            #         'epoch': epoch,
            #         'train_loss': train_loss,
            #         'val_loss': val_loss,
            #         'miou': miou,
            #         'pixel_acc': pixel_acc
            #     })
            
            print("-" * 50)
        
        print("\nTraining completed!")
        print(f"Best mIoU: {self.best_miou:.4f}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train RGB-IR Fusion Model')
    
    # Model arguments
    parser.add_argument('--rgb_model_path', type=str, default='',
                        help='Path to pretrained RGB model')
    parser.add_argument('--ir_model_path', type=str, default='',
                        help='Path to pretrained IR model')
    parser.add_argument('--fusion_strategy', type=str, default='average_summation',
                        choices=['simple', 'middle_attention','average_summation'],
                        help='Fusion strategy')
    parser.add_argument('--fuse_type', type=str, default='scalar',#pixel',scalar
                        choices=['scalar', 'channel','pixel'])
    parser.add_argument('--distill_type', type=str, default='mul',
                        choices=['mul', 'sum'])
    parser.add_argument('--rgb_arch', type=str, default='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
                        help='timm model name for RGB backbone')
    parser.add_argument('--ir_arch', type=str, default='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
                        help='timm model name for IR backbone')
    parser.add_argument('--context_dim', type=str, default='[96,192,384,768]')
    parser.add_argument('--freeze_backbones', action='store_true', default = False,
                        help='Freeze pretrained backbones')
    parser.add_argument('--distill_layers_enabled', action='store_true', default = False,
                        help='Freeze pretrained backbones')
    parser.add_argument('--resume', action='store_true',default=False,
        help='Resume training from checkpoint')
    parser.add_argument('--resume_with_reset', action='store_true',default=False,
        help='Resume model weights but reset optimizer and scheduler')
    parser.add_argument('--resume_from', type=str, default='',
        help='Path to checkpoint to resume from (default: exp_dir/checkpoints/best.pth)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mfnet',
        choices=['mfnet', 'pst900'],
        help='Dataset selection: mfnet (9 classes) or pst900 (5 classes)')
    parser.add_argument('--img_height', type=int, default=480,
        help='Input image height (default: 480 for MFNet, 720 for PST900)')
    parser.add_argument('--img_width', type=int, default=640,
        help='Input image width (default: 640 for MFNet, 1280 for PST900)')
    parser.add_argument('--data_root', type=str, default = '/datavolume/data/emrecanitez/Datasets/MFNet',
        help='Root directory for datasets')
    parser.add_argument('--train_source', type=str, default='train_',
        help='Training dataset split name',
        choices=['train', 'train_', 'train_balanced', 'train_balanced_2to1', 'train_mixed', 'train_boost', 'train_guardrail_night_only', 'train_guardrail_augmented', 'train_guardrail_copypaste'])
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of segmentation classes')
    parser.add_argument('--ignore_unlabeled', action='store_true', default=False,
                        help='Ignore unlabeled class in loss')
    parser.add_argument('--class_weights', action='store_true',default=True,
                        help='Use class weights in loss')
    parser.add_argument('--class_weight_multiplier', type=float, default=10.0,
                        help='Multiplier for class weights (10 for balanced, 5 for full)')
    
    # Decoder architecture
    parser.add_argument('--decoder_type', type=str, default='panet',
                        choices=['fpn', 'panet'],
                        help='Decoder type: fpn (standard) or panet (bidirectional with deep supervision)')
    parser.add_argument('--deep_supervision', action='store_true', default=False,
                        help='Enable deep supervision for PANet decoder')
    parser.add_argument('--enhanced_fusion', action='store_true', default=False,
                        help='Use EnhancedSemanticFusion for stages 3-4 (multi-scale + attention)')
    parser.add_argument('--use_safd', action='store_true', default=False,
                        help='Novel: Scene-Adaptive Frequency Decomposition')
    parser.add_argument('--use_cafg', action='store_true', default=False,
                        help='Novel: Complementarity-Aware Fusion Gate')
    parser.add_argument('--use_tpsw', action='store_true', default=False,
                        help='Novel: Thermal Prior-Guided Spatial Weighting')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--lr_backbone', type=float, default=3e-5,
                        help='Learning rate for backbone')
    parser.add_argument('--lr_fusion', type=float, default=3e-4,
                        help='Learning rate for fusion modules')
    parser.add_argument('--lr_decoder', type=float, default=3e-4,
                        help='Learning rate for decoder')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=0.1,
                        help='Gradient clipping value')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine_warmup',# 'cosine', #plateau , cosine_restart
                        choices=['plateau','cosine', 'poly', 'step', 'cosine_warmup','cosine_restart','none'],
                        help='Learning rate scheduler')
    parser.add_argument('--t0', type=int, default=50,
                        help='T0 for cosine annealing warm restarts')
    parser.add_argument('--step_size', type=int, default=45,
                        help='Step size for step scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='warmup epochs for cosine_warmup scheduler')
    parser.add_argument('--flat_epochs', type=int, default=10,
                        help='flat epochs for cosine_warmup scheduler')

    # Loss arguments
    parser.add_argument('--ce_weight', type=float, default=0.4,
                        help='Cross entropy weight in combo loss')
    parser.add_argument('--dice_weight', type=float, default=0.3,
                        help='Dice weight in combo loss')
    parser.add_argument('--lovasz_weight', type=float, default=0.3,
                        help='Lovasz weight in combo loss')
    parser.add_argument('--aux_weight_start', type=float, default=0.3,
                        help='Initial auxiliary loss weight')
    parser.add_argument('--aux_weight_end', type=float, default=0.05,
                        help='Final auxiliary loss weight')
    parser.add_argument('--aux_weight_decay_epoch', type=int, default=80,
                        help='Epochs to decay auxiliary weight')
    
    # Experiment arguments
    parser.add_argument('--exp_name', type=str, default='experiment7_2',
                        help='Experiment name')
    parser.add_argument('--exp_dir', type=str, default='Experiments',
                        help='Experiment directory')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    # System arguments
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU device ID')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Use multiple GPUs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    parser.add_argument('--ema', action='store_true', default=True, help='Track EMA of weights.')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay.')
    parser.add_argument('--eval_use_ema', action='store_true', default=True, help='Use EMA weights in validation.')
    parser.add_argument('--eval_tta', action='store_true', default=False, help='Use simple TTA in validation (slow).')
    parser.add_argument('--tta_scales', type=float, nargs='+', default=[1.0], help='TTA scales, e.g., 0.75 1.0 1.25')

    parser.add_argument('--loss_type', type=str, default='combo3', choices=['combo3', 'combo_ohem'],
                    help='combo3: CE+Dice+Lovasz; combo_ohem: CE+Dice+Lovasz+OHEM+Boundary (for rare classes)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='CE label smoothing.')
    
    # OHEM and Boundary Loss arguments
    parser.add_argument('--ohem_weight', type=float, default=0.1, help='OHEM loss weight in combo_ohem')
    parser.add_argument('--boundary_weight', type=float, default=0.1, help='Boundary loss weight in combo_ohem')
    parser.add_argument('--ohem_thresh', type=float, default=0.7, help='OHEM probability threshold (lower = more hard examples)')
    parser.add_argument('--ohem_min_kept', type=int, default=100000, help='Minimum pixels to keep in OHEM')
    parser.add_argument('--focal_weight', type=float, default=0.0, help='Focal loss weight (0=disabled, 0.2=stage2)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma (default=2.0)')
    parser.add_argument('--focal_target_class', type=int, default=-1, help='Target class for Focal Loss (-1=all classes, 3=Hand Drill for PST900, 6=Guardrail for MFNet)')

    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps.')

    args = parser.parse_args()
    
    # Set experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"fusion_{args.fusion_strategy}_{timestamp}"
    
    # Create trainer and start training
    trainer = FusionTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()