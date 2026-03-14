import os
import sys

# import torchinfo  # Optional, comment out if not installed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
from thop import profile

class FusionModel(nn.Module):    
    def __init__(
        self,
        rgb_arch='convnextv2_tiny.fcmae_ft_in22k_in1k_384', ir_arch='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
        pretrained = True,
        num_classes: int = 9,
        fusion_strategy: str = "middle_attention",
        freeze_backbones: bool = False,
        distill_layers_enabled: bool = True,
        fuse_type: str = "scalar", # "scalar", "channel", "pixel"
        distill_type: str = "mul", # "sum", "mul"
        context_dim: int = [96,192,384,768],
        input_resolution: Tuple[int, int] = (480, 640),
        rgb_backbone_resolution: Tuple[int, int] = (480, 640),  # For RGB backbone
        ir_backbone_resolution: Tuple[int, int] = (480, 640),   # For IR backbone
        output_resolution: Tuple[int, int] = (480, 640),  # Final output resolution
        decoder_type: str = "fpn",  # "fpn" (original) or "panet" (new bidirectional)
        deep_supervision: bool = False,  # Enable deep supervision for PANet decoder
        enhanced_fusion: bool = False,  # Use EnhancedSemanticFusion for stages 3-4
        use_safd: bool = False,       # Novel: Scene-Adaptive Frequency Decomposition
        use_cafg: bool = False,       # Novel: Complementarity-Aware Fusion Gate
        use_tpsw: bool = False,       # Novel: Thermal Prior-Guided Spatial Weighting
    ):
        super().__init__()

        self.decoder_type = decoder_type
        self.enhanced_fusion = enhanced_fusion
        self.use_safd = use_safd
        self.use_cafg = use_cafg
        self.use_tpsw = use_tpsw

        # ---- Decoder Selection ----
        if decoder_type == "panet":
            print(f"[FusionModel] Using PANet decoder with deep_supervision={deep_supervision}")
            self.decoder = PANetDecoder(
                in_channels=context_dim,
                num_classes=num_classes,
                output_resolution=output_resolution,
                deep_supervision=deep_supervision
            )
        else:
            print(f"[FusionModel] Using standard FPN decoder")
            self.decoder = FusionAwareNativeResolutionDecoder(
                in_channels=context_dim,
                num_classes=num_classes,
                output_resolution=output_resolution
            )

        self.input_resolution = input_resolution
        self.rgb_backbone_resolution = rgb_backbone_resolution
        self.ir_backbone_resolution = ir_backbone_resolution
        self.output_resolution = output_resolution
        self.num_classes = num_classes
        self.context_dims = context_dim
        self.distill_layers_enabled = distill_layers_enabled
        self.fuse_type = fuse_type
        self.distill_type = distill_type
        
        print(f"Fusion Model Configuration:")
        print(f"  Input Resolution: {input_resolution} (HxW)")
        print(f"  RGB Backbone Resolution: {rgb_backbone_resolution}")
        print(f"  IR Backbone Resolution: {ir_backbone_resolution}")
        print(f"  Output Resolution: {output_resolution} (HxW)")
        print(f"  Number of Classes: {num_classes}")
        
        # --- 1. RGB Backbone (ConvNeXt V2) ---
        self.rgb_encoder = timm.create_model(
            rgb_arch,
            pretrained=pretrained,
            features_only=True,
            
        )
        
        # --- 2. IR Backbone (MobileNetV4 Hybrid) ---
        self.ir_encoder = timm.create_model(
            ir_arch,
            pretrained=pretrained,
            features_only=True,
            in_chans=1, 
            #out_indices=(1, 2, 3, 4),
        )

        # IR Backbone için kanal bilgisini öğrenelim (Füzyon katmanları için gerekli olacak)
        self.rgb_channels = self.rgb_encoder.feature_info.channels()
        self.ir_channels = self.ir_encoder.feature_info.channels()
        
        print(f"RGB Backbone Kanalları: {self.rgb_channels}")
        print(f"IR Backbone Kanalları:  {self.ir_channels}")

        num_of_params = sum(
            p.numel() for p in self.rgb_encoder.parameters()
        ) / 1e6
        print(f"Number of parametre for rgb encoder {rgb_arch}: {num_of_params:.2f} M")

        num_of_params = sum(
            p.numel() for p in self.ir_encoder.parameters()
        ) / 1e6
        print(f"Number of parametre for ir encoder {ir_arch}: {num_of_params:.2f} M")

        # --- Thermal Prior Module (raw IR → spatial prior) ---
        if use_tpsw:
            print(f"[FusionModel] ③ Thermal Prior-Guided Spatial Weighting ENABLED")
            self.thermal_prior = ThermalPriorModule()

        # --- Fusion Stages ---
        # Stage 1-2: Frequency-aware fusion (low-level details)
        if use_safd or use_cafg:
            print(f"[FusionModel] Novel fusion (Stage 1-2): SAFD={use_safd}, CAFG={use_cafg}")
            self.fusion_stage1 = NovelFrequencyAwareFusionModule(
                rgb_c=self.rgb_channels[0], ir_c=self.ir_channels[0],
                use_safd=use_safd, use_cafg=use_cafg
            )
            self.fusion_stage2 = NovelFrequencyAwareFusionModule(
                rgb_c=self.rgb_channels[1], ir_c=self.ir_channels[1],
                use_safd=use_safd, use_cafg=use_cafg
            )
        else:
            self.fusion_stage1 = FrequencyAwareFusionModule(rgb_c=self.rgb_channels[0], ir_c=self.ir_channels[0])
            self.fusion_stage2 = FrequencyAwareFusionModule(rgb_c=self.rgb_channels[1], ir_c=self.ir_channels[1])

        # Stage 3-4: Semantic fusion (high-level semantics)
        if use_cafg:
            print(f"[FusionModel] Novel fusion (Stage 3-4): CAFG=True")
            self.fusion_stage3 = NovelSemanticFusionModule(rgb_c=self.rgb_channels[2], ir_c=self.ir_channels[2])
            self.fusion_stage4 = NovelSemanticFusionModule(rgb_c=self.rgb_channels[3], ir_c=self.ir_channels[3])
        elif enhanced_fusion:
            print(f"[FusionModel] Using EnhancedSemanticFusion for stages 3-4")
            self.fusion_stage3 = EnhancedSemanticFusion(rgb_c=self.rgb_channels[2], ir_c=self.ir_channels[2])
            self.fusion_stage4 = EnhancedSemanticFusion(rgb_c=self.rgb_channels[3], ir_c=self.ir_channels[3])
        else:
            print(f"[FusionModel] Using FrequencyAwareFusionModule for stages 3-4")
            #self.fusion_stage3 = SemanticCrossGatedFusion(rgb_c=self.rgb_channels[2], ir_c=self.ir_channels[2])
            #self.fusion_stage4 = SemanticCrossGatedFusion(rgb_c=self.rgb_channels[3], ir_c=self.ir_channels[3])
            self.fusion_stage3 = FrequencyAwareFusionModule(rgb_c=self.rgb_channels[2], ir_c=self.ir_channels[2])
            self.fusion_stage4 = FrequencyAwareFusionModule(rgb_c=self.rgb_channels[3], ir_c=self.ir_channels[3])
  

    @staticmethod
    def print_trainable_stats(self,module: nn.Module, title="module"):
        tot = sum(p.numel() for p in module.parameters())
        trn = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"[{title}] trainable params: {trn:,} / {tot:,} ({100*trn/max(1,tot):.2f}%)")
        
    def extract_features(self, x, backbone):
        """Extract multi-scale features from backbone"""
        features = []
        features = backbone(x)
        
        return features

        
    def forward(self, rgb, ir):
        # Extract features (with gradients now)
        rgb_features = self.extract_features(rgb, self.rgb_encoder)
        ir_features = self.extract_features(ir, self.ir_encoder)

        # Thermal Prior Maps (from raw IR, before any backbone processing)
        if self.use_tpsw:
            target_sizes = [f.shape[2:] for f in rgb_features]
            thermal_priors = self.thermal_prior(ir, target_sizes)

        # Stage-wise fusion
        fused = self.fusion_stage1(rgb_features[0], ir_features[0])  # Stage 1 Fusion
        fused = [fused]
        fused.append(self.fusion_stage2(rgb_features[1], ir_features[1]))  # Stage 2 Fusion
        fused.append(self.fusion_stage3(rgb_features[2], ir_features[2]))  # Stage 3 Fusion
        fused.append(self.fusion_stage4(rgb_features[3], ir_features[3]))  # Stage 4 Fusion 

        # Apply Thermal Prior-Guided Spatial Weighting
        if self.use_tpsw:
            fused = [f * (1 + tp) for f, tp in zip(fused, thermal_priors)]

        main_logits, aux_logits = self.decoder(fused)
        return main_logits, aux_logits

# ==========================================
# 1. SCALAR CONFIDENCE GATE (Gece/Gündüz Bekçisi)
# ==========================================
class ScalarConfidenceGate(nn.Module):
    """
    RGB görüntüsünün genel güvenilirliğini ölçer.
    Çıktı: [B, 1, 1, 1] boyutunda 0-1 arası bir skaler.
    Kanal bazlı değil, global karar verir (Modelin bir kısmını öldürmemek için).
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # Global Context
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1, bias=False),
            nn.Sigmoid()                      # 0 (Güvenilmez) - 1 (Güvenilir)
        )

    def forward(self, x):
        return self.gate(x)

# ==========================================
# 2. SAFE RESIDUAL FUSION (Güvenli Birleştirici)
# ==========================================
class SafeResidualFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # A. Güven Kapısı
        self.rgb_gate = ScalarConfidenceGate(channels)
        
        # B. Delta (Düzeltme) Üretici
        # RGB ve IR'ı alıp, RGB'yi düzeltecek "farkı" hesaplar.
        # SONUNDA RELU YOKTUR! (Negatif değerlere izin verilmeli ki gürültüyü silebilsin)
        self.refine = nn.Sequential(
            nn.Conv2d(channels*2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels) 
        )

    def forward(self, rgb_feat, ir_final):
        # 1. RGB'nin durumuna bak (Karanlık mı? Gürültülü mü?)
        score = self.rgb_gate(rgb_feat)
        
        # 2. Refine Bloğuna İpucu Ver
        # RGB'yi score ile çarparak birleştiriyoruz.
        # Eğer score düşükse, refine bloğu "RGB bozuk, IR'a güvenip negatif delta üretmeliyim" der.
        fused_input = torch.cat([rgb_feat * score, ir_final], dim=1)
        
        # 3. Delta (Düzeltme) Hesapla
        delta = self.refine(fused_input)
        
        # 4. GÜVENLİ ÇIKIŞ (Residual Connection)
        # RGB ana hattını ASLA score ile çarpmıyoruz.
        # Bu sayede gradyan akışı asla kesilmez (Backbone ölmez).
        # Gürültü varsa, 'delta' onu matematiksel olarak siler (Noise Cancellation).
        return rgb_feat + delta

# ==========================================
# 3. DUAL POOL ATTENTION (Kazanan Her Şeyi Almasın)
# ==========================================
class DualPoolSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # 2 kanal girer (Avg + Max), 1 kanal maske çıkar
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x_high negatif değerler içerebilir (fark işlemi yüzünden).
        # Dikkat maskesi "önem" belirttiği için mutlak değer (magnitude) daha anlamlı olabilir.
        # Ancak standart CBAM direkt x kullanır. Biz de standardı bozmayalım.
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        mask = self.conv(x_cat)
        return self.sigmoid(mask)

# ==========================================
# 4. ANA MODÜL (STAGE 1 ve 2 İÇİN)
# ==========================================

class GaussianLowPass(nn.Module):
    def __init__(self, channels, k=7, sigma=2.0): # k=7, sigma=2.0 genelde iyi bir dengedir
        super().__init__()
        # Gaussian Kernel Oluşturma
        grid = torch.arange(k) - k//2
        gauss = torch.exp(-grid**2 / (2*sigma**2))
        kernel = gauss[:,None] * gauss[None,:]
        kernel = kernel / kernel.sum()
        
        # Kernel'i modele kaydet (Eğitilmez, sabit kalır)
        self.register_buffer(
            "kernel", kernel[None,None,:,:].repeat(channels,1,1,1)
        )
        
        # Depthwise Convolution ayarları
        self.groups = channels
        self.pad = k//2 

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.pad, groups=self.groups)
    
class FrequencyAwareFusionModule(nn.Module):
    def __init__(self, rgb_c, ir_c):
        super().__init__()
        self.ir_proj = nn.Sequential(
            nn.Conv2d(ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU()
        )
        #nn.init.zeros_(self.ir_proj[0].weight)

        # --- A. IR İŞLEME (Frekans Ayırma & Seçme) ---

        self.lowpass = GaussianLowPass(rgb_c)
        
        # Isı bloklarını kaçırmamak için büyük kernel.
        self.sa_low = DualPoolSpatialAttention(kernel_size=7)
        # Özellikle "Guardrail" gibi ince nesneleri bulanıklaştırmamak için küçük kernel.
        self.sa_high = DualPoolSpatialAttention(kernel_size=3)  
        
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_c * 2, rgb_c, 1, bias=False),
            nn.Sigmoid()
        )

        self.fusion_block = SafeResidualFusion(rgb_c)

    def decompose(self, x):
            # x_low: Bulanıklaştırılmış (Sadece ana hatlar ve sıcaklık)
            x_low = self.lowpass(x)
            
            # x_high: Orijinal - Bulanık (Sadece keskin kenarlar)
            # Unsharp Masking tekniği
            x_high = x - x_low
            
            return x_low, x_high

    def forward(self, rgb_feat, ir_feat):
        ir_feat = self.ir_proj(ir_feat)
        # 1. IR'ı Frekanslarına Ayır
        ir_low, ir_high = self.decompose(ir_feat)
        
        # 2. IR Özelliklerini Seç (Attention)
        # Low maske sıcaklığı, High maske kenarları seçer.
        mask_low = self.sa_low(ir_low)
        mask_high = self.sa_high(ir_high)
        
        ir_low_refined = ir_low * mask_low
        ir_high_refined = ir_high * mask_high
        
        # 3. İdeal IR Karışımını Oluştur
        ir_mixed = torch.cat([ir_low_refined, ir_high_refined], dim=1)
        alpha = self.freq_gate(torch.abs(ir_mixed))                             # [B,C,1,1]
        ir_final = alpha * ir_low_refined + (1 - alpha) * ir_high_refined
        
        # 4. RGB ile Güvenli Füzyon Yap
        # RGB'nin gürültüsü burada 'delta' ile temizlenir.
        out = self.fusion_block(rgb_feat, ir_final)
        
        return out
# ==========================================
# 5. BASİT FÜZYON (STAGE 3 ve 4 İÇİN)
# ==========================================
# class SimpleFusionModule(nn.Module):
#     def __init__(self, rgb_c, ir_c):
#         super().__init__()
#         self.ir_proj = nn.Conv2d(ir_c, rgb_c, 1, bias=False)
#         self.conv = nn.Sequential(
#             nn.Conv2d(rgb_c * 2, rgb_c, 1, bias=False),
#             nn.BatchNorm2d(rgb_c),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, rgb_feat, ir_feat):
#         ir_feat = self.ir_proj(ir_feat)
#         return self.conv(torch.cat([rgb_feat, ir_feat], dim=1))

class SemanticCrossGatedFusion(nn.Module):
    def __init__(self, rgb_c, ir_c, reduction=16):
        super().__init__()

        self.ir_proj = nn.Sequential(
            nn.Conv2d(ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c)
        )

        self.cross_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_c * 2, rgb_c // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_c // reduction, rgb_c, 1),
            nn.Sigmoid()
        )

        self.spatial_refine = nn.Sequential(
            nn.Conv2d(rgb_c, rgb_c, 3, padding=1, groups=rgb_c, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c)
        )

    def forward(self, rgb_feat, ir_feat):
        ir_feat = self.ir_proj(ir_feat)

        joint = torch.cat([rgb_feat, ir_feat], dim=1)
        g = self.cross_gate(joint)

        delta = ir_feat - rgb_feat
        fused = rgb_feat + g * delta

        refined = self.spatial_refine(fused)
        return fused + refined


# ==========================================
# ENHANCED SEMANTIC FUSION (Stage 3-4 için)
# ==========================================
class EnhancedSemanticFusion(nn.Module):
    """
    Enhanced Semantic Cross-Gated Fusion for high-level stages (3-4).
    
    Improvements over SemanticCrossGatedFusion:
    1. Multi-scale convolutions (3x3 + 5x5) for capturing objects at different scales
    2. Channel attention (SE-like) for better channel-wise feature selection
    3. Spatial attention for focusing on important regions
    4. Residual dense connections for gradient flow
    5. Feature calibration for IR-RGB alignment
    
    Designed for better rare class detection (Hand Drill, Guardrail).
    """
    def __init__(self, rgb_c, ir_c, reduction=16):
        super().__init__()
        
        # === IR Feature Projection with Enhancement ===
        self.ir_proj = nn.Sequential(
            nn.Conv2d(ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU()
        )
        
        # === Cross-Modal Attention Gate (Channel-wise) ===
        # Learns when to trust RGB vs IR based on global context
        self.cross_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_c * 2, rgb_c // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_c // reduction, rgb_c * 2, 1, bias=False),  # Output for both RGB and IR
        )
        
        # === Multi-Scale Feature Extraction (Depthwise for efficiency) ===
        # 3x3 for local details, 5x5 for larger context
        self.ms_conv_3x3 = nn.Sequential(
            nn.Conv2d(rgb_c, rgb_c, 3, padding=1, groups=rgb_c, bias=False),  # Depthwise
            nn.BatchNorm2d(rgb_c),
            nn.GELU(),
            nn.Conv2d(rgb_c, rgb_c, 1, bias=False),  # Pointwise
            nn.BatchNorm2d(rgb_c)
        )
        
        self.ms_conv_5x5 = nn.Sequential(
            nn.Conv2d(rgb_c, rgb_c, 5, padding=2, groups=rgb_c, bias=False),  # Depthwise
            nn.BatchNorm2d(rgb_c),
            nn.GELU(),
            nn.Conv2d(rgb_c, rgb_c, 1, bias=False),  # Pointwise
            nn.BatchNorm2d(rgb_c)
        )
        
        # Multi-scale fusion
        self.ms_fusion = nn.Sequential(
            nn.Conv2d(rgb_c * 2, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c)
        )
        
        # === Channel Attention (SE-like) ===
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_c, rgb_c // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_c // reduction, rgb_c, 1, bias=False),
            nn.Sigmoid()
        )
        
        # === Spatial Attention ===
        # Helps focus on small objects that might be missed
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # === Final Refinement Block (Depthwise Separable for efficiency) ===
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_c, rgb_c, 3, padding=1, groups=rgb_c, bias=False),  # Depthwise
            nn.BatchNorm2d(rgb_c),
            nn.GELU(),
            nn.Conv2d(rgb_c, rgb_c, 1, bias=False),  # Pointwise
            nn.BatchNorm2d(rgb_c)
        )
        
        # Learnable residual scale
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, rgb_feat, ir_feat):
        # 1. Project IR to RGB channel space
        ir_proj = self.ir_proj(ir_feat)
        
        # 2. Cross-modal attention gate 
        joint = torch.cat([rgb_feat, ir_proj], dim=1)
        gate = self.cross_gate(joint)
        gate = torch.sigmoid(gate)
        
        # Split gate for RGB and IR
        gate_rgb, gate_ir = gate.chunk(2, dim=1)
        
        # Weighted combination
        fused = gate_rgb * rgb_feat + gate_ir * ir_proj
        
        # 3. Multi-scale feature extraction
        feat_3x3 = self.ms_conv_3x3(fused)
        feat_5x5 = self.ms_conv_5x5(fused)
        multi_scale = self.ms_fusion(torch.cat([feat_3x3, feat_5x5], dim=1))
        
        # 4. Channel attention
        ca = self.channel_attention(multi_scale)
        multi_scale = multi_scale * ca
        
        # 5. Spatial attention
        max_pool = torch.max(multi_scale, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(multi_scale, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        multi_scale = multi_scale * sa
        
        # 6. Refinement with residual
        refined = self.refine(multi_scale)
        
        # 7. Residual connection with learnable scale
        output = fused + self.res_scale * refined
        
        return output


# ==========================================
# Experiment: Scene-Adaptive Frequency Decomposition (SAFD)
# ==========================================
class SceneAdaptiveSigmaPredictor(nn.Module):
    """
    Cross-modal conditioned sigma prediction for adaptive Gaussian decomposition.
    Predicts per-image mixing weights over a bank of pre-computed Gaussian kernels
    at different sigma values, conditioned on both RGB and IR feature statistics.
    
    Novelty: Uses cross-modal context (RGB illumination quality + IR thermal patterns)
    to adaptively control the frequency split point for IR decomposition.
    """
    def __init__(self, channels, num_bands=4):
        super().__init__()
        self.num_bands = num_bands
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_bands, 1, bias=False),
        )

    def forward(self, rgb_feat, ir_feat):
        """
        Args:
            rgb_feat: [B, C, H, W] RGB features
            ir_feat:  [B, C, H, W] IR features (after channel projection)
        Returns:
            weights: [B, num_bands, 1, 1] softmax weights for kernel bank mixing
        """
        ctx = torch.cat([rgb_feat, ir_feat], dim=1)
        weights = self.predictor(ctx)  # [B, num_bands, 1, 1]
        weights = F.softmax(weights, dim=1)
        return weights


class AdaptiveGaussianLowPass(nn.Module):
    """
    Multi-band Gaussian low-pass with scene-adaptive mixing.
    Pre-computes a bank of Gaussian kernels at different sigma values,
    then mixes them based on weights from SceneAdaptiveSigmaPredictor.
    
    Fully differentiable, no per-image loops needed for the mixing.
    """
    def __init__(self, channels, k=7, num_bands=4, sigma_range=(0.5, 4.0)):
        super().__init__()
        self.channels = channels
        self.pad = k // 2
        self.num_bands = num_bands

        sigmas = torch.linspace(sigma_range[0], sigma_range[1], num_bands)
        grid = torch.arange(k).float() - k // 2

        for idx, s in enumerate(sigmas):
            gauss = torch.exp(-grid**2 / (2 * s.item()**2))
            kernel_2d = gauss[:, None] * gauss[None, :]
            kernel_2d = kernel_2d / kernel_2d.sum()
            kernel = kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1)
            self.register_buffer(f'kernel_{idx}', kernel)

    def forward(self, x, band_weights):
        """
        Args:
            x: [B, C, H, W] input features
            band_weights: [B, num_bands, 1, 1] from SceneAdaptiveSigmaPredictor
        Returns:
            low_pass: [B, C, H, W] adaptively smoothed result
        """
        result = 0
        for idx in range(self.num_bands):
            kernel = getattr(self, f'kernel_{idx}')
            smoothed = F.conv2d(x, kernel, padding=self.pad, groups=self.channels)
            result = result + band_weights[:, idx:idx+1, :, :] * smoothed
        return result


# ==========================================
# Experiment: Complementarity-Aware Fusion Gate (CAFG)
# ==========================================
class ComplementarityAwareFusionGate(nn.Module):
    """
    Measures inter-modal complementarity via feature-space cosine distance,
    then routes fusion through simple (for redundant regions) or deep
    (for complementary regions) pathways.
    
    Novelty: Explicitly quantifies WHERE two modalities provide different 
    vs similar information, and allocates fusion complexity accordingly.
    This is fundamentally different from attention-based methods that learn 
    "where to look" without explicitly modeling inter-modal agreement.
    """
    def __init__(self, channels, proj_dim=None):
        super().__init__()
        if proj_dim is None:
            proj_dim = max(channels // 4, 16)

        # Projection for complementarity measurement
        self.comp_proj = nn.Sequential(
            nn.Conv2d(channels, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim)
        )

        # Deep fusion path (activated for complementary regions)
        self.deep_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, rgb_feat, ir_feat):
        """
        Args:
            rgb_feat: [B, C, H, W]
            ir_feat:  [B, C, H, W] (must be same channel count)
        Returns:
            fused: [B, C, H, W]
        """
        # Complementarity map via cosine distance
        rgb_proj = F.normalize(self.comp_proj(rgb_feat), dim=1)
        ir_proj = F.normalize(self.comp_proj(ir_feat), dim=1)

        # High = complementary (modalities disagree), Low = redundant (agree)
        comp_map = (1.0 - (rgb_proj * ir_proj).sum(dim=1, keepdim=True)).clamp(0, 1)

        # Redundant regions → simple addition is sufficient
        simple_out = rgb_feat + ir_feat
        # Complementary regions → need deeper cross-modal reasoning
        deep_out = self.deep_fuse(torch.cat([rgb_feat, ir_feat], dim=1))

        return (1 - comp_map) * simple_out + comp_map * deep_out


# ==========================================
# Experiment: Thermal Prior-Guided Spatial Weighting (TPSW)
# ==========================================
class ThermalPriorModule(nn.Module):
    """
    Extracts physics-informed spatial prior from raw thermal input (pre-backbone).
    
    Novelty: All existing RGB-T methods process IR through a backbone first, 
    losing absolute temperature information. This module uses raw thermal values 
    as a spatial prior — high thermal contrast = likely object boundary.
    Physics motivation: Thermal cameras measure surface temperature (Planck's law),
    and temperature discontinuities strongly correlate with object boundaries.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, raw_ir, target_sizes):
        """
        Args:
            raw_ir: [B, 1, H, W] raw thermal input image
            target_sizes: list of (H, W) for each feature scale
        Returns:
            priors: list of [B, 1, Hi, Wi] thermal prior maps
        """
        prior = self.encoder(raw_ir)  # [B, 1, H, W] full-res prior
        priors = []
        for size in target_sizes:
            p = F.interpolate(prior, size=size, mode='bilinear', align_corners=False)
            priors.append(p)
        return priors


# ==========================================
# Experiment: Integrated Fusion: Stage 1-2 (SAFD + CAFG)
# ==========================================
class NovelFrequencyAwareFusionModule(nn.Module):
    """
    Enhanced frequency-aware fusion with SAFD and CAFG for stages 1-2.
    Drop-in replacement for FrequencyAwareFusionModule when novel features enabled.
    """
    def __init__(self, rgb_c, ir_c, use_safd=True, use_cafg=True, num_bands=4):
        super().__init__()
        self.use_safd = use_safd
        self.use_cafg = use_cafg

        self.ir_proj = nn.Sequential(
            nn.Conv2d(ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU()
        )

        # Frequency decomposition: adaptive (SAFD) or fixed
        if use_safd:
            self.sigma_predictor = SceneAdaptiveSigmaPredictor(rgb_c, num_bands=num_bands)
            self.adaptive_lowpass = AdaptiveGaussianLowPass(rgb_c, num_bands=num_bands)
        else:
            self.lowpass = GaussianLowPass(rgb_c)

        # Spatial attention for decomposed components
        self.sa_low = DualPoolSpatialAttention(kernel_size=7)
        self.sa_high = DualPoolSpatialAttention(kernel_size=3)

        # Frequency mixing gate
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_c * 2, rgb_c, 1, bias=False),
            nn.Sigmoid()
        )

        # Final fusion: CAFG or original SafeResidualFusion
        if use_cafg:
            self.fusion_gate = ComplementarityAwareFusionGate(rgb_c)
        else:
            self.fusion_block = SafeResidualFusion(rgb_c)

    def forward(self, rgb_feat, ir_feat):
        ir_feat = self.ir_proj(ir_feat)

        # 1. Frequency decomposition (adaptive or fixed)
        if self.use_safd:
            band_weights = self.sigma_predictor(rgb_feat, ir_feat)
            ir_low = self.adaptive_lowpass(ir_feat, band_weights)
        else:
            ir_low = self.lowpass(ir_feat)
        ir_high = ir_feat - ir_low

        # 2. Spatial attention on decomposed components
        mask_low = self.sa_low(ir_low)
        mask_high = self.sa_high(ir_high)
        ir_low_refined = ir_low * mask_low
        ir_high_refined = ir_high * mask_high

        # 3. Adaptive frequency mixing
        ir_mixed = torch.cat([ir_low_refined, ir_high_refined], dim=1)
        alpha = self.freq_gate(torch.abs(ir_mixed))
        ir_final = alpha * ir_low_refined + (1 - alpha) * ir_high_refined

        # 4. Fusion (complementarity-aware or residual)
        if self.use_cafg:
            out = self.fusion_gate(rgb_feat, ir_final)
        else:
            out = self.fusion_block(rgb_feat, ir_final)

        return out


# ==========================================
# Experiment: Integrated Fusion: Stage 3-4 (CAFG)
# ==========================================
class NovelSemanticFusionModule(nn.Module):
    """
    Semantic-level fusion with Complementarity-Aware gating for stages 3-4.
    Uses CAFG to explicitly route fusion based on modality agreement.
    """
    def __init__(self, rgb_c, ir_c, reduction=16):
        super().__init__()

        self.ir_proj = nn.Sequential(
            nn.Conv2d(ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU()
        )

        self.cafg = ComplementarityAwareFusionGate(rgb_c)

        self.spatial_refine = nn.Sequential(
            nn.Conv2d(rgb_c, rgb_c, 3, padding=1, groups=rgb_c, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c)
        )

    def forward(self, rgb_feat, ir_feat):
        ir_feat = self.ir_proj(ir_feat)
        fused = self.cafg(rgb_feat, ir_feat)
        refined = self.spatial_refine(fused)
        return fused + refined


class FusionAwareNativeResolutionDecoder(nn.Module):
    """
    Fusion-aware FPN decoder designed for frequency-aware RGB–IR fusion backbones.

    Key properties:
    - Adaptive (learnable) lateral vs top-down merging
    - Global context modulation from deepest feature
    - Native-resolution prediction (e.g. 480x640)
    - Auxiliary head for stable training on MFNet-like datasets
    """

    def __init__(self,
                 in_channels: List[int],
                 num_classes: int,
                 output_resolution: Tuple[int, int] = (480, 640)):
        super().__init__()
        assert len(in_channels) >= 2, "At least two feature levels are required"

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_resolution = output_resolution

        L = len(in_channels)

        # -------------------------------------------------
        # Top-down projections (deep -> shallow)
        # -------------------------------------------------
        self.td_projs = nn.ModuleList([
            nn.Conv2d(in_channels[i + 1], in_channels[i], kernel_size=1, bias=False)
            for i in range(L - 1)
        ])

        # -------------------------------------------------
        # Fusion-aware merge weights (learnable)
        # w -> how much to trust lateral (detail) vs top-down (semantic)
        # -------------------------------------------------
        self.merge_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(L - 1)
        ])

        # -------------------------------------------------
        # FPN refinement convolutions
        # -------------------------------------------------
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i], in_channels[i], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(L - 1)
        ])

        # -------------------------------------------------
        # Global context modulation (deepest feature)
        # -------------------------------------------------
        self.context_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels[-1], in_channels[0], kernel_size=1),
            nn.Sigmoid()
        )

        # -------------------------------------------------
        # Segmentation heads
        # -------------------------------------------------
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[0] // 2, num_classes, kernel_size=1)
        )

        mid = L // 2
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels[mid], in_channels[mid] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[mid] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[mid] // 2, num_classes, kernel_size=1)
        )

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features: list of fused backbone features
                      ordered from shallow -> deep
        """
        L = len(features)
        assert L == len(self.in_channels)

        fpn_features = []

        # -------------------------------------------------
        # Start from deepest feature
        # -------------------------------------------------
        prev = features[-1]
        fpn_features.insert(0, prev)

        # -------------------------------------------------
        # Top-down FPN with fusion-aware merging
        # -------------------------------------------------
        for i in range(L - 2, -1, -1):
            lateral = features[i]

            up = F.interpolate(
                prev,
                size=lateral.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            td = self.td_projs[i](up)

            # Fusion-aware adaptive merge
            w = torch.sigmoid(self.merge_weights[i])
            merged = w * lateral + (1.0 - w) * td

            refined = self.fpn_convs[i](merged)

            prev = refined
            fpn_features.insert(0, prev)

        # -------------------------------------------------
        # Main head (highest resolution)
        # -------------------------------------------------
        main_feat = fpn_features[0]

        # Global context modulation (IR-dominant semantic)
        ctx = self.context_proj(features[-1])  # (B, C, 1, 1)
        main_feat = main_feat * (0.5 + 0.5 * ctx)

        main_logits = self.seg_head(main_feat)
        main_logits = F.interpolate(
            main_logits,
            size=self.output_resolution,
            mode='bilinear',
            align_corners=False
        )

        # -------------------------------------------------
        # Auxiliary head (mid-level supervision)
        # -------------------------------------------------
        mid = L // 2
        aux_feat = fpn_features[mid]
        aux_logits = self.aux_head(aux_feat)
        aux_logits = F.interpolate(
            aux_logits,
            size=self.output_resolution,
            mode='bilinear',
            align_corners=False
        )

        return main_logits, aux_logits


# ==========================================
# PANet-Style Decoder with Deep Supervision
# ==========================================
class PANetDecoder(nn.Module):
    """
    PANet-style decoder with bidirectional feature aggregation.
    
    Key improvements over standard FPN:
    1. Top-down path: Semantic information flows to shallow layers
    2. Bottom-up path: Detail information flows back to deep layers
    3. Deep supervision: Loss at multiple levels for better gradient flow
    4. Multi-level aggregation: Combines all levels for final prediction
    
    This design is particularly effective for:
    - Small object detection (Hand Drill, Guardrail)
    - Class imbalance scenarios
    - Preserving fine-grained details
    """

    def __init__(self,
                 in_channels: List[int],
                 num_classes: int,
                 output_resolution: Tuple[int, int] = (480, 640),
                 deep_supervision: bool = True):
        super().__init__()
        assert len(in_channels) >= 2, "At least two feature levels are required"

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_resolution = output_resolution
        self.deep_supervision = deep_supervision

        L = len(in_channels)
        self.num_levels = L

        # Common channel dimension for aggregation
        self.fpn_channels = in_channels[0]  # Use smallest channel count (96)

        # =================================================
        # 1. Lateral projections (all levels -> fpn_channels)
        # =================================================
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i], self.fpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.fpn_channels),
                nn.ReLU(inplace=True)
            )
            for i in range(L)
        ])

        # =================================================
        # 2. Top-down path (deep -> shallow)
        # =================================================
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.fpn_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(L - 1)
        ])

        # =================================================
        # 3. Bottom-up path (shallow -> deep) - PANet
        # =================================================
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(self.fpn_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(L - 1)
        ])

        # Bottom-up merge convolutions
        self.bu_merge_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.fpn_channels * 2, self.fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.fpn_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(L - 1)
        ])

        # =================================================
        # 4. Global context module (from deepest feature)
        # =================================================
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels[-1], self.fpn_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # =================================================
        # 5. Multi-level aggregation (combine all levels)
        # =================================================
        self.agg_conv = nn.Sequential(
            nn.Conv2d(self.fpn_channels * L, self.fpn_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels * 2, self.fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True)
        )

        # =================================================
        # 6. Segmentation heads
        # =================================================
        # Main head (after aggregation)
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.fpn_channels, num_classes, kernel_size=1)
        )

        # Auxiliary head (mid-level, for training stability)
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.fpn_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels // 2, num_classes, kernel_size=1)
        )

        # Deep supervision heads (one per level, optional)
        if self.deep_supervision:
            self.deep_heads = nn.ModuleList([
                nn.Conv2d(self.fpn_channels, num_classes, kernel_size=1)
                for _ in range(L)
            ])

        print(f"[PANetDecoder] Initialized with {L} levels, fpn_channels={self.fpn_channels}")
        print(f"[PANetDecoder] Deep supervision: {deep_supervision}")

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features: list of fused backbone features [F1, F2, F3, F4]
                      ordered from shallow (high-res) -> deep (low-res)
        
        Returns:
            main_logits: Primary segmentation output
            aux_logits: Auxiliary output (or dict with deep supervision)
        """
        L = len(features)
        assert L == self.num_levels

        # -------------------------------------------------
        # Step 1: Lateral projections (uniform channels)
        # -------------------------------------------------
        laterals = [self.lateral_convs[i](features[i]) for i in range(L)]

        # -------------------------------------------------
        # Step 2: Top-down path (semantic -> detail)
        # -------------------------------------------------
        td_features = [None] * L
        td_features[-1] = laterals[-1]  # Start from deepest

        for i in range(L - 2, -1, -1):
            # Upsample deeper feature
            up = F.interpolate(
                td_features[i + 1],
                size=laterals[i].shape[2:],
                mode='bilinear',
                align_corners=False
            )
            # Add lateral and refine
            td_features[i] = self.td_convs[i](laterals[i] + up)

        # -------------------------------------------------
        # Step 3: Bottom-up path (detail -> semantic) - PANet
        # -------------------------------------------------
        bu_features = [None] * L
        bu_features[0] = td_features[0]  # Start from shallowest

        for i in range(1, L):
            # Downsample shallower feature
            down = self.bu_convs[i - 1](bu_features[i - 1])
            
            # Handle size mismatch (due to stride=2)
            if down.shape[2:] != td_features[i].shape[2:]:
                down = F.interpolate(
                    down,
                    size=td_features[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Concatenate and merge
            merged = torch.cat([td_features[i], down], dim=1)
            bu_features[i] = self.bu_merge_convs[i - 1](merged)

        # -------------------------------------------------
        # Step 4: Global context modulation
        # -------------------------------------------------
        global_ctx = self.global_context(features[-1])  # (B, C, 1, 1)

        # Apply global context to all bottom-up features
        bu_features_ctx = [f * (0.5 + 0.5 * global_ctx) for f in bu_features]

        # -------------------------------------------------
        # Step 5: Multi-level aggregation
        # -------------------------------------------------
        # Upsample all features to highest resolution
        target_size = bu_features_ctx[0].shape[2:]
        upsampled = [bu_features_ctx[0]]
        for i in range(1, L):
            up = F.interpolate(
                bu_features_ctx[i],
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            upsampled.append(up)

        # Concatenate and aggregate
        concat = torch.cat(upsampled, dim=1)  # (B, fpn_channels * L, H, W)
        aggregated = self.agg_conv(concat)

        # -------------------------------------------------
        # Step 6: Main segmentation head
        # -------------------------------------------------
        main_logits = self.seg_head(aggregated)
        main_logits = F.interpolate(
            main_logits,
            size=self.output_resolution,
            mode='bilinear',
            align_corners=False
        )

        # -------------------------------------------------
        # Step 7: Auxiliary and deep supervision
        # -------------------------------------------------
        # Standard auxiliary (mid-level)
        mid = L // 2
        aux_logits = self.aux_head(bu_features[mid])
        aux_logits = F.interpolate(
            aux_logits,
            size=self.output_resolution,
            mode='bilinear',
            align_corners=False
        )

        # Deep supervision outputs (for training only)
        if self.deep_supervision and self.training:
            deep_outputs = []
            for i in range(L):
                deep_out = self.deep_heads[i](bu_features[i])
                deep_out = F.interpolate(
                    deep_out,
                    size=self.output_resolution,
                    mode='bilinear',
                    align_corners=False
                )
                deep_outputs.append(deep_out)
            
            # Return as dict for training
            return main_logits, {
                'aux': aux_logits,
                'deep': deep_outputs  # [level0, level1, level2, level3]
            }

        return main_logits, aux_logits


def test_native_resolution_model(dataset='pst900',model='tiny'):#mfnet
    """
    Test model with dataset-specific configuration
    
    Args:
        dataset: 'mfnet' or 'pst900'
    
    Validates:
    - Model initialization
    - Forward pass
    - Output shapes
    - Phase 1 modules (CBAM, SPP) if present
    """
    import torch

    if model == 'nano':  
        rgb_arch = 'convnextv2_nano.fcmae_ft_in22k_in1k_384'
        ir_arch = 'convnextv2_nano.fcmae_ft_in22k_in1k_384'
        ctx_dim = [80,160,320,640]
    elif model == 'base':
        rgb_arch = 'convnextv2_base.fcmae_ft_in22k_in1k_384'
        ir_arch = 'convnextv2_base.fcmae_ft_in22k_in1k_384'
        ctx_dim = [128,256,512,1024]
    elif model == 'tiny':
        rgb_arch = 'convnextv2_tiny.fcmae_ft_in22k_in1k_384'
        ir_arch = 'convnextv2_tiny.fcmae_ft_in22k_in1k_384'
        ctx_dim = [96,192,384,768]
    else:
        raise ValueError(f"Model must be 'nano', 'base', or 'tiny', got '{model}'")

    """rgb_arch = 'convnextv2_base.fcmae_ft_in22k_in1k_384'
    ir_arch = 'convnextv2_base.fcmae_ft_in22k_in1k_384'
    ctx_dim = [128,256,512,1024]

    rgb_arch = 'convnextv2_tiny.fcmae_ft_in22k_in1k_384'
    ir_arch = 'convnextv2_tiny.fcmae_ft_in22k_in1k_384'
    ctx_dim = [96,192,384,768]
    """
    # Dataset configurations
    configs = {
        'mfnet': {
            'num_classes': 9,
            'resolution': (480, 640),
            'name': 'MFNet',
            'rgb_arch': rgb_arch,
            'ir_arch': ir_arch,
            'context_dim': ctx_dim
        },
        'pst900': {
            'num_classes': 5,
            'resolution': (720, 1280),
            'name': 'PST900',
            'rgb_arch': rgb_arch,
            'ir_arch': ir_arch,
            'context_dim': ctx_dim
        }
    }
    
    if dataset not in configs:
        raise ValueError(f"Dataset must be 'mfnet' or 'pst900', got '{dataset}'")
    
    cfg = configs[dataset]
    batch_size = 2
    
    print("="*60)
    print(f"Testing FusionModel with {cfg['name']} Configuration")
    print("="*60)
    
    print(f"\n📊 Configuration:")
    print(f"  - Dataset: {cfg['name']}")
    print(f"  - Classes: {cfg['num_classes']}")
    print(f"  - Resolution: {cfg['resolution'][0]}x{cfg['resolution'][1]}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Context dim: {ctx_dim}")    
    print(f"  - RGB Arch: {rgb_arch}")
    print(f"  - IR Arch: {ir_arch}")
    
    # Initialize model
    print(f"\n🔧 Initializing FusionModel...")
    model = FusionModel(
        rgb_arch=rgb_arch,
        ir_arch=ir_arch,
        context_dim=ctx_dim,
        num_classes=cfg['num_classes'],
        input_resolution=cfg['resolution'],
        rgb_backbone_resolution=cfg['resolution'],
        ir_backbone_resolution=cfg['resolution'],
        output_resolution=cfg['resolution'],
        decoder_type='fpn',
        enhanced_fusion=True,
        #deep_supervision=True,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📈 Model Statistics:")
    print(f"  - Total parameters: {total_params/1e6:.2f}M")
    print(f"  - Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Calculate GFLOPs
    try:
        from thop import profile
        model_copy = FusionModel(
            rgb_arch=rgb_arch,
            ir_arch=ir_arch,
            context_dim=ctx_dim,
            num_classes=cfg['num_classes'],
            input_resolution=cfg['resolution'],
            rgb_backbone_resolution=cfg['resolution'],
            ir_backbone_resolution=cfg['resolution'],
            output_resolution=cfg['resolution'],
            decoder_type='fpn',
            enhanced_fusion=True,
            #deep_supervision=True,
        )
        model_copy.eval()
        # Create dummy inputs for GFLOPs calculation
        dummy_rgb = torch.randn(1, 3, cfg['resolution'][0], cfg['resolution'][1])
        dummy_ir = torch.randn(1, 1, cfg['resolution'][0], cfg['resolution'][1])
        with torch.no_grad():
            flops, params = profile(model_copy, inputs=(dummy_rgb, dummy_ir), verbose=False)
        print(f"  - GFLOPs: {flops / 1e9:.2f}")
        print(f"  - Params (thop): {params / 1e6:.2f}M")
        del model_copy, dummy_rgb, dummy_ir
    except ImportError:
        print(f"  - GFLOPs: N/A (thop not installed)")
    except Exception as e:
        print(f"  - GFLOPs: Error ({e})")
    
    # Check Phase 1 modules (may not exist for older checkpoints)
    print(f"\n🔍 Phase 1 Module Check:")
    has_cbam3 = hasattr(model, 'cbam_stage3')
    has_cbam4 = hasattr(model, 'cbam_stage4')
    has_spp = hasattr(model, 'spp_stage4')
    
    print(f"  - CBAM Stage 3: {'✅' if has_cbam3 else '❌ (not present)'}")
    print(f"  - CBAM Stage 4: {'✅' if has_cbam4 else '❌ (not present)'}")
    print(f"  - SPP Stage 4: {'✅' if has_spp else '❌ (not present)'}")
    
    if has_cbam3 and has_cbam4 and has_spp:
        print(f"  → Phase 1 enhancements: ACTIVE")
    else:
        print(f"  → Phase 1 enhancements: NOT ACTIVE (baseline model)")
    
    # Create dummy inputs
    print(f"\n🎲 Creating dummy inputs...")
    rgb = torch.randn(batch_size, 3, cfg['resolution'][0], cfg['resolution'][1])
    ir = torch.randn(batch_size, 1, cfg['resolution'][0], cfg['resolution'][1])
    
    print(f"  - RGB shape: {rgb.shape}")
    print(f"  - IR shape: {ir.shape}")
    
    # Forward pass
    print(f"\n▶️  Running forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            main_out, aux_out = model(rgb, ir)
            print(f"  ✅ Forward pass successful!")
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Validate outputs
    print(f"\n📤 Output Validation:")
    expected_shape = (batch_size, cfg['num_classes'], cfg['resolution'][0], cfg['resolution'][1])
    
    print(f"  - Main output shape: {main_out.shape}")
    print(f"  - Expected shape: {expected_shape}")
    
    if main_out.shape == expected_shape:
        print(f"  - Main output: ✅ Correct shape")
    else:
        print(f"  - Main output: ❌ Wrong shape!")
        return None
    
    if aux_out is not None:
        print(f"  - Aux output shape: {aux_out.shape}")
        if aux_out.shape == expected_shape:
            print(f"  - Aux output: ✅ Correct shape")
        else:
            print(f"  - Aux output: ❌ Wrong shape!")
    else:
        print(f"  - Aux output: None (expected for some configs)")
    
    # Check value ranges
    print(f"\n📊 Output Statistics:")
    print(f"  - Main output range: [{main_out.min():.4f}, {main_out.max():.4f}]")
    print(f"  - Main output mean: {main_out.mean():.4f}")
    print(f"  - Main output std: {main_out.std():.4f}")
    
    if aux_out is not None:
        print(f"  - Aux output range: [{aux_out.min():.4f}, {aux_out.max():.4f}]")
    
    # Check for NaN/Inf
    if torch.isnan(main_out).any():
        print(f"  - ❌ Main output contains NaN!")
    elif torch.isinf(main_out).any():
        print(f"  - ❌ Main output contains Inf!")
    else:
        print(f"  - ✅ Main output is clean (no NaN/Inf)")
    
    print(f"\n" + "="*60)
    print(f"✅ {cfg['name']} Test Complete!")
    print(f"="*60)
    return model, main_out, aux_out


if __name__ == "__main__":
    print("\n" + "🚀"*30)
    print("FusionModel Architecture Test Suite")
    print("🚀"*30 + "\n")
    
    # Test PST900 configuration
    print("\n" + "="*70)
    print("TEST 1: PST900 Configuration (5 classes, 720x1280)")
    print("="*70)
    result_pst900 = test_native_resolution_model(dataset='mfnet',model='tiny')
    
    # Test MFNet configuration
    #print("\n\n" + "="*70)
    #print("TEST 2: MFNet Configuration (9 classes, 480x640)")
    #print("="*70)
    #result_mfnet = test_native_resolution_model(dataset='mfnet')
    
    # Summary
    print("\n\n" + "="*70)
    print("📋 Test Summary")
    print("="*70)
    print(f"  - PST900: {'✅ PASS' if result_pst900 is not None else '❌ FAIL'}")
    #print(f"  - MFNet:  {'✅ PASS' if result_mfnet is not None else '❌ FAIL'}")
    print("="*70 + "\n")
