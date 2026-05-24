from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class IQAVQANet(nn.Module):
    """
    IQA/VQA Unified Network

    ## Architecture:

    - Backbone: Swin-T or ResNet50

    - Temporal Fusion: Transformer (Video Only)

    - Regression Head: 3-Layer MLP

    ## Features:

    - Automatically switches between image (4D) and video (5D) modes based on input dimension

    - Loss Function: 0.7*MSE + 0.3*RankLoss
    """

    def __init__(self, config: Dict):
        super().__init__()
        model_cfg = config.get("model", {})
        self.backbone_name = model_cfg.get("backbone", "swin_t")
        self.dropout_rate = model_cfg.get("dropout", 0.3)
        self.freeze_backbone = model_cfg.get("freeze_backbone", False)
        self.num_vqa_layers = model_cfg.get("transformer_layers", 4)
        self.num_frames = model_cfg.get("num_frames", 8)

        # ----------------------------------------------------
        # 1. Backbone: Swin-T or ResNet50
        # ----------------------------------------------------
        if self.backbone_name == "swin_t":
            # Load the pre-trained Swin-Transformer backbone network
            swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            self.backbone = swin.features
            self.num_features = swin.head.in_features  # 768
            self.is_transformer = True

        elif self.backbone_name == "resnet50":
            res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, res.layer3, res.layer4)
            self.num_features = res.fc.in_features  # 2048
            self.is_transformer = False
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_name}")

        # Backbone network freeze switch mechanism
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ----------------------------------------------------
        # 2. Adaptive Pooling: Outputs a fixed-size feature map
        # ----------------------------------------------------
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # ----------------------------------------------------
        # 3. Transformer Timing Fusion (Video Only)
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.num_features,
            nhead=8,
            dim_feedforward=self.num_features * 2,
            dropout=self.dropout_rate,
            activation="gelu" if self.is_transformer else "relu",
            batch_first=True,
        )
        self.temporal_fusion = nn.TransformerEncoder(encoder_layer, num_layers=self.num_vqa_layers)

        # ----------------------------------------------------
        # 4. Regression Head: 3-Layer MLP
        # ----------------------------------------------------
        self.quality_head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.GELU() if self.is_transformer else nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 256),
            nn.GELU() if self.is_transformer else nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Output range (0, 1), aligned with the normalized MOS.
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier Initialize Regression Header Parameters"""
        for m in self.quality_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _forward_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features and compress them into vectors."""
        features = self.backbone(x)

        # The output shape of Swin-T varies depending on the torchvision version; it is uniformly converted to [B, C, H, W]
        if features.dim() == 4:
            # Determine whether the value is [B, C, H, W] or [B, H, W, C].
            if features.shape[-1] == self.num_features and features.shape[1] != self.num_features:
                # [B, H, W, C] -> [B, C, H, W]
                features = features.permute(0, 3, 1, 2)

        elif features.dim() == 3:
            # If the form is [B, L, C] -> dynamically restore H and W.
            B, L, C = features.shape
            H = W = int(L**0.5)
            if H * W == L:
                features = features.permute(0, 2, 1).view(B, C, H, W)

        pooled = self.spatial_pool(features)  # Compress to [B, num_features, 1, 1]
        return torch.flatten(pooled, 1)  # Flattened to [B, num_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Routing based on input dimension:

        - 4D Tensor [B, 3, H, W] -> Images, directly extract spatial features

        - 5D Tensor [B, F, 3, H, W] -> Videos, extract features from each frame first, then perform temporal fusion
        """

        # Adapt grayscale images
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
        elif x.dim() == 5 and x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)  # [B, F, 1, H, W] -> [B, F, 3, H, W]

        if x.dim() == 4:
            v_global = self._forward_backbone_features(x)
            score = self.quality_head(v_global)
            return score.squeeze(-1)  # Ensures the one-dimensional tensor [B] is returned.

        elif x.dim() == 5:
            B, F, C, H, W = x.shape

            # If the input frame rate differs from the configuration, perform sampling or issue warnings here.
            if F != self.num_frames:
                # Number of target frames sampled
                indices = np.linspace(0, F - 1, self.num_frames, dtype=int)
                x = x[:, indices, :, :, :]
                B, F, C, H, W = x.shape

            # Ensure it is a 3-channel image.
            if C != 3:
                raise ValueError(f"Expected 3 channels after grayscale conversion, got {C}")

            # 1. Extract spatial features from each frame
            x_reshaped = x.view(B * F, C, H, W)
            v_frames = self._forward_backbone_features(x_reshaped)

            # 2. Organize into a frame sequence
            v = v_frames.view(B, F, self.num_features)

            # 3. Transformer Timing Fusion
            v_fused = self.temporal_fusion(v)

            # 4. Frame-based average pooling
            v_global = torch.mean(v_fused, dim=1)

            # 5. Regression Header Outputs Quality Score
            score = self.quality_head(v_global)
            return score.squeeze(-1)  # Ensures the one-dimensional tensor is returned as [B]

        else:
            raise ValueError(f"Invalid input tensor dim: {x.dim()}. Expected 4D for IQA or 5D for VQA.")


class IQAVQALoss(nn.Module):
    """
    Loss Function: 0.7*MSE + 0.3*RankLoss
    """

    def __init__(self, config: Dict):
        super().__init__()
        loss_cfg = config.get("loss", {})
        model_cfg = config.get("model", {})
        self.num_frames = model_cfg.get("num_frames", 8)
        self.mse_weight = loss_cfg.get("mse_weight", 0.7)
        self.rank_weight = loss_cfg.get("rank_weight", 0.3)
        self.max_pairs = loss_cfg.get("max_pairs", 5000)  # Add sampling limit configuration
        self.mse_loss = nn.MSELoss()

    def rank_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Sampling method to calculate rank loss"""
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        n = len(y_pred)

        if n > self.max_pairs:
            # Random sampling
            idx = torch.randperm(n)[: self.max_pairs]
            y_pred = y_pred[idx]
            y_true = y_true[idx]
            n = self.max_pairs

        pred_diff = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(0) - y_true.unsqueeze(1)

        # Use torch.clamp for truncation and smoothing to ensure that
        # the gradient has an upper bound on a single pair of samples,
        # thus avoiding gradient explosion.
        loss = torch.relu(-pred_diff * true_diff).mean()
        return loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, model: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        # Standardize and regulate dimensions to eliminate potential problems caused by dimensional mismatches.
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1).float()

        # Truncate values ​​that exceed the boundary to avoid NaN.
        y_pred = torch.clamp(y_pred, min=1e-6, max=1.0 - 1e-6)

        mse = self.mse_loss(y_pred, y_true)
        rank = self.rank_loss(y_pred, y_true)
        total_loss = self.mse_weight * mse + self.rank_weight * rank

        # 📌 The param loop that previously exhausted video memory for map building has been completely removed.
        # L2 regularization is implemented using AdamW's weight_decay parameter.
        return {"total_loss": total_loss, "mse_loss": mse, "rank_loss": rank}
