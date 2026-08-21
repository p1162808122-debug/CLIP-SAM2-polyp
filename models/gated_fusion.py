"""Three-scale dual-gated fusion of SAM2 and BiomedCLIP image features."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class _DualGate(nn.Module):
    """Produce independent sigmoid gates for two same-width feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        input_channels = int(channels) * 2
        self.sam_gate = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.clip_gate = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, sam_feature: torch.Tensor, clip_feature: torch.Tensor):
        joined = torch.cat([sam_feature, clip_feature], dim=1)
        gate_sam = self.sam_gate(joined)
        gate_clip = self.clip_gate(joined)
        fused = gate_sam * sam_feature + gate_clip * clip_feature
        return fused, gate_sam, gate_clip


class GatedMultiScaleFusion(nn.Module):
    """Fuse SAM2 and BiomedCLIP features at 22, 44, and 88 pixel grids."""

    def __init__(
        self,
        image_channels: int = 256,
        high_res_channels: Tuple[int, int] = (64, 32),
    ):
        super().__init__()
        self.image_channels = int(image_channels)
        self.high_res_channels = tuple(int(value) for value in high_res_channels)
        if self.high_res_channels != (64, 32):
            raise ValueError(
                "SAM2-polyp7 expects high_res_channels=(64, 32), "
                f"got {self.high_res_channels}"
            )

        self.clip_to_44 = nn.Conv2d(self.image_channels, 64, kernel_size=1)
        self.clip_to_88 = nn.Conv2d(self.image_channels, 32, kernel_size=1)
        self.gate22 = _DualGate(self.image_channels)
        self.gate44 = _DualGate(64)
        self.gate88 = _DualGate(32)

    @staticmethod
    def _validate_feature(name: str, feature: torch.Tensor, expected):
        if feature.ndim != 4 or tuple(feature.shape[1:]) != tuple(expected):
            raise ValueError(
                f"{name} must be [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(feature.shape)}"
            )

    def forward(
        self,
        sam22: torch.Tensor,
        clip22: torch.Tensor,
        sam44: torch.Tensor,
        sam88: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._validate_feature("sam22", sam22, (self.image_channels, 22, 22))
        self._validate_feature("clip22", clip22, (self.image_channels, 22, 22))
        self._validate_feature("sam44", sam44, (64, 44, 44))
        self._validate_feature("sam88", sam88, (32, 88, 88))
        batch = sam22.shape[0]
        if any(feature.shape[0] != batch for feature in (clip22, sam44, sam88)):
            raise ValueError("all gated fusion features must have the same batch size")

        clip44 = self.clip_to_44(
            F.interpolate(clip22, size=(44, 44), mode="bilinear", align_corners=False)
        )
        clip88 = self.clip_to_88(
            F.interpolate(clip22, size=(88, 88), mode="bilinear", align_corners=False)
        )

        fused22, gate_sam22, gate_clip22 = self.gate22(sam22, clip22)
        fused44, gate_sam44, gate_clip44 = self.gate44(sam44, clip44)
        fused88, gate_sam88, gate_clip88 = self.gate88(sam88, clip88)
        return {
            "fused22": fused22,
            "fused44": fused44,
            "fused88": fused88,
            "gate_sam22": gate_sam22,
            "gate_clip22": gate_clip22,
            "gate_sam44": gate_sam44,
            "gate_clip44": gate_clip44,
            "gate_sam88": gate_sam88,
            "gate_clip88": gate_clip88,
        }


__all__ = ["GatedMultiScaleFusion"]
