"""Three-stage CFBR refinement after the static SAM2 mask decoder.

The CFBR blocks mirror RePraNet3/lib/CFBR.py.  Polyp6 uses the three
available SAM2 decoder feature scales (22, 44, and 88) instead of the four
backbone levels used by RePraNet3's reverse-attention decoder.
"""

from typing import Dict, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ContextExplorationBlock(nn.Module):
    """The context exploration block copied from RePraNet3's CFBR."""

    def __init__(self, input_channels: int):
        super().__init__()
        self.input_channels = int(input_channels)
        self.channels_single = self.input_channels // 4
        c = self.channels_single

        self.channel_mixing = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
        )

        self.b1 = BasicConv2d(c, c, kernel_size=3, padding=1, dilation=1)
        self.b2 = BasicConv2d(c, c, kernel_size=3, padding=2, dilation=2)
        self.b3 = BasicConv2d(c, c, kernel_size=5, padding=2, dilation=1)
        self.b4 = BasicConv2d(c, c, kernel_size=5, padding=4, dilation=2)

        def gate():
            return nn.Sequential(
                nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.Sigmoid(),
            )

        self.gate_x1, self.gate_h1 = gate(), gate()
        self.gate_x2, self.gate_h2 = gate(), gate()
        self.gate_x3, self.gate_h3 = gate(), gate()
        self.fusion = BasicConv2d(
            self.input_channels,
            self.input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

    def forward(self, x):
        x = self.channel_mixing(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        p1 = self.b4(x1)
        g_x1 = self.gate_x1(torch.cat([x2, p1], dim=1))
        g_h1 = self.gate_h1(torch.cat([x2, p1], dim=1))
        p2 = self.b3(g_h1 * x2 + g_x1 * p1)

        g_x2 = self.gate_x2(torch.cat([x3, p2], dim=1))
        g_h2 = self.gate_h2(torch.cat([x3, p2], dim=1))
        p3 = self.b2(g_h2 * x3 + g_x2 * p2)

        g_x3 = self.gate_x3(torch.cat([x4, p3], dim=1))
        g_h3 = self.gate_h3(torch.cat([x4, p3], dim=1))
        # Preserve the RePraNet3 implementation exactly, including its x2
        # term in the final context branch.
        p4 = self.b1(g_h3 * x2 + g_x3 * p3)

        return self.fusion(torch.cat([p1, p2, p3, p4], dim=1))


class CFBR(nn.Module):
    """Contextual foreground-background refinement block."""

    def __init__(self, channel: int):
        super().__init__()
        self.channel = int(channel)
        self.fp = ContextExplorationBlock(self.channel)
        self.fn = ContextExplorationBlock(self.channel)
        self.alpha = nn.Parameter(torch.ones(self.channel, 1, 1))
        self.beta = nn.Parameter(torch.ones(self.channel, 1, 1))
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map):
        f_feature = x * in_map
        b_feature = x * (1 - in_map)
        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = self.relu1(self.bn1(y - self.alpha * fp))
        refine2 = self.relu2(self.bn2(refine1 + self.beta * fn))
        return refine2


class CFBRRefinementDecoder(nn.Module):
    """Refine the selected SAM2 mask through 22x22, 44x44, and 88x88."""

    num_cfbr_stages = 3

    def __init__(self, output_size: int = 352):
        super().__init__()
        self.output_size = int(output_size)

        self.focus22 = CFBR(256)
        self.focus44 = CFBR(64)
        self.focus88 = CFBR(32)

        self.ra22_conv1 = BasicConv2d(256, 128, kernel_size=1)
        self.ra22_conv2 = nn.Conv2d(128, 1, kernel_size=1)
        self.ra44_conv1 = BasicConv2d(64, 32, kernel_size=1)
        self.ra44_conv2 = nn.Conv2d(32, 1, kernel_size=1)
        self.ra88_conv1 = BasicConv2d(32, 16, kernel_size=1)
        self.ra88_conv2 = nn.Conv2d(16, 1, kernel_size=1)

        self.gamma22 = nn.Parameter(torch.ones(1))
        self.gamma44 = nn.Parameter(torch.ones(1))
        self.gamma88 = nn.Parameter(torch.ones(1))

    @staticmethod
    def _resize(x: torch.Tensor, size) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)

    @staticmethod
    def _validate(
        initial_mask_logits: torch.Tensor,
        combined_image_feature: torch.Tensor,
        high_res_features: Sequence[torch.Tensor],
    ) -> None:
        if initial_mask_logits.ndim != 4 or initial_mask_logits.shape[1:] != (1, 88, 88):
            raise ValueError(
                "initial_mask_logits must be [B,1,88,88], "
                f"got {tuple(initial_mask_logits.shape)}"
            )
        if combined_image_feature.ndim != 4 or combined_image_feature.shape[1:] != (256, 22, 22):
            raise ValueError(
                "combined_image_feature must be [B,256,22,22], "
                f"got {tuple(combined_image_feature.shape)}"
            )
        if len(high_res_features) != 2:
            raise ValueError(f"high_res_features must contain [88x88, 44x44], got {len(high_res_features)} maps")
        expected = ((32, 88, 88), (64, 44, 44))
        for index, (feature, shape) in enumerate(zip(high_res_features, expected)):
            if feature.ndim != 4 or tuple(feature.shape[1:]) != shape:
                raise ValueError(
                    f"high_res_features[{index}] must be [B,{shape[0]},{shape[1]},{shape[2]}], "
                    f"got {tuple(feature.shape)}"
                )
        batch = initial_mask_logits.shape[0]
        if combined_image_feature.shape[0] != batch or any(feature.shape[0] != batch for feature in high_res_features):
            raise ValueError("all CFBR inputs must have the same batch size")

    def forward(
        self,
        initial_mask_logits: torch.Tensor,
        combined_image_feature: torch.Tensor,
        high_res_features: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self._validate(initial_mask_logits, combined_image_feature, high_res_features)
        feature88, feature44 = high_res_features

        crop22 = self._resize(initial_mask_logits, (22, 22))
        x22 = self.focus22(combined_image_feature, combined_image_feature, crop22)
        refined22 = self.ra22_conv2(self.ra22_conv1(x22)) + self.gamma22 * crop22

        crop44 = self._resize(refined22, (44, 44))
        x44 = self.focus44(feature44, feature44, crop44)
        refined44 = self.ra44_conv2(self.ra44_conv1(x44)) + self.gamma44 * crop44

        crop88 = self._resize(refined44, (88, 88))
        x88 = self.focus88(feature88, feature88, crop88)
        refined88 = self.ra88_conv2(self.ra88_conv1(x88)) + self.gamma88 * crop88

        refined_high_res = F.interpolate(
            refined88.float(),
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        ).to(refined88.dtype)
        return {
            "refined_22_logits": refined22,
            "refined_44_logits": refined44,
            "refined_88_logits": refined88,
            "refined_low_res_masks": refined88,
            "refined_high_res_masks": refined_high_res,
        }


__all__ = ["BasicConv2d", "ContextExplorationBlock", "CFBR", "CFBRRefinementDecoder"]
