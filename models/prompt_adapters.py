"""Trainable BiomedCLIP-to-SAM2 prompt adapters."""

import torch
from torch import nn
import torch.nn.functional as F


class SemanticPromptAdapter(nn.Module):
    """Convert one 512-D global visual feature into one SAM2 sparse token."""

    def __init__(self, input_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 2 or feature.shape[-1] != 512:
            raise ValueError(f"expected [B, 512], got {tuple(feature.shape)}")
        return self.proj(self.norm(feature)).unsqueeze(1)


class SoftMaskPromptAdapter(nn.Module):
    """Turn 14x14 BiomedCLIP patch tokens into the 88x88 SAM2 mask prompt."""

    def __init__(self, token_dim: int = 512, hidden_dim: int = 256, output_size: int = 88):
        super().__init__()
        self.output_size = int(output_size)
        self.norm = nn.LayerNorm(token_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(token_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3 or patch_tokens.shape[1:] != (196, 512):
            raise ValueError(f"expected [B, 196, 512], got {tuple(patch_tokens.shape)}")
        x = self.norm(patch_tokens).transpose(1, 2).reshape(-1, 512, 14, 14)
        logits = self.decoder(x)
        logits = F.interpolate(
            logits,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(logits)


class TextPrototypeCostPromptAdapter(nn.Module):
    """Build a SAM2 mask prompt from text and background prototype costs."""

    def __init__(
        self,
        initial_background_features: torch.Tensor,
        hidden_dim: int = 32,
        output_size: int = 88,
    ):
        super().__init__()
        if initial_background_features.ndim != 2 or initial_background_features.shape[-1] != 512:
            raise ValueError(
                "expected initial_background_features [K, 512], "
                f"got {tuple(initial_background_features.shape)}"
            )
        self.output_size = int(output_size)
        self.background_prototypes = nn.Parameter(initial_background_features.clone())
        self.decoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.cost_residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        patch_tokens: torch.Tensor,
        positive_text_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 3 or patch_tokens.shape[1:] != (196, 512):
            raise ValueError(f"expected patch_tokens [B, 196, 512], got {tuple(patch_tokens.shape)}")
        if positive_text_features.ndim != 2 or positive_text_features.shape[-1] != 512:
            raise ValueError(
                "expected positive_text_features [P, 512], "
                f"got {tuple(positive_text_features.shape)}"
            )

        patch_tokens = F.normalize(patch_tokens, dim=-1)
        positive_text_features = F.normalize(positive_text_features, dim=-1)
        background_prototypes = F.normalize(self.background_prototypes, dim=-1)

        positive_cost = (patch_tokens @ positive_text_features.T).mean(dim=-1)
        background_cost = (patch_tokens @ background_prototypes.T).max(dim=-1).values
        positive_cost = positive_cost.reshape(-1, 1, 14, 14)
        background_cost = background_cost.reshape(-1, 1, 14, 14)
        cost_map = positive_cost - background_cost

        prompt_logits = self.decoder(torch.cat((positive_cost, background_cost, cost_map), dim=1))
        prompt_logits = prompt_logits + self.cost_residual_scale * cost_map
        prompt_logits = F.interpolate(
            prompt_logits,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return {
            "mask_prompt": torch.sigmoid(prompt_logits),
            "prompt_logits": prompt_logits,
            "positive_cost": positive_cost,
            "background_cost": background_cost,
            "cost_map": cost_map,
        }
