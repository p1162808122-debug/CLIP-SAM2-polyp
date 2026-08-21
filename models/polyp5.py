"""Gated static SAM2 + BiomedCLIP prompting with three-stage CFBR for polyp7."""

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from BioMedCLIP.model.biomedclip import BiomedCLIP
from sam2.build_sam import build_sam2
from utils.losses import boundary_similarity_logits

from .lora import (
    count_lora_parameters,
    inject_lora_into_hf_text,
    inject_lora_into_hiera,
    inject_lora_into_vit,
    set_lora_train_mode,
)
from .cfbr_decoder import CFBRRefinementDecoder
from .static_sam2 import StaticSAM2GatedPrompt


POLYP5_POSITIVE_TEXT = "a colonoscopy image showing a colorectal polyp"
POLYP5_NEGATIVE_TEXT = "a normal colonoscopy image showing healthy colon tissue"
POLYP5_TEXTS = (POLYP5_POSITIVE_TEXT, POLYP5_NEGATIVE_TEXT)

POLYP5_SAM_LORA_RANKS = (128, 256, 512)
POLYP5_CLIP_LORA_RANKS = (128, 256)
POLYP5_TEXT_LORA_RANKS = (64, 128)


def resolve_polyp5_ranks(
    legacy_lora_rank: Optional[int] = None,
    sam_lora_rank: Optional[int] = None,
    clip_lora_rank: Optional[int] = None,
    text_lora_rank: Optional[int] = None,
) -> tuple[int, int, int]:
    """Resolve independent ranks while preserving the old single-rank API."""

    if legacy_lora_rank is not None:
        legacy = int(legacy_lora_rank)
        sam_lora_rank = legacy if sam_lora_rank is None else sam_lora_rank
        clip_lora_rank = legacy if clip_lora_rank is None else clip_lora_rank
        text_lora_rank = legacy if text_lora_rank is None else text_lora_rank
    sam = 128 if sam_lora_rank is None else int(sam_lora_rank)
    clip = 128 if clip_lora_rank is None else int(clip_lora_rank)
    text = 64 if text_lora_rank is None else int(text_lora_rank)
    if sam not in POLYP5_SAM_LORA_RANKS:
        raise ValueError(f"SAM2 Image Encoder rank must be one of {POLYP5_SAM_LORA_RANKS}, got {sam}")
    if clip not in POLYP5_CLIP_LORA_RANKS:
        raise ValueError(f"BiomedCLIP visual rank must be one of {POLYP5_CLIP_LORA_RANKS}, got {clip}")
    if text not in POLYP5_TEXT_LORA_RANKS:
        raise ValueError(f"BiomedCLIP text rank must be one of {POLYP5_TEXT_LORA_RANKS}, got {text}")
    return sam, clip, text


def polyp5_rank_label(sam_lora_rank: int, clip_lora_rank: int, text_lora_rank: int) -> str:
    sam, _, _ = resolve_polyp5_ranks(sam_lora_rank=sam_lora_rank)
    _, clip, _ = resolve_polyp5_ranks(clip_lora_rank=clip_lora_rank)
    _, _, text = resolve_polyp5_ranks(text_lora_rank=text_lora_rank)
    return f"sam{sam}_clip{clip}_text{text}"


class ClipDensePromptAdapter(nn.Module):
    """Map raw 512-D CLIP patch tokens into SAM2 dense-prompt space."""

    def __init__(self, input_dim: int = 512, output_dim: int = 256, grid_size: int = 22):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.grid_size = int(grid_size)
        self.norm = nn.LayerNorm(self.input_dim)
        self.proj = nn.Linear(self.input_dim, self.output_dim)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.grid_size * self.grid_size
        if patch_tokens.ndim != 3 or patch_tokens.shape[1:] != (expected_tokens, self.input_dim):
            raise ValueError(
                f"expected [B, {expected_tokens}, {self.input_dim}], got {tuple(patch_tokens.shape)}"
            )
        projected = self.proj(self.norm(patch_tokens)) * self.gate
        batch = projected.shape[0]
        return projected.transpose(1, 2).reshape(
            batch, self.output_dim, self.grid_size, self.grid_size
        )


def patch_text_similarity(
    patch_tokens: torch.Tensor,
    text_features: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    grid_size: int = 22,
) -> torch.Tensor:
    """Compute two-class patch/text similarities while still in 512-D space."""

    expected_tokens = int(grid_size) * int(grid_size)
    if patch_tokens.ndim != 3 or patch_tokens.shape[1:] != (expected_tokens, 512):
        raise ValueError(f"expected patch_tokens [B, {expected_tokens}, 512], got {tuple(patch_tokens.shape)}")
    if text_features.ndim != 2 or text_features.shape[-1] != 512:
        raise ValueError(f"expected text_features [T, 512], got {tuple(text_features.shape)}")
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    similarities = torch.einsum("bnc,tc->bnt", patch_tokens, text_features)
    similarities = similarities.transpose(1, 2).reshape(
        patch_tokens.shape[0], text_features.shape[0], grid_size, grid_size
    )
    if scale is not None:
        similarities = similarities * scale.to(dtype=similarities.dtype, device=similarities.device)
    return similarities


class Polyp5Model(nn.Module):
    """Polyp7 with three-scale dual-gated SAM2/BiomedCLIP fusion."""

    SUPPORTED_LORA_RANKS = (128,)
    SUPPORTED_SAM_LORA_RANKS = POLYP5_SAM_LORA_RANKS
    SUPPORTED_CLIP_LORA_RANKS = POLYP5_CLIP_LORA_RANKS
    SUPPORTED_TEXT_LORA_RANKS = POLYP5_TEXT_LORA_RANKS
    IMAGE_SIZE = 352
    CLIP_GRID_SIZE = 22

    def __init__(
        self,
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[float] = None,
        lora_dropout: float = 0.0,
        project_root: Optional[Path] = None,
        sam_lora_rank: Optional[int] = None,
        clip_lora_rank: Optional[int] = None,
        text_lora_rank: Optional[int] = None,
        boundary_threshold: float = 0.1,
        boundary_temperature: float = 0.05,
    ):
        super().__init__()
        self.sam_lora_rank, self.clip_lora_rank, self.text_lora_rank = resolve_polyp5_ranks(
            legacy_lora_rank=lora_rank,
            sam_lora_rank=sam_lora_rank,
            clip_lora_rank=clip_lora_rank,
            text_lora_rank=text_lora_rank,
        )
        self.lora_rank = self.sam_lora_rank if (
            self.sam_lora_rank == self.clip_lora_rank == self.text_lora_rank
        ) else None
        self.lora_alpha = lora_alpha
        self.lora_dropout = float(lora_dropout)
        if boundary_threshold <= 0:
            raise ValueError("boundary_threshold must be positive")
        if boundary_temperature <= 0:
            raise ValueError("boundary_temperature must be positive")
        self.boundary_threshold = float(boundary_threshold)
        self.boundary_temperature = float(boundary_temperature)
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])

        sam_checkpoint = self.project_root.parent / "pretrained" / "sam2_hiera_large.pt"
        if not sam_checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {sam_checkpoint}")
        self.sam2 = build_sam2(
            config_file="sam2_hiera_l_352",
            ckpt_path=str(sam_checkpoint),
            device="cpu",
            mode="eval",
            apply_postprocessing=False,
        )
        self._freeze_module(self.sam2)
        self.sam_lora_target_names = inject_lora_into_hiera(
            self.sam2.image_encoder.trunk,
            rank=self.sam_lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        self._unfreeze_module(self.sam2.sam_mask_decoder)

        self.biomedclip = BiomedCLIP.from_pretrained(
            self.project_root / "BioMedCLIP",
            device="cpu",
            precision="fp32",
            freeze_image=True,
            freeze_text=True,
            image_size=self.IMAGE_SIZE,
        )
        self._freeze_module(self.biomedclip)
        visual_trunk = getattr(self.biomedclip.model.visual, "trunk", None)
        if visual_trunk is None:
            raise TypeError("The local BiomedCLIP visual tower must expose a timm trunk")
        self.clip_lora_target_names = inject_lora_into_vit(
            visual_trunk,
            rank=self.clip_lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        self.text_lora_target_names = inject_lora_into_hf_text(
            self.biomedclip.model.text,
            rank=self.text_lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        self.register_buffer(
            "polyp5_prompt_tokens",
            self.biomedclip.tokenizer(list(POLYP5_TEXTS)),
        )

        self.prompt_adapter = ClipDensePromptAdapter(grid_size=self.CLIP_GRID_SIZE)
        self.static_sam2 = StaticSAM2GatedPrompt(self.sam2)
        self.cfbr_decoder = CFBRRefinementDecoder(output_size=self.IMAGE_SIZE)

        self.sam_lora_parameter_count = count_lora_parameters(self.sam2.image_encoder.trunk)
        self.clip_lora_parameter_count = count_lora_parameters(visual_trunk)
        self.text_lora_parameter_count = count_lora_parameters(self.biomedclip.model.text)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _unfreeze_module(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = True

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen towers in eval mode, while LoRA and the requested decoder
        # follow the caller's train/eval mode.
        self.sam2.eval()
        self.biomedclip.eval()
        self.sam2.sam_mask_decoder.train(mode)
        self.prompt_adapter.train(mode)
        self.cfbr_decoder.train(mode)
        set_lora_train_mode(self.sam2, mode)
        set_lora_train_mode(self.biomedclip, mode)
        return self

    def forward(self, sam_images: torch.Tensor, clip_images: torch.Tensor):
        clip_tokens = self.biomedclip.encode_image_tokens(clip_images, normalize=False)
        expected_tokens = self.CLIP_GRID_SIZE * self.CLIP_GRID_SIZE + 1
        if clip_tokens.ndim != 3 or clip_tokens.shape[1:] != (expected_tokens, 512):
            raise RuntimeError(
                f"BiomedCLIP 352 token output must be [B, {expected_tokens}, 512], "
                f"got {tuple(clip_tokens.shape)}"
            )

        patch_tokens_512 = clip_tokens[:, 1:]
        text_features_512 = self.biomedclip.encode_text(
            self.polyp5_prompt_tokens,
            normalize=True,
        )
        logit_scale = self.biomedclip.model.logit_scale.exp().detach()
        similarity_cosine = patch_text_similarity(
            patch_tokens_512,
            text_features_512,
            scale=None,
            grid_size=self.CLIP_GRID_SIZE,
        )
        similarity_logits = similarity_cosine * logit_scale.to(
            dtype=similarity_cosine.dtype,
            device=similarity_cosine.device,
        )
        similarity_delta = similarity_cosine[:, 0:1] - similarity_cosine[:, 1:2]
        boundary_logits = boundary_similarity_logits(
            similarity_delta,
            threshold=self.boundary_threshold,
            temperature=self.boundary_temperature,
        )

        dense_prompt = self.prompt_adapter(patch_tokens_512)
        sam_outputs = self.static_sam2(sam_images, dense_prompt)
        refinement = self.cfbr_decoder(
            sam_outputs["selected_mask_logits"],
            sam_outputs["fused_image_embeddings"],
            sam_outputs["fused_high_res_features"],
        )
        outputs = {
            **sam_outputs,
            **refinement,
            "sam_mask_logits": sam_outputs["selected_mask_logits"],
            "sam_high_res_masks": sam_outputs["high_res_masks"],
            "low_res_masks": refinement["refined_low_res_masks"],
            "high_res_masks": refinement["refined_high_res_masks"],
        }
        return {
            **outputs,
            "similarity_logits": similarity_logits,
            "similarity_pos": similarity_logits[:, 0:1],
            "similarity_neg": similarity_logits[:, 1:2],
            "similarity_cosine": similarity_cosine,
            "similarity_delta": similarity_delta,
            "boundary_logits": boundary_logits,
            "text_features_512": text_features_512,
            "clip_patch_tokens_512": patch_tokens_512,
        }

    def trainable_parameter_summary(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "sam_lora_rank": self.sam_lora_rank,
            "clip_lora_rank": self.clip_lora_rank,
            "text_lora_rank": self.text_lora_rank,
            "sam_lora_parameters": self.sam_lora_parameter_count,
            "clip_lora_parameters": self.clip_lora_parameter_count,
            "text_lora_parameters": self.text_lora_parameter_count,
            "sam_lora_targets": len(self.sam_lora_target_names),
            "clip_lora_targets": len(self.clip_lora_target_names),
            "text_lora_targets": len(self.text_lora_target_names),
            "prompt_adapter_parameters": sum(p.numel() for p in self.prompt_adapter.parameters()),
            "gated_fusion_parameters": sum(p.numel() for p in self.static_sam2.feature_fusion.parameters()),
            "cfbr_decoder_parameters": sum(p.numel() for p in self.cfbr_decoder.parameters()),
            "mask_decoder_parameters": sum(p.numel() for p in self.sam2.sam_mask_decoder.parameters()),
            "prompt_encoder_parameters_trainable": sum(
                p.numel() for p in self.sam2.sam_prompt_encoder.parameters() if p.requires_grad
            ),
        }


__all__ = [
    "POLYP5_POSITIVE_TEXT",
    "POLYP5_NEGATIVE_TEXT",
    "POLYP5_TEXTS",
    "POLYP5_SAM_LORA_RANKS",
    "POLYP5_CLIP_LORA_RANKS",
    "POLYP5_TEXT_LORA_RANKS",
    "resolve_polyp5_ranks",
    "polyp5_rank_label",
    "ClipDensePromptAdapter",
    "patch_text_similarity",
    "Polyp5Model",
]
