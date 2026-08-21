"""Complete static SAM2 + BiomedCLIP polyp segmenters."""

from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn

from BioMedCLIP.model.biomedclip import BiomedCLIP
from sam2.build_sam import build_sam2

from .lora import (
    count_lora_parameters,
    inject_lora_into_hf_text,
    inject_lora_into_hiera,
    inject_lora_into_vit,
    set_lora_train_mode,
)
from .prompt_adapters import SemanticPromptAdapter, SoftMaskPromptAdapter, TextPrototypeCostPromptAdapter
from .static_sam2 import StaticSAM2


POLYP3_POSITIVE_PROMPTS = (
    "an endoscopic image of a colorectal polyp",
    "a colorectal polyp in colonoscopy",
    "a polyp lesion on the colonic mucosa",
)
POLYP3_BACKGROUND_PROMPTS = (
    "normal colonic mucosa in a colonoscopy image",
    "background tissue in a colonoscopy image",
    "a normal colonic fold in an endoscopic image",
    "a non-polyp region in a colonoscopy image",
)


class PolypSAM2Model(nn.Module):
    SUPPORTED_LORA_RANKS = (32, 64, 128)

    def __init__(
        self,
        variant: str,
        lora_rank: int,
        lora_alpha: Optional[float] = None,
        lora_dropout: float = 0.0,
        project_root: Optional[Path] = None,
        text_lora: bool = False,
    ):
        super().__init__()
        if variant not in {"polyp1", "polyp2", "polyp3"}:
            raise ValueError(f"variant must be polyp1, polyp2, or polyp3, got {variant}")
        if lora_rank not in self.SUPPORTED_LORA_RANKS:
            raise ValueError(
                f"lora_rank must be one of {self.SUPPORTED_LORA_RANKS}, got {lora_rank}"
            )

        self.variant = variant
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(2 * lora_rank if lora_alpha is None else lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.text_lora = bool(text_lora)

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
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        self._unfreeze_module(self.sam2.sam_prompt_encoder)
        self._unfreeze_module(self.sam2.sam_mask_decoder)

        self.biomedclip = BiomedCLIP.from_pretrained(
            self.project_root / "BioMedCLIP",
            device="cpu",
            precision="fp32",
            freeze_image=True,
            freeze_text=True,
        )
        self._freeze_module(self.biomedclip)
        visual_trunk = getattr(self.biomedclip.model.visual, "trunk", None)
        if visual_trunk is None:
            raise TypeError("The local BiomedCLIP visual tower must expose a timm trunk")
        self.clip_lora_target_names = inject_lora_into_vit(
            visual_trunk,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )

        self.text_lora_target_names = ()
        self.text_lora_parameter_count = 0
        if variant == "polyp3":
            prompt_tokens = self.biomedclip.tokenizer(
                [*POLYP3_POSITIVE_PROMPTS, *POLYP3_BACKGROUND_PROMPTS]
            )
            self.register_buffer("polyp3_prompt_tokens", prompt_tokens)
            self.biomedclip.eval()
            with torch.no_grad():
                prompt_features = self.biomedclip.encode_text(prompt_tokens, normalize=True)
            self.register_buffer(
                "polyp3_positive_text_features",
                prompt_features[:len(POLYP3_POSITIVE_PROMPTS)].clone(),
            )
            self.prompt_adapter = TextPrototypeCostPromptAdapter(
                initial_background_features=prompt_features[len(POLYP3_POSITIVE_PROMPTS):],
                hidden_dim=32,
                output_size=88,
            )
            if self.text_lora:
                self.text_lora_target_names = inject_lora_into_hf_text(
                    self.biomedclip.model.text,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                )
                self.text_lora_parameter_count = count_lora_parameters(self.biomedclip.model.text)

        self.static_sam2 = StaticSAM2(self.sam2)
        if variant == "polyp1":
            self.prompt_adapter = SemanticPromptAdapter()
        elif variant == "polyp2":
            self.prompt_adapter = SoftMaskPromptAdapter(output_size=88)

        self.sam_lora_parameter_count = count_lora_parameters(self.sam2.image_encoder.trunk)
        self.clip_lora_parameter_count = count_lora_parameters(visual_trunk)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _unfreeze_module(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = True

    def _trainable_modules(self):
        return (
            self.sam2.sam_prompt_encoder,
            self.sam2.sam_mask_decoder,
            self.prompt_adapter,
        )

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen towers stay in eval mode; only LoRA and requested heads follow train/eval.
        self.sam2.eval()
        self.biomedclip.eval()
        set_lora_train_mode(self.sam2, mode)
        set_lora_train_mode(self.biomedclip, mode)
        for module in self._trainable_modules():
            module.train(mode)
        return self

    def forward(self, sam_images: torch.Tensor, clip_images: torch.Tensor):
        clip_tokens = self.biomedclip.encode_image_tokens(clip_images, normalize=True)
        if clip_tokens.ndim != 3 or clip_tokens.shape[-1] != 512:
            raise RuntimeError(f"BiomedCLIP token output must be [B, 197, 512], got {tuple(clip_tokens.shape)}")

        if self.variant == "polyp1":
            semantic_prompt = self.prompt_adapter(clip_tokens[:, 0])
            return self.static_sam2(sam_images, semantic_prompt=semantic_prompt)

        patch_tokens = clip_tokens[:, 1:]
        if self.variant == "polyp2":
            mask_prompt = self.prompt_adapter(patch_tokens)
            return self.static_sam2(sam_images, mask_prompt=mask_prompt)

        positive_text_features = self.polyp3_positive_text_features
        if self.text_lora:
            positive_text_features = self.biomedclip.encode_text(
                self.polyp3_prompt_tokens[:len(POLYP3_POSITIVE_PROMPTS)],
                normalize=True,
            )
        cost_prompt = self.prompt_adapter(patch_tokens, positive_text_features)
        outputs = self.static_sam2(sam_images, mask_prompt=cost_prompt["mask_prompt"])
        return {**outputs, **cost_prompt}

    def trainable_parameter_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "sam_lora_parameters": self.sam_lora_parameter_count,
            "clip_lora_parameters": self.clip_lora_parameter_count,
            "sam_lora_targets": len(self.sam_lora_target_names),
            "clip_lora_targets": len(self.clip_lora_target_names),
            "text_lora_parameters": self.text_lora_parameter_count,
            "text_lora_targets": len(self.text_lora_target_names),
            "background_prototypes": (
                self.prompt_adapter.background_prototypes.numel()
                if self.variant == "polyp3"
                else 0
            ),
        }
