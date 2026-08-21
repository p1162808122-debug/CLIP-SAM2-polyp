"""Portable BiomedCLIP model built from vendored OpenCLIP source.

The complete visual tower and Hugging Face text tower implementation lives in
``model/open_clip``.  This file only provides a stable project-level API and
redirects the Hugging Face configuration/tokenizer to local files under
``pretrained/text_encoder`` so the project can be moved without downloading
the model definition again.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch import nn

from . import open_clip as _open_clip
from .open_clip.factory import _MODEL_CONFIGS


PathLike = Union[str, Path]
_LOCAL_MODEL_NAME = "biomedclip_local"


def _read_model_bundle(pretrained_dir: Path, weight_path: Optional[Path] = None):
    config_path = pretrained_dir / "open_clip_config.json"
    text_encoder_dir = pretrained_dir / "text_encoder"

    if weight_path is None:
        local_weight_path = pretrained_dir / "open_clip_pytorch_model.bin"
        # The SAM2-GA layout keeps the large CLIP checkpoint in the shared
        # project-level pretrained directory, next to SAM2 weights.
        external_weight_path = (
            pretrained_dir.parent.parent.parent / "pretrained" / "open_clip_pytorch_model.bin"
        )
        weight_candidates = (local_weight_path, external_weight_path)
        weight_path = next(
            (candidate for candidate in weight_candidates if candidate.is_file()),
            local_weight_path,
        )
    else:
        weight_path = Path(weight_path).expanduser().resolve()

    missing = [
        str(path)
        for path in (config_path, weight_path, text_encoder_dir / "config.json")
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "BiomedCLIP pretrained bundle is incomplete. Missing: " + ", ".join(missing)
        )

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    model_cfg = copy.deepcopy(payload.get("model_cfg", payload))
    if not all(key in model_cfg for key in ("embed_dim", "vision_cfg", "text_cfg")):
        raise ValueError(f"Invalid OpenCLIP model config: {config_path}")

    model_cfg["text_cfg"]["hf_model_name"] = str(text_encoder_dir)
    model_cfg["text_cfg"]["hf_tokenizer_name"] = str(text_encoder_dir)
    model_cfg["text_cfg"]["hf_model_pretrained"] = False

    preprocess_cfg = payload.get("preprocess_cfg", {})
    return model_cfg, preprocess_cfg, weight_path, text_encoder_dir


class BiomedCLIP(nn.Module):
    """BiomedCLIP with local OpenCLIP image/text encoder source.

    ``self.model.visual`` contains the complete ViT image encoder.  For this
    model ``self.model.text`` contains the complete PubMedBERT-based text
    encoder and its projection.  Both towers remain trainable by default.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        preprocess_train,
        preprocess_val,
        pretrained_dir: Path,
        model_cfg: dict,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val
        self.pretrained_dir = pretrained_dir
        self.model_cfg = model_cfg

    @classmethod
    def from_pretrained(
        cls,
        pretrained_dir: PathLike,
        device: Optional[Union[str, torch.device]] = None,
        precision: str = "fp32",
        freeze_image: bool = False,
        freeze_text: bool = False,
        weight_path: Optional[PathLike] = None,
        image_size: Optional[Union[int, tuple[int, int]]] = None,
    ) -> "BiomedCLIP":
        """Load the official BiomedCLIP checkpoint from a local directory.

        ``pretrained_dir`` may be the ``BioMedCLIP`` project root or its
        ``pretrained`` subdirectory. If the large checkpoint is stored in the
        parent SAM2-GA ``pretrained`` directory, it is discovered
        automatically; ``weight_path`` can override that location.
        """

        pretrained_dir = Path(pretrained_dir).expanduser().resolve()
        if not (pretrained_dir / "open_clip_config.json").is_file() and (
            pretrained_dir / "pretrained"
        ).is_dir():
            pretrained_dir = pretrained_dir / "pretrained"
        resolved_weight_path = Path(weight_path).expanduser().resolve() if weight_path else None
        model_cfg, preprocess_cfg, resolved_weight_path, _ = _read_model_bundle(
            pretrained_dir, weight_path=resolved_weight_path
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The upstream factory expects a named config.  Register the local
        # official config in its in-memory registry without writing another
        # copy of the configuration into the source tree.
        _MODEL_CONFIGS[_LOCAL_MODEL_NAME] = model_cfg

        transform_kwargs = {}
        if preprocess_cfg.get("mean") is not None:
            transform_kwargs["image_mean"] = tuple(preprocess_cfg["mean"])
        if preprocess_cfg.get("std") is not None:
            transform_kwargs["image_std"] = tuple(preprocess_cfg["std"])
        if preprocess_cfg.get("interpolation") is not None:
            transform_kwargs["image_interpolation"] = preprocess_cfg["interpolation"]
        if preprocess_cfg.get("resize_mode") is not None:
            transform_kwargs["image_resize_mode"] = preprocess_cfg["resize_mode"]

        model_kwargs = {}
        if image_size is not None:
            model_kwargs["force_image_size"] = image_size
        model, preprocess_train, preprocess_val = _open_clip.create_model_and_transforms(
            _LOCAL_MODEL_NAME,
            pretrained=str(resolved_weight_path),
            precision=precision,
            device=device,
            pretrained_hf=False,
            output_dict=False,
            **model_kwargs,
            **transform_kwargs,
        )
        tokenizer = _open_clip.get_tokenizer(_LOCAL_MODEL_NAME)

        instance = cls(
            model=model,
            tokenizer=tokenizer,
            preprocess_train=preprocess_train,
            preprocess_val=preprocess_val,
            pretrained_dir=pretrained_dir,
            model_cfg=model_cfg,
        )
        instance.set_trainable(image=not freeze_image, text=not freeze_text, projection=True)
        return instance

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode images with the vendored OpenCLIP ViT image encoder."""

        return self.model.encode_image(images, normalize=normalize)

    def encode_image_tokens(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Return the projected CLS-plus-patch token sequence from the timm ViT."""

        visual = self.model.visual
        if not hasattr(visual, "forward_tokens"):
            raise RuntimeError("This BiomedCLIP visual tower does not expose token features")
        features = visual.forward_tokens(images)
        if features.ndim != 3:
            raise RuntimeError(f"Expected BiomedCLIP token features [B, N, C], got {tuple(features.shape)}")
        return torch.nn.functional.normalize(features, dim=-1) if normalize else features

    def encode_text(self, tokens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode token IDs with the vendored PubMedBERT text encoder."""

        if hasattr(self.model, "text"):
            features = self.model.text(tokens)
            if normalize:
                features = torch.nn.functional.normalize(features, dim=-1)
            return features
        return self.model.encode_text(tokens, normalize=normalize)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ):
        """Return ``(image_features, text_features, logit_scale)``."""

        image_features = (
            self.encode_image(images, normalize=normalize) if images is not None else None
        )
        text_features = (
            self.encode_text(tokens, normalize=normalize) if tokens is not None else None
        )
        return image_features, text_features, self.model.logit_scale.exp()

    def set_trainable(self, image: bool = True, text: bool = True, projection: bool = True):
        """Selectively enable gradients for image/text towers and logit scale."""

        for name, parameter in self.model.named_parameters():
            if name.startswith("visual."):
                parameter.requires_grad = image
            elif name.startswith("text.") or name.startswith("transformer."):
                parameter.requires_grad = text
            elif name.startswith("token_embedding.") or name.startswith("positional_embedding"):
                parameter.requires_grad = text
            else:
                parameter.requires_grad = projection
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Yield only parameters enabled for optimization."""

        return (parameter for parameter in self.parameters() if parameter.requires_grad)


def load_biomedclip(pretrained_dir: PathLike, **kwargs) -> BiomedCLIP:
    """Functional alias for :meth:`BiomedCLIP.from_pretrained`."""

    return BiomedCLIP.from_pretrained(pretrained_dir, **kwargs)


__all__ = ["BiomedCLIP", "load_biomedclip"]
