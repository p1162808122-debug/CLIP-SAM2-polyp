"""Differentiable static-image path through SAM2's image/prompt/mask heads."""

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .gated_fusion import GatedMultiScaleFusion


class StaticSAM2(nn.Module):
    """Use SAM2 image features and SAM heads without any video-memory call."""

    def __init__(self, sam2_model: nn.Module):
        super().__init__()
        self.sam2 = sam2_model

    def _empty_points(self, batch: int, device: torch.device):
        coords = torch.zeros(batch, 1, 2, device=device)
        labels = torch.full((batch, 1), -1, dtype=torch.int32, device=device)
        return coords, labels

    def forward(
        self,
        images: torch.Tensor,
        semantic_prompt: Optional[torch.Tensor] = None,
        mask_prompt: Optional[torch.Tensor] = None,
    ):
        backbone_out = self.sam2.forward_image(images)
        image_embeddings = backbone_out["vision_features"]
        feature_levels = backbone_out["backbone_fpn"][-self.sam2.num_feature_levels :]
        high_res_features = feature_levels[:-1] if self.sam2.use_high_res_features_in_sam else None

        if getattr(self.sam2, "directly_add_no_mem_embed", False):
            no_mem = self.sam2.no_mem_embed.transpose(1, 2).reshape(1, -1, 1, 1)
            image_embeddings = image_embeddings + no_mem

        return self.forward_embeddings(
            image_embeddings,
            high_res_features,
            semantic_prompt=semantic_prompt,
            mask_prompt=mask_prompt,
        )

    def forward_embeddings(
        self,
        image_embeddings: torch.Tensor,
        high_res_features,
        semantic_prompt: Optional[torch.Tensor] = None,
        mask_prompt: Optional[torch.Tensor] = None,
    ):
        batch = image_embeddings.shape[0]
        device = image_embeddings.device
        points = self._empty_points(batch, device)

        if mask_prompt is not None:
            expected_size = self.sam2.sam_prompt_encoder.mask_input_size
            if tuple(mask_prompt.shape[-2:]) != tuple(expected_size):
                mask_prompt = F.interpolate(
                    mask_prompt.float(), size=expected_size, mode="bilinear", align_corners=False
                )

        sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=mask_prompt,
        )
        if semantic_prompt is not None:
            if semantic_prompt.ndim != 3 or semantic_prompt.shape[0] != batch or semantic_prompt.shape[-1] != 256:
                raise ValueError(f"expected semantic prompt [B, 1, 256], got {tuple(semantic_prompt.shape)}")
            sparse_embeddings = torch.cat([sparse_embeddings, semantic_prompt], dim=1)

        low_res_masks, ious, _, object_score_logits = self.sam2.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam2.sam_prompt_encoder.get_dense_pe().to(image_embeddings),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(self.sam2.image_size, self.sam2.image_size),
            mode="bilinear",
            align_corners=False,
        ).to(low_res_masks.dtype)
        return {
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "ious": ious,
            "object_score_logits": object_score_logits,
            "semantic_prompt": semantic_prompt,
            "mask_prompt": mask_prompt,
        }


class StaticSAM2DensePrompt(nn.Module):
    """Run SAM2's static image and mask heads with an external dense prompt.

    The prompt encoder is deliberately not touched.  ``dense_prompt`` is
    already in SAM2's decoder space and the image encoder's FPN positional
    encoding is passed directly to the decoder transformer.
    """

    def __init__(self, sam2_model: nn.Module):
        super().__init__()
        self.sam2 = sam2_model

    def forward(self, images: torch.Tensor, dense_prompt: torch.Tensor):
        backbone_out = self.sam2.forward_image(images)
        image_embeddings = backbone_out["vision_features"]
        batch = image_embeddings.shape[0]
        if dense_prompt.ndim != 4 or tuple(dense_prompt.shape) != (
            batch,
            image_embeddings.shape[1],
            image_embeddings.shape[2],
            image_embeddings.shape[3],
        ):
            raise ValueError(
                "dense_prompt must match SAM2 image embeddings, got "
                f"{tuple(dense_prompt.shape)} vs {tuple(image_embeddings.shape)}"
            )

        feature_levels = backbone_out["backbone_fpn"][-self.sam2.num_feature_levels :]
        high_res_features = (
            feature_levels[:-1] if self.sam2.use_high_res_features_in_sam else None
        )

        if getattr(self.sam2, "directly_add_no_mem_embed", False):
            no_mem = self.sam2.no_mem_embed.transpose(1, 2).reshape(1, -1, 1, 1)
            image_embeddings = image_embeddings + no_mem
        combined_image_embeddings = image_embeddings + dense_prompt

        vision_pos_enc = backbone_out.get("vision_pos_enc")
        if not vision_pos_enc:
            raise RuntimeError("SAM2 image encoder did not return vision_pos_enc")
        image_pe = vision_pos_enc[-1]
        if image_pe.ndim != 4 or tuple(image_pe.shape[-2:]) != tuple(image_embeddings.shape[-2:]):
            raise ValueError(
                "SAM2 image positional encoding does not match image embeddings: "
                f"{tuple(image_pe.shape)} vs {tuple(image_embeddings.shape)}"
            )
        # MaskDecoder repeats image_pe across the batch and requires a single
        # leading batch element, matching SAM2's normal get_dense_pe() path.
        image_pe = image_pe[:1].to(dtype=image_embeddings.dtype)
        sparse_prompt_embeddings = image_embeddings.new_empty(batch, 0, image_embeddings.shape[1])

        low_res_masks, ious, _, object_score_logits = self.sam2.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(self.sam2.image_size, self.sam2.image_size),
            mode="bilinear",
            align_corners=False,
        ).to(low_res_masks.dtype)
        return {
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "ious": ious,
            "object_score_logits": object_score_logits,
            "dense_prompt": dense_prompt,
            "image_pe": image_pe,
            "selected_mask_logits": low_res_masks,
            "combined_image_embeddings": combined_image_embeddings,
            "high_res_features": high_res_features,
        }


class StaticSAM2GatedPrompt(nn.Module):
    """Fuse SAM2 and BiomedCLIP features before both SAM2 and CFBR decoders."""

    def __init__(self, sam2_model: nn.Module):
        super().__init__()
        self.sam2 = sam2_model
        self.feature_fusion = GatedMultiScaleFusion()

    @staticmethod
    def _validate_high_res_features(high_res_features):
        if high_res_features is None or len(high_res_features) != 2:
            raise ValueError("SAM2-polyp7 requires two high-resolution feature levels")
        expected = ((32, 88, 88), (64, 44, 44))
        for index, (feature, shape) in enumerate(zip(high_res_features, expected)):
            if feature.ndim != 4 or tuple(feature.shape[1:]) != shape:
                raise ValueError(
                    f"high_res_features[{index}] must be [B,{shape[0]},{shape[1]},{shape[2]}], "
                    f"got {tuple(feature.shape)}"
                )

    def forward(self, images: torch.Tensor, clip_feature: torch.Tensor):
        backbone_out = self.sam2.forward_image(images)
        image_embeddings = backbone_out["vision_features"]
        batch = image_embeddings.shape[0]
        if clip_feature.ndim != 4 or tuple(clip_feature.shape) != (
            batch,
            image_embeddings.shape[1],
            image_embeddings.shape[2],
            image_embeddings.shape[3],
        ):
            raise ValueError(
                "clip_feature must match SAM2 22x22 image embeddings, got "
                f"{tuple(clip_feature.shape)} vs {tuple(image_embeddings.shape)}"
            )

        feature_levels = backbone_out["backbone_fpn"][-self.sam2.num_feature_levels :]
        raw_high_res_features = (
            feature_levels[:-1] if self.sam2.use_high_res_features_in_sam else None
        )
        self._validate_high_res_features(raw_high_res_features)

        if getattr(self.sam2, "directly_add_no_mem_embed", False):
            no_mem = self.sam2.no_mem_embed.transpose(1, 2).reshape(1, -1, 1, 1)
            image_embeddings = image_embeddings + no_mem

        fused = self.feature_fusion(
            image_embeddings,
            clip_feature,
            raw_high_res_features[1],
            raw_high_res_features[0],
        )
        fused_image_embeddings = fused["fused22"]
        fused_high_res_features = [fused["fused88"], fused["fused44"]]

        vision_pos_enc = backbone_out.get("vision_pos_enc")
        if not vision_pos_enc:
            raise RuntimeError("SAM2 image encoder did not return vision_pos_enc")
        image_pe = vision_pos_enc[-1]
        if image_pe.ndim != 4 or tuple(image_pe.shape[-2:]) != tuple(fused_image_embeddings.shape[-2:]):
            raise ValueError(
                "SAM2 image positional encoding does not match fused image embeddings: "
                f"{tuple(image_pe.shape)} vs {tuple(fused_image_embeddings.shape)}"
            )
        image_pe = image_pe[:1].to(dtype=fused_image_embeddings.dtype)
        sparse_prompt_embeddings = fused_image_embeddings.new_empty(
            batch, 0, fused_image_embeddings.shape[1]
        )
        # The CLIP feature is already present in all three fused maps.  A zero
        # dense prompt prevents MaskDecoder from adding it a second time.
        zero_dense_prompt = torch.zeros_like(fused_image_embeddings)
        low_res_masks, ious, _, object_score_logits = self.sam2.sam_mask_decoder(
            image_embeddings=fused_image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=zero_dense_prompt,
            multimask_output=False,
            repeat_image=False,
            high_res_features=fused_high_res_features,
        )
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(self.sam2.image_size, self.sam2.image_size),
            mode="bilinear",
            align_corners=False,
        ).to(low_res_masks.dtype)
        return {
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "ious": ious,
            "object_score_logits": object_score_logits,
            "dense_prompt": clip_feature,
            "image_pe": image_pe,
            "selected_mask_logits": low_res_masks,
            "combined_image_embeddings": fused_image_embeddings,
            "high_res_features": fused_high_res_features,
            "fused_image_embeddings": fused_image_embeddings,
            "fused_high_res_features": fused_high_res_features,
            "raw_image_embeddings": image_embeddings,
            "raw_high_res_features": raw_high_res_features,
            "gate_sam22": fused["gate_sam22"],
            "gate_clip22": fused["gate_clip22"],
            "gate_sam44": fused["gate_sam44"],
            "gate_clip44": fused["gate_clip44"],
            "gate_sam88": fused["gate_sam88"],
            "gate_clip88": fused["gate_clip88"],
        }


__all__ = ["StaticSAM2", "StaticSAM2DensePrompt", "StaticSAM2GatedPrompt"]
