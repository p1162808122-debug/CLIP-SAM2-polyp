"""Minimal symmetric CLIP contrastive loss for BiomedCLIP training."""

import torch
from torch.nn import functional as F


def clip_loss(image_features, text_features, logit_scale, labels=None):
    """Compute the symmetric image-to-text and text-to-image InfoNCE loss."""

    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("image_features and text_features must be rank-2 tensors")
    if image_features.shape[0] != text_features.shape[0]:
        raise ValueError("image and text batches must have the same size")

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    if labels is None:
        labels = torch.arange(image_features.shape[0], device=image_features.device)
    return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2


__all__ = ["clip_loss"]
