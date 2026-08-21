import torch
import torch.nn.functional as F


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction="mean")
    pred_soft = torch.sigmoid(pred)
    inter = (pred_soft * mask).sum(dim=(2, 3))
    union = (pred_soft + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    iou = iou.mean()

    edge_zone = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    edge_error = torch.abs(pred_soft - mask)
    edge_loss = (edge_error * edge_zone).sum(dim=(2, 3)) / (edge_zone.sum(dim=(2, 3)) + 1e-6)
    return bce + iou.mean() + 0.5 * edge_loss.mean()


def dice_score(pred_probability: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (pred_probability > threshold).float()
    target = target.float()
    intersection = (pred * target).flatten(1).sum(dim=1)
    denominator = pred.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    return (2.0 * intersection / (denominator + 1e-6)).mean().item()


def prompt_supervision_loss(prompt_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    target = F.interpolate(target_mask.float(), size=prompt_logits.shape[-2:], mode="area")
    bce = F.binary_cross_entropy_with_logits(prompt_logits, target)
    prompt_probability = torch.sigmoid(prompt_logits)
    intersection = (prompt_probability * target).sum(dim=(2, 3))
    denominator = prompt_probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_loss = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
    return bce + dice_loss.mean()


def soft_patch_similarity_loss(similarity_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Classify each 22x22 patch using the 512-D positive/negative scores."""

    if similarity_logits.ndim != 4 or similarity_logits.shape[1] != 2:
        raise ValueError(
            f"expected similarity_logits [B, 2, H, W], got {tuple(similarity_logits.shape)}"
        )
    target = F.interpolate(
        target_mask.float(), size=similarity_logits.shape[-2:], mode="area"
    ).squeeze(1).clamp(0.0, 1.0)
    binary_logits = similarity_logits[:, 0] - similarity_logits[:, 1]
    per_patch = F.binary_cross_entropy_with_logits(binary_logits, target, reduction="none")

    # Give foreground and background equal total weight while retaining
    # fractional labels for patches cut by the object boundary.
    positive_mass = target.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    negative_mass = (1.0 - target).sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    weights = target / positive_mass + (1.0 - target) / negative_mass
    return (per_patch * weights).sum() / weights.sum().clamp_min(1e-6)


def morphological_boundary_target(
    target_mask: torch.Tensor,
    output_size: tuple[int, int] = (22, 22),
    radius: int = 8,
) -> torch.Tensor:
    """Build a soft morphological boundary and area-pool it to patch space."""

    if target_mask.ndim != 4 or target_mask.shape[1] != 1:
        raise ValueError(
            f"expected target_mask [B, 1, H, W], got {tuple(target_mask.shape)}"
        )
    if int(radius) < 1:
        raise ValueError(f"radius must be positive, got {radius}")
    mask = target_mask.float().clamp(0.0, 1.0)
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=int(radius))
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=int(radius))
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    return F.interpolate(boundary, size=tuple(output_size), mode="area")


def boundary_similarity_logits(
    similarity_delta: torch.Tensor,
    threshold: float = 0.1,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Return differentiable boundary logits from raw positive-minus-negative cosine scores."""

    if similarity_delta.ndim != 4 or similarity_delta.shape[1] != 1:
        raise ValueError(
            f"expected similarity_delta [B, 1, H, W], got {tuple(similarity_delta.shape)}"
        )
    if float(threshold) <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return (float(threshold) - similarity_delta.abs()) / float(temperature)


def boundary_similarity_loss(
    similarity_delta: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.1,
    temperature: float = 0.05,
    radius: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boundary_target = morphological_boundary_target(
        target_mask,
        output_size=tuple(similarity_delta.shape[-2:]),
        radius=radius,
    ).to(dtype=similarity_delta.dtype, device=similarity_delta.device)
    boundary_logits = boundary_similarity_logits(
        similarity_delta,
        threshold=threshold,
        temperature=temperature,
    )
    per_patch = F.binary_cross_entropy_with_logits(
        boundary_logits,
        boundary_target,
        reduction="none",
    )
    positive_mass = boundary_target.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    negative_mass = (1.0 - boundary_target).sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    weights = boundary_target / positive_mass + (1.0 - boundary_target) / negative_mass
    loss = (per_patch * weights).sum() / weights.sum().clamp_min(1e-6)
    return loss, boundary_logits, boundary_target


def compute_polyp4_loss(
    output: dict[str, torch.Tensor],
    masks: torch.Tensor,
    similarity_loss_weight: float = 0.5,
) -> torch.Tensor:
    """Combine final-mask structure loss and the 512-D patch alignment loss."""

    if "similarity_logits" not in output:
        raise KeyError("polyp4 output must contain similarity_logits")
    mask_loss = structure_loss(output["high_res_masks"], masks)
    similarity_loss = soft_patch_similarity_loss(output["similarity_logits"], masks)
    return mask_loss + float(similarity_loss_weight) * similarity_loss


def compute_polyp5_loss_components(
    output: dict[str, torch.Tensor],
    masks: torch.Tensor,
    similarity_loss_weight: float = 0.5,
    boundary_loss_weight: float = 0.25,
    boundary_threshold: float = 0.1,
    boundary_temperature: float = 0.05,
    boundary_radius: int = 8,
) -> dict[str, torch.Tensor]:
    """Return mask, patch-alignment, boundary, and total losses for polyp5."""

    required = ("high_res_masks", "similarity_logits", "similarity_delta")
    missing = [name for name in required if name not in output]
    if missing:
        raise KeyError(f"polyp5 output is missing required tensors: {missing}")
    mask_loss = structure_loss(output["high_res_masks"], masks)
    similarity_loss = soft_patch_similarity_loss(output["similarity_logits"], masks)
    boundary_loss, computed_boundary_logits, boundary_target = boundary_similarity_loss(
        output["similarity_delta"],
        masks,
        threshold=boundary_threshold,
        temperature=boundary_temperature,
        radius=boundary_radius,
    )
    boundary_logits = output.get("boundary_logits", computed_boundary_logits)
    total_loss = (
        mask_loss
        + float(similarity_loss_weight) * similarity_loss
        + float(boundary_loss_weight) * boundary_loss
    )
    return {
        "mask_loss": mask_loss,
        "similarity_loss": similarity_loss,
        "boundary_loss": boundary_loss,
        "total_loss": total_loss,
        "boundary_logits": boundary_logits,
        "boundary_target": boundary_target,
    }


def compute_polyp5_loss(
    output: dict[str, torch.Tensor],
    masks: torch.Tensor,
    similarity_loss_weight: float = 0.5,
    boundary_loss_weight: float = 0.25,
    boundary_threshold: float = 0.1,
    boundary_temperature: float = 0.05,
    boundary_radius: int = 8,
) -> torch.Tensor:
    return compute_polyp5_loss_components(
        output,
        masks,
        similarity_loss_weight=similarity_loss_weight,
        boundary_loss_weight=boundary_loss_weight,
        boundary_threshold=boundary_threshold,
        boundary_temperature=boundary_temperature,
        boundary_radius=boundary_radius,
    )["total_loss"]


def compute_training_loss(
    output: dict[str, torch.Tensor], masks: torch.Tensor, prompt_loss_weight: float
) -> torch.Tensor:
    loss = structure_loss(output["high_res_masks"], masks)
    if prompt_loss_weight > 0 and "prompt_logits" in output:
        loss = loss + float(prompt_loss_weight) * prompt_supervision_loss(output["prompt_logits"], masks)
    return loss
