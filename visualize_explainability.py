#!/usr/bin/env python3
"""Visualize semantic-margin, gradient, boundary, and gated-fusion signals."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.polyp5 import Polyp5Model
from utils.checkpoint import load_state_dict
from utils.dataloader import CLIP_MEAN, CLIP_STD, TestDataset


DATASETS = ["CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB"]
IMAGE_SIZE = 352
EPS = 1e-8


def normalize_heatmap(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high - low < EPS:
        low = float(finite.min())
        high = float(finite.max())
    if high - low < EPS:
        return np.zeros_like(finite, dtype=np.float32)
    return np.clip((finite - low) / (high - low), 0.0, 1.0).astype(np.float32)


def gradient_to_heatmap(gradient: torch.Tensor) -> np.ndarray:
    if gradient is None:
        raise ValueError("gradient cannot be None")
    if gradient.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got {tuple(gradient.shape)}")
    reduced = torch.abs(gradient.detach()).mean(dim=1)[0].cpu().numpy()
    return normalize_heatmap(reduced)


def boundary_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8) > 0
    if int(radius) < 1:
        raise ValueError("radius must be positive")
    kernel_size = 2 * int(radius) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(binary.astype(np.uint8), kernel) > 0
    eroded = cv2.erode(binary.astype(np.uint8), kernel) > 0
    return np.logical_xor(dilated, eroded)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def boundary_statistics(heatmap: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = normalize_heatmap(heatmap)
    binary = np.asarray(mask, dtype=bool)
    band = boundary_band(binary, radius=2)
    interior = binary & ~band
    background = ~binary
    boundary_mean = _safe_mean(values[band])
    background_mean = _safe_mean(values[background])
    return {
        "boundary_mean": boundary_mean,
        "interior_mean": _safe_mean(values[interior]),
        "background_mean": background_mean,
        "boundary_background_ratio": float(boundary_mean / (background_mean + EPS)),
    }


def resize_map(array: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.shape == (size, size):
        return values
    return cv2.resize(values, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def resize_mask(mask: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    values = np.asarray(mask, dtype=np.uint8)
    if values.shape == (size, size):
        return values.astype(bool)
    resized = cv2.resize(values, (size, size), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def reduce_feature_map(feature: torch.Tensor) -> np.ndarray:
    if feature.ndim != 4:
        raise ValueError(f"expected feature [B,C,H,W], got {tuple(feature.shape)}")
    return feature.detach().mean(dim=1)[0].cpu().numpy().astype(np.float32)


def signed_display(array: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    limit = float(np.percentile(np.abs(values), 99.0))
    if limit < EPS:
        return np.zeros_like(values)
    return np.clip(values / limit, -1.0, 1.0)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _gradient(score: torch.Tensor, sam_input: torch.Tensor, clip_input: torch.Tensor):
    grad_sam, grad_clip = torch.autograd.grad(
        score,
        (sam_input, clip_input),
        retain_graph=True,
        allow_unused=True,
    )
    if grad_sam is None:
        grad_sam = torch.zeros_like(sam_input)
    if grad_clip is None:
        grad_clip = torch.zeros_like(clip_input)
    return gradient_to_heatmap(grad_sam), gradient_to_heatmap(grad_clip)


def extract_explanations(
    model: torch.nn.Module,
    sam_image: torch.Tensor,
    clip_image: torch.Tensor,
    use_amp: bool = True,
) -> dict[str, np.ndarray]:
    sam_input = sam_image.detach().clone().requires_grad_(True)
    clip_input = clip_image.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    device = sam_input.device
    with _autocast_context(device, use_amp):
        output = model(sam_input, clip_input)
        semantic_score = output["similarity_delta"].mean()
        mask_probability = torch.sigmoid(output["high_res_masks"])
        flat_probability = mask_probability.flatten(1)
        topk_count = max(1, flat_probability.shape[1] // 10)
        mask_score = torch.topk(flat_probability, topk_count, dim=1).values.mean()
        boundary_probability = torch.sigmoid(output["boundary_logits"])
        boundary_score = boundary_probability.mean()

    semantic_grad_sam, semantic_grad_clip = _gradient(
        semantic_score, sam_input, clip_input
    )
    mask_grad_sam, mask_grad_clip = _gradient(mask_score, sam_input, clip_input)
    boundary_grad_sam, boundary_grad_clip = _gradient(
        boundary_score, sam_input, clip_input
    )

    result = {
        "semantic_margin": resize_map(output["similarity_delta"][0, 0].detach().float().cpu().numpy()),
        "semantic_grad_sam": semantic_grad_sam,
        "semantic_grad_clip": semantic_grad_clip,
        "mask_probability": mask_probability[0, 0].detach().float().cpu().numpy(),
        "mask_grad_sam": mask_grad_sam,
        "mask_grad_clip": mask_grad_clip,
        "boundary_response": resize_map(boundary_probability[0, 0].detach().float().cpu().numpy()),
        "boundary_grad_sam": boundary_grad_sam,
        "boundary_grad_clip": boundary_grad_clip,
    }
    for name in (
        "gate_sam22",
        "gate_clip22",
        "gate_sam44",
        "gate_clip44",
        "gate_sam88",
        "gate_clip88",
    ):
        result[name] = resize_map(reduce_feature_map(output[name]))
    return result


def _denormalize_clip(clip_image: torch.Tensor) -> np.ndarray:
    array = clip_image.detach().cpu()[0].numpy().transpose(1, 2, 0)
    mean = np.asarray(CLIP_MEAN, dtype=np.float32)
    std = np.asarray(CLIP_STD, dtype=np.float32)
    return np.clip(array * std + mean, 0.0, 1.0)


def _draw_boundary_overlay(image: np.ndarray, gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    canvas = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8).copy()
    for mask, color in ((gt, (255, 0, 0)), (prediction, (0, 255, 0))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)
    return canvas.astype(np.float32) / 255.0


def _plot_heatmap(ax, array, title, cmap="magma", vmin=None, vmax=None):
    ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_composite(
    image: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray,
    maps: dict[str, np.ndarray],
    output_path: Path,
    title: str,
):
    figure, axes = plt.subplots(4, 4, figsize=(17, 17), dpi=140)
    signed_margin = signed_display(maps["semantic_margin"])
    boundary_overlay = _draw_boundary_overlay(image, gt, prediction)
    semantic_overlay = image * 0.55 + plt.get_cmap("magma")(maps["semantic_grad_clip"])[..., :3] * 0.45
    panels = [
        (image, "Input", None, None, None),
        (boundary_overlay, "GT red / prediction green", None, None, None),
        (signed_margin, "Text margin: polyp - normal", "coolwarm", -1, 1),
        (maps["semantic_grad_clip"], "Semantic gradient (CLIP)", "magma", 0, 1),
        (maps["mask_grad_clip"], "Mask gradient (CLIP)", "magma", 0, 1),
        (maps["mask_grad_sam"], "Mask gradient (SAM2)", "magma", 0, 1),
        (maps["boundary_response"], "Boundary response", "viridis", 0, 1),
        (maps["boundary_grad_clip"], "Boundary gradient (CLIP)", "magma", 0, 1),
        (maps["gate_clip22"], "CLIP gate 22", "viridis", 0, 1),
        (maps["gate_clip44"], "CLIP gate 44", "viridis", 0, 1),
        (maps["gate_clip88"], "CLIP gate 88", "viridis", 0, 1),
        (maps["gate_sam22"], "SAM gate 22", "viridis", 0, 1),
        (maps["gate_sam44"], "SAM gate 44", "viridis", 0, 1),
        (maps["gate_sam88"], "SAM gate 88", "viridis", 0, 1),
        (boundary_overlay, "Boundary overlay", None, None, None),
        (semantic_overlay, "Semantic gradient overlay", None, None, None),
    ]
    for axis, (array, panel_title, cmap, vmin, vmax) in zip(axes.flat, panels):
        if cmap is None:
            axis.imshow(array)
            axis.set_title(panel_title, fontsize=9)
            axis.axis("off")
        else:
            _plot_heatmap(axis, array, panel_title, cmap, vmin, vmax)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _select_index(dataset: TestDataset, dataset_index: int, seed: int) -> int:
    candidates = list(range(len(dataset)))
    random.Random(seed + dataset_index).shuffle(candidates)
    for index in candidates:
        _, _, target, _ = dataset[index]
        if float(target.sum()) > 0:
            return index
    return candidates[0]


def _read_ranks(checkpoint: Path):
    summary_path = checkpoint.parent / "checkpoint_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    return (
        int(config.get("sam_lora_rank", 128)),
        int(config.get("clip_lora_rank", 128)),
        int(config.get("text_lora_rank", 64)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=Path("data/TestDataset"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    run_name = args.checkpoint.parent.name
    model_name = args.checkpoint.stem
    sam_rank, clip_rank, text_rank = _read_ranks(args.checkpoint)
    model = Polyp5Model(
        project_root=args.project_root,
        sam_lora_rank=sam_rank,
        clip_lora_rank=clip_rank,
        text_lora_rank=text_rank,
    )
    model.load_state_dict(load_state_dict(str(args.checkpoint)), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)

    output_root = args.output_root / run_name / model_name
    manifest = []
    for dataset_index, dataset_name in enumerate(DATASETS):
        dataset = TestDataset(str(args.test_path / dataset_name), image_size=IMAGE_SIZE)
        sample_index = _select_index(dataset, dataset_index, args.seed)
        sam_image, clip_image, target, image_name = dataset[sample_index]
        sam_image = sam_image.unsqueeze(0).to(device)
        clip_image = clip_image.unsqueeze(0).to(device)
        maps = extract_explanations(model, sam_image, clip_image, use_amp=not args.no_amp)
        image = _denormalize_clip(clip_image)
        gt = resize_mask(target.cpu().numpy() > 0.5)
        prediction = maps["mask_probability"] >= 0.5
        sample_dir = output_root / dataset_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_stem = Path(image_name).stem
        png_path = sample_dir / f"{image_stem}_explainability.png"
        npz_path = sample_dir / f"{image_stem}_maps.npz"
        json_path = sample_dir / f"{image_stem}_metadata.json"
        render_composite(
            image,
            gt,
            prediction,
            maps,
            png_path,
            f"polyp7 {dataset_name}/{image_name} | {model_name}",
        )
        np.savez_compressed(npz_path, **maps, ground_truth=gt.astype(np.uint8), prediction=prediction.astype(np.uint8))
        stats = {
            name: boundary_statistics(array, gt)
            for name, array in maps.items()
            if array.ndim == 2 and array.shape == gt.shape
        }
        metadata = {
            "dataset": dataset_name,
            "image_name": image_name,
            "sample_index": sample_index,
            "seed": args.seed,
            "checkpoint": str(args.checkpoint),
            "sam_lora_rank": sam_rank,
            "clip_lora_rank": clip_rank,
            "text_lora_rank": text_rank,
            "semantic_target": "mean(similarity_delta)",
            "mask_target": "mean(top_10_percent(sigmoid(high_res_masks)))",
            "boundary_target": "mean(sigmoid(boundary_logits))",
            "ground_truth_foreground_pixels": int(gt.sum()),
            "prediction_foreground_pixels": int(prediction.sum()),
            "boundary_statistics": stats,
            "outputs": [str(png_path), str(npz_path), str(json_path)],
        }
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest.append(metadata)
        print(f"[Explain] {dataset_name}/{image_name} -> {png_path}")

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Done] {len(manifest)} samples -> {output_root}")


if __name__ == "__main__":
    main()
