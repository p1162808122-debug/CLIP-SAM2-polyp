from pathlib import Path
from typing import Dict
import csv
import json

import numpy as np
from PIL import Image


EPS = 1e-8
FIXED_THRESHOLD = 0.5


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64)


def _find_gt(mask_dir: Path, name: str) -> Path:
    stem = Path(name).stem
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = mask_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Ground-truth mask not found for {name} in {mask_dir}")


def _normalize(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float64)
    return (array - array.min()) / (array.max() - array.min() + EPS)


def evaluate_dataset(
    pred_dir: Path,
    gt_dir: Path,
    threshold: float = FIXED_THRESHOLD,
) -> Dict[str, object]:
    """Evaluate one dataset with a single fixed threshold after per-image min-max normalization."""
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    predictions = sorted(pred_dir.glob("*.png"))
    if not predictions:
        raise RuntimeError(f"No prediction png files found in {pred_dir}")

    dice_scores = []
    iou_scores = []

    for prediction_path in predictions:
        pred = _normalize(_read_mask(prediction_path) / 255.0)
        gt = _read_mask(_find_gt(gt_dir, prediction_path.name)) > 128

        if pred.shape != gt.shape:
            pred = np.asarray(
                Image.fromarray((pred * 255).astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float64,
            ) / 255.0
            pred = _normalize(pred)

        binary = pred >= threshold
        gt_pixels = np.count_nonzero(gt)
        pred_pixels = np.count_nonzero(binary)
        intersection = np.count_nonzero(binary & gt)

        dice_scores.append(2.0 * intersection / (gt_pixels + pred_pixels + EPS))
        iou_scores.append(
            intersection / (gt_pixels + pred_pixels - intersection + EPS)
        )

    return {
        "count": len(predictions),
        "threshold": threshold,
        "meanDic": float(np.mean(dice_scores)),
        "meanIoU": float(np.mean(iou_scores)),
    }


def write_result_files(
    output_dir: Path,
    run_name: str,
    model_name: str,
    results: Dict[str, Dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {}
    csv_path = output_dir / f"{model_name}_summary.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["run", "model", "dataset", "count", "threshold", "meanDic", "meanIoU"]
        )

        for dataset, result in results.items():
            json_payload[dataset] = dict(result)
            writer.writerow(
                [
                    run_name,
                    model_name,
                    dataset,
                    result["count"],
                    result["threshold"],
                    result["meanDic"],
                    result["meanIoU"],
                ]
            )
            np.savez(
                output_dir / f"{dataset}.npz",
                count=result["count"],
                threshold=result["threshold"],
                meanDic=result["meanDic"],
                meanIoU=result["meanIoU"],
            )

    (output_dir / f"{model_name}_summary.json").write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )
