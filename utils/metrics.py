from pathlib import Path
from typing import Dict, Iterable
import csv
import json

import numpy as np
from PIL import Image


EPS = 1e-8


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


def evaluate_dataset(pred_dir: Path, gt_dir: Path) -> Dict[str, object]:
    predictions = sorted(pred_dir.glob("*.png"))
    if not predictions:
        raise RuntimeError(f"No prediction png files found in {pred_dir}")
    thresholds = np.linspace(1.0, 0.0, 256)
    dice = np.zeros((len(predictions), len(thresholds)), dtype=np.float64)
    iou = np.zeros_like(dice)
    for index, prediction_path in enumerate(predictions):
        pred = _normalize(_read_mask(prediction_path) / 255.0)
        gt = _read_mask(_find_gt(gt_dir, prediction_path.name)) > 128
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
            pred = _normalize(pred)
        gt_pixels = np.count_nonzero(gt)
        for threshold_index, threshold in enumerate(thresholds):
            binary = pred >= threshold
            pred_pixels = np.count_nonzero(binary)
            intersection = np.count_nonzero(binary & gt)
            dice[index, threshold_index] = 2.0 * intersection / (gt_pixels + pred_pixels + EPS)
            iou[index, threshold_index] = intersection / (gt_pixels + pred_pixels - intersection + EPS)
    mean_dice_curve = dice.mean(axis=0)
    mean_iou_curve = iou.mean(axis=0)
    return {
        "meanDic": float(mean_dice_curve.mean()),
        "maxDic": float(mean_dice_curve.max()),
        "meanIoU": float(mean_iou_curve.mean()),
        "maxIoU": float(mean_iou_curve.max()),
        "column_Dic": mean_dice_curve,
        "column_IoU": mean_iou_curve,
    }


def write_result_files(output_dir: Path, run_name: str, model_name: str, results: Dict[str, Dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {}
    csv_path = output_dir / f"{model_name}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "model", "dataset", "meanDic", "maxDic", "meanIoU", "maxIoU"])
        for dataset, result in results.items():
            json_payload[dataset] = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
            writer.writerow([run_name, model_name, dataset, result["meanDic"], result["maxDic"], result["meanIoU"], result["maxIoU"]])
            np.savez(
                output_dir / f"{dataset}.npz",
                column_Dic=result["column_Dic"],
                column_IoU=result["column_IoU"],
                meanDic=result["meanDic"],
                maxDic=result["maxDic"],
                meanIoU=result["meanIoU"],
                maxIoU=result["maxIoU"],
            )
    (output_dir / f"{model_name}_summary.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
