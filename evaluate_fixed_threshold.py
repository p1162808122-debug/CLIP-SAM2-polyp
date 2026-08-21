#!/usr/bin/env python3
"""Evaluate saved polyp predictions at one fixed threshold."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.metrics import EPS, _find_gt, _normalize, _read_mask


DATASETS = ["CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB"]


def evaluate_dataset(pred_dir: Path, gt_dir: Path, threshold: float):
    predictions = sorted(pred_dir.glob("*.png"))
    if not predictions:
        raise RuntimeError(f"No prediction png files found in {pred_dir}")
    dices = []
    ious = []
    for prediction_path in predictions:
        pred = _normalize(_read_mask(prediction_path) / 255.0)
        gt = _read_mask(_find_gt(gt_dir, prediction_path.name)) > 128
        if pred.shape != gt.shape:
            pred = np.asarray(
                Image.fromarray((pred * 255).astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR
                ),
                dtype=np.float64,
            ) / 255.0
            pred = _normalize(pred)
        binary = pred >= threshold
        gt_pixels = np.count_nonzero(gt)
        pred_pixels = np.count_nonzero(binary)
        intersection = np.count_nonzero(binary & gt)
        dices.append(2.0 * intersection / (gt_pixels + pred_pixels + EPS))
        ious.append(intersection / (gt_pixels + pred_pixels - intersection + EPS))
    return {
        "count": len(predictions),
        "threshold": threshold,
        "meanDice": float(np.mean(dices)),
        "meanIoU": float(np.mean(ious)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    source_root = args.project_root / "results" / args.run_name
    model_names = sorted(
        path.name for path in source_root.iterdir()
        if path.is_dir() and path.name in {"best_model2", "last_model"}
    )
    if model_names != ["best_model2", "last_model"]:
        raise RuntimeError(f"Expected best_model2 and last_model, found {model_names}")

    all_results = {}
    for model_name in model_names:
        model_results = {}
        model_output = args.output_root / args.run_name / model_name
        model_output.mkdir(parents=True, exist_ok=True)
        for dataset_name in DATASETS:
            result = evaluate_dataset(
                source_root / model_name / dataset_name,
                args.data_path / dataset_name / "masks",
                args.threshold,
            )
            model_results[dataset_name] = result
            print(
                f"{model_name}/{dataset_name} | count={result['count']} | "
                f"threshold={args.threshold:.1f} | meanDice={result['meanDice']:.4f} | "
                f"meanIoU={result['meanIoU']:.4f}"
            )
        all_results[model_name] = model_results
        (model_output / f"{model_name}_threshold_{args.threshold:.1f}.txt").write_text(
            "\n".join(
                f"Run={args.run_name} Model={model_name} Dataset={dataset} "
                f"count={result['count']} threshold={args.threshold:.1f} "
                f"meanDice={result['meanDice']:.6f} meanIoU={result['meanIoU']:.6f}"
                for dataset, result in model_results.items()
            ) + "\n",
            encoding="utf-8",
        )
        with (model_output / f"{model_name}_threshold_{args.threshold:.1f}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["run", "model", "dataset", "count", "threshold", "meanDice", "meanIoU"])
            for dataset, result in model_results.items():
                writer.writerow([
                    args.run_name, model_name, dataset, result["count"], result["threshold"],
                    result["meanDice"], result["meanIoU"],
                ])
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / args.run_name / "fixed_threshold_summary.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
