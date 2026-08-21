#!/usr/bin/env python3
"""Render standalone positive-text minus negative-text response maps."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    outputs = []
    for npz_path in sorted(args.input_root.glob("*/*_maps.npz")):
        dataset = npz_path.parent.name
        image_stem = npz_path.stem.removesuffix("_maps")
        maps = np.load(npz_path)
        response = np.nan_to_num(
            maps["semantic_margin"].astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        limit = float(np.percentile(np.abs(response), 99.0))
        if limit <= 1e-8:
            limit = 1.0
        output_path = args.output_root / dataset / f"{image_stem}_semantic_margin.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(6.4, 5.8), dpi=180)
        image = axis.imshow(response, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"{dataset}/{image_stem}\npositive text − negative text", fontsize=11)
        axis.axis("off")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("similarity response difference", fontsize=9)
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output_path)
        print(f"[Margin] {dataset}/{image_stem} -> {output_path}")

    if len(outputs) != 5:
        raise RuntimeError(f"expected 5 response maps, found {len(outputs)}")
    print(f"[Done] {len(outputs)} standalone response maps")


if __name__ == "__main__":
    main()
