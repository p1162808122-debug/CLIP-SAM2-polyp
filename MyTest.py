import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.polyp5 import Polyp5Model, polyp5_rank_label, resolve_polyp5_ranks
from utils.checkpoint import discover_weights, load_state_dict
from utils.dataloader import TestDataset


DATASETS = ["CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB"]


def project_variant() -> str:
    name = Path(__file__).resolve().parent.name
    if not name.endswith("polyp7"):
        raise RuntimeError(f"This entrypoint is reserved for SAM2-polyp7, got {name}")
    return "polyp7"


def discover_latest_run(project_root: Path):
    checkpoint_root = project_root / "checkpoint"
    if not checkpoint_root.is_dir():
        return None
    candidates = []
    for path in checkpoint_root.iterdir():
        match = re.match(r"^run(\d+)_", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _checkpoint_config(run_dir: Path):
    summary = run_dir / "checkpoint_summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        return payload.get("config", {})
    return {}


def infer_ranks(run_dir: Path, legacy_requested=None, sam_requested=None, clip_requested=None, text_requested=None):
    config = _checkpoint_config(run_dir)
    legacy = legacy_requested if legacy_requested is not None else config.get("lora_rank")
    sam = sam_requested if sam_requested is not None else config.get("sam_lora_rank")
    clip = clip_requested if clip_requested is not None else config.get("clip_lora_rank")
    text = text_requested if text_requested is not None else config.get("text_lora_rank")
    if sam is None and clip is None and text is None and legacy is None:
        match = re.search(r"sam(128|256|512)_clip(128|256)_text(64|128)", run_dir.name)
        if match:
            sam, clip, text = (int(value) for value in match.groups())
    return resolve_polyp5_ranks(
        legacy_lora_rank=None if legacy is None else int(legacy),
        sam_lora_rank=None if sam is None else int(sam),
        clip_lora_rank=None if clip is None else int(clip),
        text_lora_rank=None if text is None else int(text),
    )


def infer_rank(run_dir: Path, requested):
    """Backward-compatible helper returning the SAM rank."""
    return infer_ranks(run_dir, legacy_requested=requested)[0]


def infer_text_lora(run_dir: Path, requested):
    # Text LoRA is a fixed part of the polyp7 architecture.
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Test SAM2-polyp7 gated-fusion checkpoints with CFBR refinement.")
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--lora-rank", type=int, choices=Polyp5Model.SUPPORTED_LORA_RANKS, default=None)
    parser.add_argument("--sam-lora-rank", type=int, choices=Polyp5Model.SUPPORTED_SAM_LORA_RANKS, default=None)
    parser.add_argument("--clip-lora-rank", type=int, choices=Polyp5Model.SUPPORTED_CLIP_LORA_RANKS, default=None)
    parser.add_argument("--text-lora-rank", type=int, choices=Polyp5Model.SUPPORTED_TEXT_LORA_RANKS, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--boundary-threshold", type=float, default=0.1)
    parser.add_argument("--boundary-temperature", type=float, default=0.05)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--pth-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default="data/TestDataset")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="optional per-dataset smoke-test limit")
    return parser.parse_args()


def _save_prediction(path: Path, prediction: torch.Tensor):
    array = prediction.detach().cpu().numpy().squeeze().astype(np.float32)
    array = (array - array.min()) / (array.max() - array.min() + 1e-8)
    Image.fromarray((array * 255.0).astype(np.uint8), mode="L").save(path)


def test_one_model(
    model_path: Path,
    model_stem: str,
    run_name: str,
    args,
    project_root: Path,
    sam_rank: int,
    clip_rank: int,
    text_rank: int,
    device: torch.device,
):
    variant = project_variant()
    model = Polyp5Model(
        lora_alpha=args.lora_alpha,
        project_root=project_root,
        sam_lora_rank=sam_rank,
        clip_lora_rank=clip_rank,
        text_lora_rank=text_rank,
        boundary_threshold=args.boundary_threshold,
        boundary_temperature=args.boundary_temperature,
    )
    model.load_state_dict(load_state_dict(str(model_path)), strict=True)
    model.to(device).eval()
    result_root = project_root / "results" / run_name / model_stem

    for dataset_name in args.datasets:
        dataset_root = Path(args.test_path) / dataset_name
        if not dataset_root.is_dir():
            print(f"[Skip] missing dataset: {dataset_root}")
            continue
        output_dir = result_root / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = TestDataset(str(dataset_root), image_size=args.testsize)
        sample_count = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
        for index in range(sample_count):
            sam_image, clip_image, gt, name = dataset[index]
            sam_image = sam_image.unsqueeze(0).to(device)
            clip_image = clip_image.unsqueeze(0).to(device)
            with torch.inference_mode():
                if args.no_amp or device.type != "cuda":
                    output = model(sam_image, clip_image)
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(sam_image, clip_image)
                logits = output["high_res_masks"]
                logits = F.interpolate(logits.float(), size=tuple(gt.shape), mode="bilinear", align_corners=False)
                probability = torch.sigmoid(logits)
            _save_prediction(output_dir / name, probability)
        print(f"[Test] {model_stem} {dataset_name}: {sample_count} predictions -> {output_dir}")


def main():
    args = parse_args()
    if args.testsize != 352:
        raise ValueError("This static SAM2 configuration is fixed at testsize=352")
    project_root = Path(__file__).resolve().parent
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    if args.pth_path:
        model_path = Path(args.pth_path).resolve()
        run_dir = model_path.parent
        weights = [(model_path.stem, model_path)]
    else:
        run_dir = Path(args.run_dir).resolve() if args.run_dir else discover_latest_run(project_root)
        if run_dir is None or not run_dir.is_dir():
            raise FileNotFoundError("No checkpoint run found; provide --run-dir or --pth-path")
        weights = discover_weights(run_dir)
    if not weights:
        raise FileNotFoundError(f"No checkpoint weights found in {run_dir}")

    sam_rank, clip_rank, text_rank = infer_ranks(
        run_dir,
        legacy_requested=args.lora_rank,
        sam_requested=args.sam_lora_rank,
        clip_requested=args.clip_lora_rank,
        text_requested=args.text_lora_rank,
    )
    print(
        f"[Test] variant={project_variant()} {polyp5_rank_label(sam_rank, clip_rank, text_rank)} "
        f"text_lora=True device={device}"
    )
    for stem, path in weights:
        test_one_model(path, stem, run_dir.name, args, project_root, sam_rank, clip_rank, text_rank, device)


if __name__ == "__main__":
    main()
