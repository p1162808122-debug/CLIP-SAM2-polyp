"""Measure a safe polyp5 batch size on one CUDA device.

The controller intentionally starts a fresh worker process for every candidate
so a failed CUDA allocation cannot poison the measurements for later sizes.
The worker prints exactly one JSON result on stdout; diagnostic tracebacks stay
on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


DEFAULT_CANDIDATES = (16, 8)


def select_batch_size(
    results: Iterable[dict[str, Any]],
    target_min_mib: int = 15360,
    target_max_mib: int = 20480,
) -> int:
    """Select the largest successful batch within the memory limit.

    A result inside the requested 15--20 GiB band is preferred.  If every
    safe result is below the band, the largest safe result is returned.  A
    result over the hard upper limit is never selected.
    """

    safe = []
    if target_min_mib > target_max_mib:
        raise ValueError("target_min_mib must not exceed target_max_mib")

    for result in results:
        if str(result.get("status", "")).lower() not in {"success", "ok"}:
            continue
        reserved = result.get("peak_reserved_mib")
        batch_size = result.get("batch_size")
        if reserved is None or batch_size is None:
            continue
        try:
            reserved_value = float(reserved)
            batch_value = int(batch_size)
        except (TypeError, ValueError):
            continue
        if reserved_value <= float(target_max_mib):
            normalized = dict(result)
            normalized["peak_reserved_mib"] = reserved_value
            normalized["batch_size"] = batch_value
            safe.append(normalized)
    if not safe:
        raise RuntimeError("No successful batch-size measurement is within the CUDA memory limit")

    preferred = [
        result for result in safe
        if float(result["peak_reserved_mib"]) >= float(target_min_mib)
    ]
    candidates = preferred or safe
    return int(max(candidates, key=lambda result: int(result["batch_size"]))["batch_size"])


def _mib(value: int) -> int:
    return int(round(float(value) / (1024 * 1024)))


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _worker_result(
    batch_size: int,
    sam_lora_rank: int,
    clip_lora_rank: int,
    text_lora_rank: int,
    status: str,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "batch_size": int(batch_size),
        "sam_lora_rank": int(sam_lora_rank),
        "clip_lora_rank": int(clip_lora_rank),
        "text_lora_rank": int(text_lora_rank),
        "peak_allocated_mib": 0,
        "peak_reserved_mib": 0,
        "status": status,
    }
    if error:
        result["error"] = error
    return result


def run_worker(args: argparse.Namespace) -> int:
    """Run one real forward/backward and emit one machine-readable result."""

    result: dict[str, Any]
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        project_root = Path(args.project_root).resolve()
        sys.path.insert(0, str(project_root))
        from models.polyp5 import Polyp5Model, resolve_polyp5_ranks
        from utils.dataloader import get_loader
        from utils.losses import compute_polyp5_loss

        device = torch.device("cuda")
        split_file = _resolve_path(project_root, args.split_file)
        loader = get_loader(
            args.train_path,
            str(split_file),
            352,
            int(args.batch_size),
            shuffle=False,
            use_augmentation=False,
            num_workers=0,
        )
        sam_images, clip_images, masks = next(iter(loader))
        if sam_images.shape[0] != int(args.batch_size):
            raise RuntimeError(
                f"dataset yielded only {sam_images.shape[0]} samples for batch {args.batch_size}; "
                "choose a smaller candidate or a larger training split"
            )

        sam_rank, clip_rank, text_rank = resolve_polyp5_ranks(
            legacy_lora_rank=args.rank,
            sam_lora_rank=args.sam_lora_rank,
            clip_lora_rank=args.clip_lora_rank,
            text_lora_rank=args.text_lora_rank,
        )
        model = Polyp5Model(
            sam_lora_rank=sam_rank,
            clip_lora_rank=clip_rank,
            text_lora_rank=text_rank,
            project_root=project_root,
            boundary_threshold=args.boundary_threshold,
            boundary_temperature=args.boundary_temperature,
        ).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        sam_images = sam_images.to(device, non_blocking=True)
        clip_images = clip_images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(sam_images, clip_images)
            loss = compute_polyp5_loss(
                output,
                masks,
                similarity_loss_weight=0.5,
                boundary_loss_weight=args.boundary_loss_weight,
                boundary_threshold=args.boundary_threshold,
                boundary_temperature=args.boundary_temperature,
                boundary_radius=args.boundary_radius,
            )
        loss.backward()
        torch.cuda.synchronize(device)
        result = {
            "batch_size": int(args.batch_size),
            "sam_lora_rank": sam_rank,
            "clip_lora_rank": clip_rank,
            "text_lora_rank": text_rank,
            "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved(device)),
            "status": "success",
        }
    except RuntimeError as exc:
        message = str(exc)
        status = "oom" if "out of memory" in message.lower() else "error"
        result = _worker_result(
            int(args.batch_size),
            int(args.sam_lora_rank or args.rank or 256),
            int(args.clip_lora_rank or args.rank or 128),
            int(args.text_lora_rank or args.rank or 64),
            status,
            error=message[-1000:],
        )
    except Exception as exc:  # pragma: no cover - exercised by remote environment failures
        result = _worker_result(
            int(args.batch_size),
            int(args.sam_lora_rank or args.rank or 256),
            int(args.clip_lora_rank or args.rank or 128),
            int(args.text_lora_rank or args.rank or 64),
            "error",
            error=f"{type(exc).__name__}: {exc}",
        )

    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


def _parse_worker_json(stdout: str, batch_size: int, sam_rank: int, clip_rank: int, text_rank: int) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("batch_size") == batch_size:
            return value
    return _worker_result(batch_size, sam_rank, clip_rank, text_rank, "error", error="worker did not emit a JSON result")


def run_controller(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    script = Path(__file__).resolve()
    output_path = Path(args.output).resolve()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    python_path = str(project_root)
    if env.get("PYTHONPATH"):
        python_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path

    sam_rank, clip_rank, text_rank = _resolved_args(args)
    attempts = []
    for batch_size in args.candidates:
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--project-root",
            str(project_root),
            "--train-path",
            args.train_path,
            "--split-file",
            args.split_file,
            "--batch-size",
            str(batch_size),
            "--sam-lora-rank",
            str(sam_rank),
            "--clip-lora-rank",
            str(clip_rank),
            "--text-lora-rank",
            str(text_rank),
            "--boundary-loss-weight",
            str(args.boundary_loss_weight),
            "--boundary-threshold",
            str(args.boundary_threshold),
            "--boundary-temperature",
            str(args.boundary_temperature),
            "--boundary-radius",
            str(args.boundary_radius),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
        result = _parse_worker_json(completed.stdout, int(batch_size), sam_rank, clip_rank, text_rank)
        if completed.returncode != 0 and result.get("status") == "success":
            result["status"] = "error"
        if completed.stderr.strip():
            result["stderr_tail"] = completed.stderr[-2000:]
        attempts.append(result)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))

    try:
        selected = select_batch_size(attempts, args.target_min_mib, args.target_max_mib)
    except RuntimeError:
        selected = None

    profile = {
        "variant": "polyp5",
        "sam_lora_rank": sam_rank,
        "clip_lora_rank": clip_rank,
        "text_lora_rank": text_rank,
        "boundary_loss_weight": float(args.boundary_loss_weight),
        "boundary_threshold": float(args.boundary_threshold),
        "boundary_temperature": float(args.boundary_temperature),
        "boundary_radius": int(args.boundary_radius),
        "candidates": [int(value) for value in args.candidates],
        "target_min_mib": int(args.target_min_mib),
        "target_max_mib": int(args.target_max_mib),
        "attempts": attempts,
        "selected_batch_size": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(json.dumps(profile, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(json.dumps({"selected_batch_size": selected, "profile": str(output_path)}, ensure_ascii=True))
    if selected is None:
        raise RuntimeError(f"No safe batch size was measured; see {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Calibrate a safe polyp5 CUDA batch size.")
    parser.add_argument("--mode", choices=("controller", "worker"), default="controller", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--train-path", default="data/TrainDataset")
    parser.add_argument("--split-file", default="utils/TrainDataset/train.txt")
    parser.add_argument("--rank", type=int, choices=(128,), default=None, help="legacy single-rank option")
    parser.add_argument("--sam-lora-rank", type=int, choices=(128, 256, 512), default=None)
    parser.add_argument("--clip-lora-rank", type=int, choices=(128, 256), default=None)
    parser.add_argument("--text-lora-rank", type=int, choices=(64, 128), default=None)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.25)
    parser.add_argument("--boundary-threshold", type=float, default=0.1)
    parser.add_argument("--boundary-temperature", type=float, default=0.05)
    parser.add_argument("--boundary-radius", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--candidates", type=int, nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--target-min-mib", type=int, default=15360)
    parser.add_argument("--target-max-mib", type=int, default=20480)
    parser.add_argument("--output", default=str(project_root / "batch_profile.json"))
    parser.add_argument("--cuda-visible-devices", default="0")
    args = parser.parse_args()
    args.worker = args.worker or args.mode == "worker"
    if args.worker and args.batch_size is None:
        parser.error("--worker requires --batch-size")
    if not args.worker and args.batch_size is not None:
        parser.error("--batch-size is only valid with --worker")
    return args


def _resolved_args(args: argparse.Namespace) -> tuple[int, int, int]:
    from models.polyp5 import resolve_polyp5_ranks

    return resolve_polyp5_ranks(
        legacy_lora_rank=args.rank,
        sam_lora_rank=args.sam_lora_rank,
        clip_lora_rank=args.clip_lora_rank,
        text_lora_rank=args.text_lora_rank,
    )


def main() -> int:
    args = parse_args()
    return run_worker(args) if args.worker else run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
