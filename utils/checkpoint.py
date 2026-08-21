"""Checkpoint management matching the RePraNet2 run layout."""

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Union

import torch


class CheckpointManager:
    def __init__(self, project_root: Path, variant: str, rank: Union[int, str], total_epochs: int, config: Dict[str, Any], patience: int = 10):
        self.project_root = Path(project_root)
        self.variant = variant
        self.rank = rank
        self.total_epochs = int(total_epochs)
        self.config = config
        self.patience = max(int(patience), 1)
        self.checkpoint_root = self.project_root / "checkpoint"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.run_id = self._next_run_id()
        rank_label = f"rank{rank}" if isinstance(rank, int) else str(rank)
        self.run_name = f"run{self.run_id}_{variant}_{rank_label}_{total_epochs}epoch"
        self.run_dir = self.checkpoint_root / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.run_dir / "best_model.pth"
        self.last_model_path = self.run_dir / "last_model.pth"
        self.log_path = self.run_dir / "train.log"
        self.summary_path = self.run_dir / "checkpoint_summary.json"
        self.best_metric = float("-inf")
        self.best_epoch = -1
        self.snapshot_counter = 0
        self.epochs_without_improvement = 0
        self._write_log("#" * 80)
        self._write_log(f"start_time: {datetime.now()}")
        self._write_log(f"run_dir: {self.run_dir}")
        self._write_log(f"config: {json.dumps(config, ensure_ascii=True, default=str)}")

    def _next_run_id(self) -> int:
        pattern = re.compile(r"^run(\d+)_")
        ids = []
        for path in self.checkpoint_root.iterdir():
            match = pattern.match(path.name)
            if path.is_dir() and match:
                ids.append(int(match.group(1)))
        return max(ids, default=0) + 1

    def _write_log(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    @staticmethod
    def _save(model, path: Path) -> None:
        # Raw full state_dict: frozen backbones and non-used SAM2 compatibility modules are included.
        torch.save(model.state_dict(), path)

    def save_last(self, model) -> None:
        self._save(model, self.last_model_path)

    def maybe_save_best(self, model, metric: float, epoch: int) -> bool:
        if metric > self.best_metric:
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
            self._save(model, self.best_model_path)
            if epoch >= 10:
                self.snapshot_counter += 1
                self._save(model, self.run_dir / f"best_model{self.snapshot_counter}.pth")
            return True
        self.epochs_without_improvement += 1
        return False

    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        valid_dice: Optional[float],
        is_best: bool,
        loss_components: Optional[Dict[str, float]] = None,
    ) -> None:
        parts = [f"{datetime.now()}", f"epoch={epoch}", f"train_loss={train_loss:.6f}"]
        if loss_components:
            for name in ("mask_loss", "similarity_loss", "boundary_loss", "total_loss"):
                if name in loss_components:
                    parts.append(f"{name}={float(loss_components[name]):.6f}")
        if valid_dice is not None:
            parts.append(f"valid_dice={valid_dice:.6f}")
        parts.append(f"is_best={is_best}")
        self._write_log(" | ".join(parts))

    def finalize(self, model=None) -> None:
        if model is not None:
            self.save_last(model)
        if self.best_epoch < 0 and self.last_model_path.is_file():
            state = torch.load(self.last_model_path, map_location="cpu", weights_only=True)
            torch.save(state, self.best_model_path)
        payload = {
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "best_model": str(self.best_model_path),
            "last_model": str(self.last_model_path),
            "best_metric": None if self.best_epoch < 0 else self.best_metric,
            "best_epoch": None if self.best_epoch < 0 else self.best_epoch,
            "end_time": str(datetime.now()),
            "config": self.config,
        }
        self.summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def load_state_dict(path: str):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    return payload


def discover_weights(run_dir: Path):
    numbered = []
    for path in run_dir.glob("best_model*.pth"):
        match = re.fullmatch(r"best_model(\d+)", path.stem)
        if match:
            numbered.append((int(match.group(1)), path))
    weights = []
    last = run_dir / "last_model.pth"
    best = run_dir / "best_model.pth"
    if last.is_file():
        weights.append(("last_model", last))
    if numbered:
        weights.append((max(numbered, key=lambda item: item[0])[1].stem, max(numbered, key=lambda item: item[0])[1]))
    elif best.is_file():
        weights.append(("best_model", best))
    return weights
