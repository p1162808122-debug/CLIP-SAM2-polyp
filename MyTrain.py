import argparse
from contextlib import nullcontext
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from models.polyp5 import Polyp5Model, polyp5_rank_label, resolve_polyp5_ranks
from utils.checkpoint import CheckpointManager
from utils.dataloader import get_loader
from utils.losses import compute_polyp5_loss_components, dice_score


def project_variant() -> str:
    name = Path(__file__).resolve().parent.name
    if not name.endswith("polyp7"):
        raise RuntimeError(f"This entrypoint is reserved for SAM2-polyp7, got {name}")
    return "polyp7"


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM2-polyp7 with three-scale gated fusion, CFBR refinement, and boundary similarity.")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--trainsize", type=int, default=352)
    # Legacy single-rank option remains available for the completed rank128
    # run; new experiments use three independent rank arguments.
    parser.add_argument("--lora-rank", type=int, choices=Polyp5Model.SUPPORTED_LORA_RANKS, default=None)
    parser.add_argument("--sam-lora-rank", type=int, choices=Polyp5Model.SUPPORTED_SAM_LORA_RANKS, default=None)
    parser.add_argument("--clip-lora-rank", type=int, choices=Polyp5Model.SUPPORTED_CLIP_LORA_RANKS, default=None)
    parser.add_argument("--text-lora-rank", type=int, choices=Polyp5Model.SUPPORTED_TEXT_LORA_RANKS, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--similarity-loss-weight", type=float, default=0.5)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.25)
    parser.add_argument("--boundary-threshold", type=float, default=0.1)
    parser.add_argument("--boundary-temperature", type=float, default=0.05)
    parser.add_argument("--boundary-radius", type=int, default=8)
    parser.add_argument("--train-path", type=str, default="data/TrainDataset")
    parser.add_argument("--split-dir", type=str, default="utils/TrainDataset")
    parser.add_argument("--valid-interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.set_defaults(use_augmentation=True, amp=True)
    parser.add_argument("--use-augmentation", dest="use_augmentation", action="store_true")
    parser.add_argument("--no-augmentation", dest="use_augmentation", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_totals = {name: 0.0 for name in ("mask_loss", "similarity_loss", "boundary_loss", "total_loss")}
    total_samples = 0
    max_steps = args.max_train_samples

    for step, (sam_images, clip_images, masks) in enumerate(loader, start=1):
        if max_steps is not None and step > max_steps:
            break
        sam_images = sam_images.to(device, non_blocking=True)
        clip_images = clip_images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with _autocast(device, args.amp):
            output = model(sam_images, clip_images)
            components = compute_polyp5_loss_components(
                output,
                masks,
                similarity_loss_weight=args.similarity_loss_weight,
                boundary_loss_weight=args.boundary_loss_weight,
                boundary_threshold=args.boundary_threshold,
                boundary_temperature=args.boundary_temperature,
                boundary_radius=args.boundary_radius,
            )
            loss = components["total_loss"]
            scaled_loss = loss / max(args.grad_accum_steps, 1)
        scaler.scale(scaled_loss).backward()

        is_last = step == len(loader) or (max_steps is not None and step >= max_steps)
        if step % max(args.grad_accum_steps, 1) == 0 or is_last:
            scaler.unscale_(optimizer)
            clip_grad_norm_(list(model.trainable_parameters()), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = sam_images.shape[0]
        for name in loss_totals:
            loss_totals[name] += float(components[name].detach()) * batch_size
        total_samples += batch_size

    return {name: value / max(total_samples, 1) for name, value in loss_totals.items()}


@torch.inference_mode()
def validate(model, loader, device, max_samples=None):
    model.eval()
    dice_total = 0.0
    samples = 0
    for step, (sam_images, clip_images, masks) in enumerate(loader, start=1):
        if max_samples is not None and step > max_samples:
            break
        output = model(sam_images.to(device), clip_images.to(device))
        probability = torch.sigmoid(output["high_res_masks"])
        batch_size = sam_images.shape[0]
        dice_total += dice_score(probability.cpu(), masks) * batch_size
        samples += batch_size
    return dice_total / max(samples, 1)


def main():
    args = parse_args()
    if args.trainsize != 352:
        raise ValueError("This static SAM2 configuration is fixed at trainsize=352")
    seed_everything()
    project_root = Path(__file__).resolve().parent
    variant = project_variant()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    sam_rank, clip_rank, text_rank = resolve_polyp5_ranks(
        legacy_lora_rank=args.lora_rank,
        sam_lora_rank=args.sam_lora_rank,
        clip_lora_rank=args.clip_lora_rank,
        text_lora_rank=args.text_lora_rank,
    )
    rank_label = polyp5_rank_label(sam_rank, clip_rank, text_rank)
    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = project_root / split_dir
    train_loader = get_loader(
        args.train_path,
        str(split_dir / "train.txt"),
        args.trainsize,
        args.batchsize,
        shuffle=True,
        use_augmentation=args.use_augmentation,
        num_workers=args.num_workers,
    )
    val_loader = get_loader(
        args.train_path,
        str(split_dir / "val.txt"),
        args.trainsize,
        args.batchsize,
        shuffle=False,
        use_augmentation=False,
        num_workers=args.num_workers,
    )

    model = Polyp5Model(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        project_root=project_root,
        sam_lora_rank=sam_rank,
        clip_lora_rank=clip_rank,
        text_lora_rank=text_rank,
        boundary_threshold=args.boundary_threshold,
        boundary_temperature=args.boundary_temperature,
    ).to(device)
    trainable_params = list(model.trainable_parameters())
    if not trainable_params:
        raise RuntimeError("No trainable parameters were found")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epoch - 1, 1)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    config = vars(args).copy()
    config.update({
        "variant": variant,
        "device": str(device),
        "sam_lora_rank": sam_rank,
        "clip_lora_rank": clip_rank,
        "text_lora_rank": text_rank,
        "rank_label": rank_label,
        "boundary_threshold": args.boundary_threshold,
        "boundary_temperature": args.boundary_temperature,
        "boundary_radius": args.boundary_radius,
        "parameter_summary": model.trainable_parameter_summary(),
    })
    manager = CheckpointManager(
        project_root=project_root,
        variant=variant,
        rank=rank_label,
        total_epochs=args.epoch,
        config=config,
        patience=args.patience,
    )

    print(
        f"[Model] variant={variant} sam_rank={sam_rank} clip_rank={clip_rank} "
        f"text_rank={text_rank} text_lora=True "
        f"similarity_lambda={args.similarity_loss_weight} "
        f"boundary_lambda={args.boundary_loss_weight} "
        f"boundary_threshold={args.boundary_threshold} "
        f"boundary_temperature={args.boundary_temperature} "
        f"boundary_radius={args.boundary_radius} device={device}"
    )
    print(f"[Model] parameters={model.trainable_parameter_summary()}")
    for epoch in range(1, args.epoch + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        valid_dice = None
        is_best = False
        if epoch % max(args.valid_interval, 1) == 0:
            valid_dice = validate(model, val_loader, device, args.max_val_samples)
            is_best = manager.maybe_save_best(model, valid_dice, epoch)
        manager.log_epoch(epoch, train_losses["total_loss"], valid_dice, is_best, train_losses)
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.epoch:03d} | "
            f"mask={train_losses['mask_loss']:.6f} "
            f"similarity={train_losses['similarity_loss']:.6f} "
            f"boundary={train_losses['boundary_loss']:.6f} "
            f"total={train_losses['total_loss']:.6f} "
            f"| dice={valid_dice if valid_dice is not None else float('nan'):.6f} "
            f"| best={is_best}"
        )
        if valid_dice is not None and manager.should_stop():
            print(f"[EarlyStop] no validation Dice improvement for {manager.patience} checks")
            break
    manager.finalize(model)
    print(f"[Done] checkpoint directory: {manager.run_dir}")


if __name__ == "__main__":
    main()
