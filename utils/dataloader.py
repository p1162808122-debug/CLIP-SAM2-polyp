"""Joint image/mask loading for the two static SAM2 experiments."""

from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


SAM_MEAN = (0.485, 0.456, 0.406)
SAM_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_map = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS}
    mask_map = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in VALID_EXTS}
    names = sorted(set(image_map) & set(mask_map))
    if not names:
        raise RuntimeError(f"No matched image/mask pairs found under {root}")
    return [(image_map[name], mask_map[name]) for name in names]


def _pairs_from_split(split_file: Path, dataset_root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for raw_line in split_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid split line in {split_file}: {line}")
        pairs.append(((dataset_root / parts[0]).resolve(), (dataset_root / parts[1]).resolve()))
    if not pairs:
        raise RuntimeError(f"Split file is empty: {split_file}")
    return pairs


def _joint_augmentation(image_size: int):
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.08, 0.08),
            rotate=(-20, 20),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.6,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
    ])


def _read_pair(image_path: Path, mask_path: Path):
    with Image.open(image_path) as image_handle:
        image = np.asarray(image_handle.convert("RGB"))
    with Image.open(mask_path) as mask_handle:
        mask = np.asarray(mask_handle.convert("L"))
    return image, mask


def _to_inputs(image: np.ndarray, mask: np.ndarray, image_size: int):
    image_pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    raw = TF.to_tensor(image_pil)
    sam = TF.resize(raw, [image_size, image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    sam = TF.normalize(sam, SAM_MEAN, SAM_STD)
    # polyp4 uses the same 352x352 geometry for SAM2 and BiomedCLIP.  The
    # vendored timm tower is constructed with an interpolated 22x22 grid.
    clip = TF.resize(raw, [image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    clip = TF.normalize(clip, CLIP_MEAN, CLIP_STD)

    mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
    mask_tensor = TF.resize(mask_tensor, [image_size, image_size], interpolation=InterpolationMode.NEAREST)
    mask_tensor = (mask_tensor > 0.5).float()
    return sam, clip, mask_tensor


class PolypDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        image_size: int = 352,
        split_file: Optional[str] = None,
        use_augmentation: bool = False,
    ):
        self.dataset_root = Path(dataset_root).resolve()
        self.image_size = int(image_size)
        self.pairs = (
            _pairs_from_split(Path(split_file).resolve(), self.dataset_root)
            if split_file is not None
            else _collect_pairs(self.dataset_root)
        )
        self.augmentation = _joint_augmentation(self.image_size) if use_augmentation else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, mask = _read_pair(*self.pairs[index])
        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        sam, clip, target = _to_inputs(image, mask, self.image_size)
        return sam, clip, target


class TestDataset(Dataset):
    def __init__(self, dataset_root: str, image_size: int = 352):
        self.dataset_root = Path(dataset_root).resolve()
        self.image_size = int(image_size)
        self.pairs = _collect_pairs(self.dataset_root)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, mask = _read_pair(*self.pairs[index])
        sam, clip, _ = _to_inputs(image, mask, self.image_size)
        target = torch.from_numpy((mask > 127).astype(np.float32))
        return sam, clip, target, self.pairs[index][0].stem + ".png"


def get_loader(
    dataset_root: str,
    split_file: str,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    use_augmentation: bool,
    num_workers: int = 0,
):
    dataset = PolypDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        split_file=split_file,
        use_augmentation=use_augmentation,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
