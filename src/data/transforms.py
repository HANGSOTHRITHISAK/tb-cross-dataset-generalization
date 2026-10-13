"""Fixed ImageNet normalization; no statistics are fitted to project data."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_rgb(image: Image.Image) -> Image.Image:
    # This release is audited 8-bit. Reject high-bit-depth/float inputs rather
    # than silently clipping their intensities through PIL's RGB conversion.
    if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
        raise ValueError(f"Unsupported image mode: {image.mode}; expected 8-bit image")
    return image.convert("RGB")


def build_transform(*, training: bool = False, image_size: int = 224):
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    steps = [
        transforms.Lambda(to_rgb),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ]
    if training:
        # Small translations/rotations; preserve laterality and the full image.
        steps.append(
            transforms.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        )
    steps.extend(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return transforms.Compose(steps)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
