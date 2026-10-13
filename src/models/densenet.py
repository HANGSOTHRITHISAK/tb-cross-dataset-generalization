from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.data.manifest import file_sha256


def build_densenet121(
    *,
    device: str | torch.device = "cpu",
    pretrained: bool = True,
    allow_download: bool = False,
) -> nn.Module:
    """Three-class baseline. Never silently substitute random initialization."""
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    if weights is not None:
        filename = Path(urlparse(weights.url).path).name
        cache = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not cache.is_file() and not allow_download:
            raise FileNotFoundError(
                "ImageNet DenseNet-121 weights are not cached; explicitly allow "
                "download, or use pretrained=False only for engineering fixtures"
            )
        if cache.is_file() and not file_sha256(cache).startswith("a639ec97"):
            raise ValueError(
                "Cached DenseNet-121 weights failed the published hash check"
            )
    try:
        model = densenet121(weights=weights)
    except Exception as error:
        raise RuntimeError(
            "DenseNet-121 initialization failed; check torchvision compatibility and weight availability"
        ) from error
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model.to(device)
