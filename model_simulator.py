from __future__ import annotations
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int] = [256, 128, 64],
                 n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)
        self._hidden_dims = hidden_dims

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        acts = {}
        h = x
        layer_idx = 0
        for module in self.backbone:
            h = module(h)
            if isinstance(module, nn.ReLU):
                acts[f"layer_{layer_idx}"] = h.detach()
                layer_idx += 1
        return acts

    def mc_dropout_forward(self, x: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
        self.train()
        with torch.no_grad():
            preds = torch.stack([F.softmax(self(x), dim=-1) for _ in range(n_samples)])
        self.eval()
        return preds

class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch or stride != 1 else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class CNN(nn.Module):
    def __init__(self, n_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU())
        self.layer1 = nn.Sequential(_ResBlock(64, 64),  _ResBlock(64, 64))
        self.layer2 = nn.Sequential(_ResBlock(64, 128, 2), _ResBlock(128, 128))
        self.layer3 = nn.Sequential(_ResBlock(128, 256, 2), _ResBlock(256, 256))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(dropout)
        self.head   = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.head(x)

    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        acts = {}
        x = self.stem(x);    acts["stem"]   = self.pool(x).flatten(1).detach()
        x = self.layer1(x);  acts["layer1"] = self.pool(x).flatten(1).detach()
        x = self.layer2(x);  acts["layer2"] = self.pool(x).flatten(1).detach()
        x = self.layer3(x);  acts["layer3"] = self.pool(x).flatten(1).detach()
        return acts

    def mc_dropout_forward(self, x: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
        self.train()
        with torch.no_grad():
            preds = torch.stack([F.softmax(self(x), dim=-1) for _ in range(n_samples)])
        self.eval()
        return preds

class ViTWrapper(nn.Module):

    def __init__(self, n_classes: int = 10, pretrained: bool = False):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("pip install timm")

        self.vit = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=False,
            num_classes=n_classes,
            img_size=32,
            patch_size=4,
        )
        self._attention_maps: List[torch.Tensor] = []
        for blk in self.vit.blocks:
            blk.attn.register_forward_hook(self._attn_hook)

    def _attn_hook(self, module, inputs, output):
        x = inputs[0]
        B, T, C = x.shape
        qkv = module.qkv(x).reshape(B, T, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        scale  = math.sqrt(q.shape[-1])
        attn_w = (q @ k.transpose(-2, -1)) / scale
        attn_w = attn_w.softmax(dim=-1)
        self._attention_maps.append(attn_w.detach().cpu())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._attention_maps.clear()
        return self.vit(x)

    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._attention_maps.clear()
        _ = self.vit(x)
        acts = {}
        for i, attn in enumerate(self._attention_maps):
            acts[f"attn_block_{i}"] = attn.mean(dim=(1, 2))
        return acts

    def get_attention_entropy(self, x: torch.Tensor) -> torch.Tensor:
        self._attention_maps.clear()
        with torch.no_grad():
            _ = self.forward(x)
        entropies = []
        for attn in self._attention_maps:
            cls_attn = attn[:, :, 0, 1:].mean(dim=1)
            eps = 1e-8
            H   = -(cls_attn * (cls_attn + eps).log()).sum(dim=-1)
            entropies.append(H)
        return torch.stack(entropies, dim=1)

    def mc_dropout_forward(self, x: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
        self.train()
        with torch.no_grad():
            preds = torch.stack([F.softmax(self.vit(x), dim=-1) for _ in range(n_samples)])
        self.eval()
        return preds

def build_model(arch: str, in_features: int = None, n_classes: int = 10,
                pretrained: bool = False, device: str = "cpu") -> nn.Module:
    arch = arch.lower()
    if arch == "mlp":
        assert in_features is not None
        model = MLP(in_features=in_features, n_classes=n_classes)
    elif arch == "cnn":
        model = CNN(n_classes=n_classes)
    elif arch == "vit":
        model = ViTWrapper(n_classes=n_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model.to(device)