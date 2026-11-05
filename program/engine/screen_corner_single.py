"""Single-graph coarse-to-local corner model and self-contained TorchScript loader.

The exported graph owns resizing, ImageNet normalization, global features,
coarse decoding, image/feature crop sampling, and local refinement. It reads no
other checkpoints, configuration files, torchvision installation, or network.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GlobalPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, 32, 1) for c in [24, 48, 576]])
        self.fuse = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.corners = nn.Conv2d(32, 4, 1)
        self.boundary = nn.Conv2d(32, 2, 1)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        fused = self.laterals[0](features[0])
        fused = F.interpolate(fused, size=[64, 64], mode='bilinear', align_corners=False)
        for i, layer in enumerate(self.laterals):
            if i > 0:
                fused = fused + F.interpolate(layer(features[i]), size=[64, 64], mode='bilinear', align_corners=False)
        fused = self.fuse(fused)
        return self.corners(fused), self.boundary(fused), fused


def decode_points(logits: torch.Tensor) -> torch.Tensor:
    h, w = logits.shape[-2:]
    weights = torch.softmax(logits.flatten(2) * 12.0, dim=-1)
    xs = torch.linspace(0., 1., w, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(0., 1., h, device=logits.device, dtype=logits.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([(weights * gx.flatten()).sum(-1), (weights * gy.flatten()).sum(-1)], -1)


def patch_boxes(points: torch.Tensor) -> torch.Tensor:
    # Coordinate normalization is per image axis; preserve center near borders.
    span = (points.amax(1, keepdim=True) - points.amin(1, keepdim=True)).clamp(min=.05)
    radius = span * .15
    return torch.cat([points - radius, points + radius], -1)


def sample_corner_patches(image: torch.Tensor, boxes: torch.Tensor, size: int) -> torch.Tensor:
    axis = torch.linspace(0., 1., size, device=image.device, dtype=image.dtype)
    gy, gx = torch.meshgrid(axis, axis, indexing='ij')
    grid = torch.stack([gx, gy], -1)[None, None]
    coords = boxes[:, :, None, None, :2] + grid * (boxes[:, :, None, None, 2:] - boxes[:, :, None, None, :2])
    patches = []
    # Avoid replicating a full-resolution photograph four times in memory.
    for corner in range(4):
        patches.append(F.grid_sample(image, coords[:, corner] * 2. - 1., mode='bilinear',
                                     padding_mode='border', align_corners=True))
    return torch.stack(patches, 1).flatten(0, 1)


class LocalPointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Conv2d(32, 8, 1)
        self.tower = nn.Sequential(nn.Conv2d(17, 24, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(),
                                   nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.ReLU(),
                                   nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.heatmap = nn.Conv2d(64, 1, 1)
        self.offset = nn.Conv2d(64, 2, 1)
        nn.init.zeros_(self.heatmap.weight)
        nn.init.zeros_(self.heatmap.bias)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        axis = torch.linspace(0., 1., 64)
        gy, gx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('position', torch.stack([gx, gy], 0)[None])
        grid16 = torch.linspace(0., 1., 16)
        py, px = torch.meshgrid(grid16, grid16, indexing='ij')
        prior = .5 * torch.exp(-((px-.5)**2 + (py-.5)**2) / (2*(1.5/15)**2))
        self.register_buffer('center_prior', prior[None, None])
        self.register_buffer('corner_identity', torch.eye(4))

    def forward(self, rgb: torch.Tensor, context: torch.Tensor):
        n = rgb.shape[0]
        context = F.interpolate(self.context(context), size=[64, 64], mode='bilinear', align_corners=False)
        position = self.position.expand(n, -1, -1, -1)
        identity = self.corner_identity.repeat(n // 4, 1)[:, :, None, None].expand(-1, -1, 64, 64)
        feature = self.tower(torch.cat([rgb, context, position, identity], 1))
        logits = self.heatmap(feature) + self.center_prior
        offsets = torch.tanh(self.offset(feature))
        weights = torch.softmax(logits.flatten(2) * 12., -1)
        point = decode_points(logits)[:, 0] + (offsets.flatten(2) * weights).sum(-1) / 15.
        return point.clamp(0., 1.), logits


class ScreenCornerSingle(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.global_head = GlobalPyramid()
        self.local_head = LocalPointHead()
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        small = F.interpolate(image, size=[256, 256], mode='bilinear', align_corners=False)
        logits, _, feature = self.global_head(self.encoder((small-self.mean)/self.std))
        coarse = decode_points(logits)
        boxes = patch_boxes(coarse)
        patches = sample_corner_patches(image, boxes, 64)
        context = sample_corner_patches(feature, boxes, 16)
        relative, _ = self.local_head(patches, context)
        relative = relative.view(-1, 4, 2)
        refined = (boxes[..., :2] + relative * (boxes[..., 2:] - boxes[..., :2])).clamp(0., 1.)
        return {'coarse_quad': coarse, 'final_quad': refined}


class SingleModelPredictor:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.model_id = "screen_corner_single_" + hashlib.sha256(self.path.read_bytes()).hexdigest()[:12]
        self.model = torch.jit.load(str(path), map_location='cpu').eval()

    def predict(self, image: np.ndarray) -> dict[str, np.ndarray]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected a BGR photograph')
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255.
        with torch.inference_mode():
            output = self.model(tensor)
        scale = np.array([image.shape[1], image.shape[0]], np.float32)
        return {k: v[0].cpu().numpy() * scale for k, v in output.items()}

    def __call__(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        return self.predict(image)['final_quad']
