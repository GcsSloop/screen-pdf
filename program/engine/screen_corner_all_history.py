"""Region -> region-conditioned coarse points -> local refinement, one graph.

This architecture is trained independently of ScreenPDF task checkpoints.
The full-region mask includes occlusion and blank slide margins. A learned
coarse head consumes that mask and global features; it does not require a hard
thresholded mask to have four already-correct sides.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from screen_corner_single import GlobalPyramid, LocalPointHead, patch_boxes, sample_corner_patches


class PortableHardSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 3.).clamp(0., 6.) / 6.


class PortableHardSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * ((x + 3.).clamp(0., 6.) / 6.)


class RegionCoarseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pyramid = GlobalPyramid()
        # Region probability gates image features while retaining unmasked
        # global context for occluded corners and dark/partial screens.
        self.coarse = nn.Sequential(nn.Conv2d(36, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 4, 1))
        axis = torch.linspace(0., 1., 64)
        gy, gx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('coordinates', torch.stack([gx, gy])[None])

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        _, boundary, feature = self.pyramid(features)
        probability = boundary.sigmoid()
        # A soft region prior is differentiable during the entire global stage.
        conditioned = torch.cat([feature * (.25 + .75 * probability[:, :1]), probability,
                                 self.coordinates.expand(feature.shape[0], -1, -1, -1)], 1)
        logits = self.coarse(conditioned)
        weights = torch.softmax(logits.flatten(2), -1)
        coords = self.coordinates.flatten(2)
        quad = torch.stack([(weights * coords[:, :1]).sum(-1), (weights * coords[:, 1:]).sum(-1)], -1)
        return quad, logits, boundary, feature


class AllHistoryScreenCorner(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.global_head = RegionCoarseHead()
        self.local_head = LocalPointHead()
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    @torch.jit.export
    def global_prediction(self, image: torch.Tensor):
        small = F.interpolate(image, [256, 256], mode='bilinear', align_corners=False)
        return self.global_head(self.encoder((small - self.mean) / self.std))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        coarse, _, _, feature = self.global_prediction(image)
        boxes = patch_boxes(coarse)
        rgb = sample_corner_patches(image, boxes, 64)
        context = sample_corner_patches(feature, boxes, 16)
        relative, _ = self.local_head(rgb, context)
        final = (boxes[..., :2] + relative.view(-1, 4, 2) * (boxes[..., 2:] - boxes[..., :2])).clamp(0., 1.)
        return {'coarse_quad': coarse, 'final_quad': final}


def make_generic_encoder():
    """ImageNet weights only; neither current runtime nor task teacher loaded."""
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    def portable(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Hardsigmoid):
                setattr(module, name, PortableHardSigmoid())
            elif isinstance(child, nn.Hardswish):
                setattr(module, name, PortableHardSwish())
            else:
                portable(child)
    portable(backbone)
    feature = create_feature_extractor(backbone.features, return_nodes={'3': 'low', '8': 'mid', '12': 'high'})
    from training.train_screen_corner_single import TupleEncoder
    return TupleEncoder(feature)
