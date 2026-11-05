"""Full-region geometry prior for occluded coarse screen corners.

The mask describes the complete manually annotated PPT quadrilateral, including
occluded portions. Four fitted sides retain perspective; an axis-aligned box is
never substituted for the quadrilateral. All operations export in one graph.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from screen_corner_single import (GlobalPyramid, LocalPointHead, patch_boxes,
                                  sample_corner_patches)


def weighted_line(independent: torch.Tensor, dependent: torch.Tensor,
                  weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    total = weights.sum(-1).clamp(min=1e-6)
    mx = (independent * weights).sum(-1) / total
    my = (dependent * weights).sum(-1) / total
    dx = independent - mx[:, None]
    dy = dependent - my[:, None]
    variance = (weights * dx.square()).sum(-1) / total
    covariance = (weights * dx * dy).sum(-1) / total
    slope = covariance / variance.clamp(min=1e-5)
    return slope, my - slope * mx


def intersect_sides(x_slope: torch.Tensor, x_intercept: torch.Tensor,
                    y_slope: torch.Tensor, y_intercept: torch.Tensor) -> torch.Tensor:
    denominator = 1. - x_slope * y_slope
    # A failed/empty region must still emit finite points and remain in scoring.
    denominator = torch.where(denominator >= 0., denominator.clamp(min=.05),
                              denominator.clamp(max=-.05))
    x = (x_slope * y_intercept + x_intercept) / denominator
    y = y_slope * x + y_intercept
    return torch.stack([x, y], -1).clamp(0., 1.)


def region_to_quad(probability: torch.Tensor) -> torch.Tensor:
    """Fit left/right/top/bottom outer transitions of a complete-region mask.

    Cumulative maxima ignore interior holes, rather than fitting audience edges
    inside the PPT. They cannot reconstruct a wholly missing side by themselves;
    supervised complete-region prediction and the learned correction handle that.
    """
    support = (probability[:, 0] - .5).clamp(min=0.)
    h, w = support.shape[-2:]
    xs = torch.linspace(0., 1., w, dtype=support.dtype, device=support.device)
    ys = torch.linspace(0., 1., h, dtype=support.dtype, device=support.device)
    left_envelope = torch.cummax(support, dim=2)[0]
    right_envelope = torch.flip(torch.cummax(torch.flip(support, [2]), dim=2)[0], [2])
    top_envelope = torch.cummax(support, dim=1)[0]
    bottom_envelope = torch.flip(torch.cummax(torch.flip(support, [1]), dim=1)[0], [1])
    left = (left_envelope - F.pad(left_envelope[:, :, :-1], [1, 0])).clamp(min=0.)
    right = (right_envelope - F.pad(right_envelope[:, :, 1:], [0, 1])).clamp(min=0.)
    top = (top_envelope - F.pad(top_envelope[:, :-1, :], [0, 0, 1, 0])).clamp(min=0.)
    bottom = (bottom_envelope - F.pad(bottom_envelope[:, 1:, :], [0, 0, 0, 1])).clamp(min=0.)
    lw = left.sum(2); rw = right.sum(2); tw = top.sum(1); bw = bottom.sum(1)
    lx = (left * xs[None, None]).sum(2) / lw.clamp(min=1e-6)
    rx = (right * xs[None, None]).sum(2) / rw.clamp(min=1e-6)
    ty = (top * ys[None, :, None]).sum(1) / tw.clamp(min=1e-6)
    by = (bottom * ys[None, :, None]).sum(1) / bw.clamp(min=1e-6)
    # Pixel transitions occur at half a cell outside the first/last filled cell.
    lx = lx - .5 / max(w - 1, 1); rx = rx + .5 / max(w - 1, 1)
    ty = ty - .5 / max(h - 1, 1); by = by + .5 / max(h - 1, 1)
    yl = ys[None].expand_as(lx); xr = xs[None].expand_as(ty)
    # End rows/columns intersect adjacent sloping sides near a perspective
    # corner. Fit each side through the middle half, then extrapolate its line.
    active_y = support.amax(2) > .01
    active_x = support.amax(1) > .01
    ylo = torch.where(active_y, yl, torch.ones_like(yl)).amin(1)
    yhi = torch.where(active_y, yl, torch.zeros_like(yl)).amax(1)
    xlo = torch.where(active_x, xr, torch.ones_like(xr)).amin(1)
    xhi = torch.where(active_x, xr, torch.zeros_like(xr)).amax(1)
    middle_y = ((yl >= (ylo + .25 * (yhi - ylo))[:, None]) &
                (yl <= (yhi - .25 * (yhi - ylo))[:, None])).to(support.dtype)
    middle_x = ((xr >= (xlo + .25 * (xhi - xlo))[:, None]) &
                (xr <= (xhi - .25 * (xhi - xlo))[:, None])).to(support.dtype)
    lw = lw * middle_y; rw = rw * middle_y
    tw = tw * middle_x; bw = bw * middle_x
    al, bl = weighted_line(yl, lx, lw); ar, br = weighted_line(yl, rx, rw)
    at, bt = weighted_line(xr, ty, tw); ab, bb = weighted_line(xr, by, bw)
    return torch.stack([intersect_sides(al, bl, at, bt),
                        intersect_sides(ar, br, at, bt),
                        intersect_sides(ar, br, ab, bb),
                        intersect_sides(al, bl, ab, bb)], 1)


class RegionGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.correction = nn.Sequential(nn.Linear(34 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 8))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, feature: torch.Tensor, boundary_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.sigmoid(boundary_logits)
        # PyTorch 2.0 on Apple Silicon lacks MPS cummax. Keep this small,
        # differentiable geometry calculation explicitly on CPU; inference is
        # already CPU and the transfer remains inside the exported graph.
        proposal = region_to_quad(probability[:, :1].cpu()).to(feature.device)
        context = torch.cat([feature * probability[:, :1], probability], 1)
        correction = torch.tanh(self.correction(F.adaptive_avg_pool2d(context, [4, 4]).flatten(1))).view(-1, 4, 2)
        span = (proposal.amax(1, keepdim=True) - proposal.amin(1, keepdim=True)).clamp(min=.05)
        return (proposal + .2 * span * correction).clamp(0., 1.), proposal


class RegionGuidedScreenCornerSingle(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.global_head = GlobalPyramid()
        self.region_head = RegionGeometry()
        self.local_head = LocalPointHead()
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        small = F.interpolate(image, [256, 256], mode='bilinear', align_corners=False)
        _, boundary, feature = self.global_head(self.encoder((small - self.mean) / self.std))
        coarse, proposal = self.region_head(feature, boundary)
        boxes = patch_boxes(coarse)
        rgb = sample_corner_patches(image, boxes, 64)
        context = sample_corner_patches(feature, boxes, 16)
        relative, _ = self.local_head(rgb, context)
        final = (boxes[..., :2] + relative.view(-1, 4, 2) * (boxes[..., 2:] - boxes[..., :2])).clamp(0., 1.)
        return {'coarse_quad': coarse, 'final_quad': final, 'region_quad': proposal}
