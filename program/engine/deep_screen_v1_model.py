from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from corner_train import decode_model_output


def build_roi_boxes_from_quads(quads: torch.Tensor, expand_ratio: float = 0.08) -> torch.Tensor:
    mins = torch.clamp(torch.amin(quads, dim=1), 0.0, 1.0)
    maxs = torch.clamp(torch.amax(quads, dim=1), 0.0, 1.0)
    span = torch.clamp(maxs - mins, min=1e-3)
    expand = span * expand_ratio
    x0y0 = torch.clamp(mins - expand, 0.0, 1.0)
    x1y1 = torch.clamp(maxs + expand, 0.0, 1.0)
    return torch.cat([x0y0, x1y1], dim=-1)


def sample_roi_features(feature_map: torch.Tensor, boxes: torch.Tensor, roi_size: int) -> torch.Tensor:
    batch = feature_map.shape[0]
    xs = torch.linspace(0.0, 1.0, roi_size, device=feature_map.device, dtype=feature_map.dtype)
    ys = torch.linspace(0.0, 1.0, roi_size, device=feature_map.device, dtype=feature_map.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    x0 = boxes[:, 0].view(batch, 1, 1, 1)
    y0 = boxes[:, 1].view(batch, 1, 1, 1)
    x1 = boxes[:, 2].view(batch, 1, 1, 1)
    y1 = boxes[:, 3].view(batch, 1, 1, 1)
    grid = torch.empty_like(base_grid)
    grid[..., 0] = x0[..., 0] + base_grid[..., 0] * (x1[..., 0] - x0[..., 0])
    grid[..., 1] = y0[..., 0] + base_grid[..., 1] * (y1[..., 0] - y0[..., 0])
    grid = grid * 2.0 - 1.0
    return F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True)


def build_roi_context_features(roi_features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(roi_features, (1, 1)).flatten(1)
    geom = torch.stack(
        [
            (boxes[:, 0] + boxes[:, 2]) * 0.5,
            (boxes[:, 1] + boxes[:, 3]) * 0.5,
            torch.clamp(boxes[:, 2] - boxes[:, 0], min=1e-3),
            torch.clamp(boxes[:, 3] - boxes[:, 1], min=1e-3),
        ],
        dim=-1,
    )
    return torch.cat([pooled, geom], dim=-1)


def apply_visibility_guided_process_delta(
    coarse_quad: torch.Tensor,
    roi_boxes: torch.Tensor,
    process_delta: torch.Tensor,
    process_visibility: torch.Tensor,
    process_fallback_logits: torch.Tensor,
    refine_scale: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    roi_span = (roi_boxes[:, None, 2:4] - roi_boxes[:, None, 0:2]).clamp(min=1e-3)
    corner_visibility = torch.amin(process_visibility, dim=-1)
    low_visibility_gate = torch.sigmoid((0.5 - corner_visibility) * 8.0)
    fallback_gate = torch.sigmoid(process_fallback_logits)
    refine_gate = low_visibility_gate * fallback_gate
    if refine_scale is None:
        scale = torch.tensor(1.0, dtype=coarse_quad.dtype, device=coarse_quad.device)
    elif isinstance(refine_scale, torch.Tensor):
        scale = refine_scale.to(device=coarse_quad.device, dtype=coarse_quad.dtype)
    else:
        scale = torch.tensor(float(refine_scale), dtype=coarse_quad.dtype, device=coarse_quad.device)
    refined_quad = torch.clamp(coarse_quad + scale * refine_gate.unsqueeze(-1) * process_delta * roi_span, 0.0, 1.0)
    return refined_quad, refine_gate


def load_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible.keys())


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualRoiAdapterBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.act(self.conv1(x)))
        return x + residual


class SceneContextHead(nn.Module):
    def __init__(self, in_channels: int, scene_classes: int = 4, embedding_dim: int = 8) -> None:
        super().__init__()
        hidden = max(in_channels, 32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
        )
        self.scene_logits = nn.Linear(hidden, scene_classes)
        self.scene_embedding = nn.Linear(hidden, embedding_dim)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(feature_map).flatten(1)
        hidden = self.mlp(pooled)
        return self.scene_logits(hidden), self.scene_embedding(hidden)


class CoarseSceneAdapter(nn.Module):
    def __init__(self, channels: int, scene_embedding_dim: int = 8) -> None:
        super().__init__()
        self.adapter = None
        if int(scene_embedding_dim) > 0:
            self.adapter = nn.Linear(int(scene_embedding_dim), channels)
            nn.init.zeros_(self.adapter.weight)
            nn.init.zeros_(self.adapter.bias)

    def forward(self, feature_map: torch.Tensor, scene_embedding: torch.Tensor | None = None) -> torch.Tensor:
        if scene_embedding is None or self.adapter is None:
            return feature_map
        return feature_map + self.adapter(scene_embedding).unsqueeze(-1).unsqueeze(-1)


class RoiStageHead(nn.Module):
    def __init__(self, in_channels: int, scene_embedding_dim: int = 8) -> None:
        super().__init__()
        input_dim = in_channels + 4
        hidden = max(in_channels, 32)
        self.scene_adapter = None
        if int(scene_embedding_dim) > 0:
            self.scene_adapter = nn.Linear(int(scene_embedding_dim), input_dim)
            nn.init.zeros_(self.scene_adapter.weight)
            nn.init.zeros_(self.scene_adapter.bias)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 8),
        )
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(
        self,
        roi_features: torch.Tensor,
        boxes: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = build_roi_context_features(roi_features, boxes)
        if scene_embedding is not None and self.scene_adapter is not None:
            features = features + self.scene_adapter(scene_embedding)
        roi_points = torch.sigmoid(self.head(features)).view(-1, 4, 2)
        top_left = boxes[:, None, 0:2]
        span = (boxes[:, None, 2:4] - boxes[:, None, 0:2]).clamp(min=1e-3)
        return torch.clamp(top_left + roi_points * span, 0.0, 1.0)


class ProcessDistillationHead(nn.Module):
    def __init__(self, in_channels: int, scene_embedding_dim: int = 8) -> None:
        super().__init__()
        input_dim = in_channels + 4
        hidden = max(in_channels, 32)
        self.scene_adapter = None
        if int(scene_embedding_dim) > 0:
            self.scene_adapter = nn.Linear(int(scene_embedding_dim), input_dim)
            nn.init.zeros_(self.scene_adapter.weight)
            nn.init.zeros_(self.scene_adapter.bias)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Linear(hidden, 8)
        self.visibility_head = nn.Linear(hidden, 8)
        self.edge_head = nn.Linear(hidden, 20)
        self.fallback_head = nn.Linear(hidden, 4)
        for head in (self.delta_head, self.visibility_head, self.edge_head, self.fallback_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        roi_features: torch.Tensor,
        boxes: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = build_roi_context_features(roi_features, boxes)
        if scene_embedding is not None and self.scene_adapter is not None:
            features = features + self.scene_adapter(scene_embedding)
        hidden = self.trunk(features)
        process_delta = torch.tanh(self.delta_head(hidden)).view(-1, 4, 2)
        process_visibility = torch.sigmoid(self.visibility_head(hidden)).view(-1, 4, 2)
        process_edge = self.edge_head(hidden).view(-1, 4, 5)
        process_fallback_logits = self.fallback_head(hidden).view(-1, 4)
        return process_delta, process_visibility, process_edge, process_fallback_logits


class LocalRefineMoEHead(nn.Module):
    def __init__(self, in_channels: int, experts: int = 3, scene_embedding_dim: int = 8) -> None:
        super().__init__()
        self.experts = experts
        input_dim = in_channels + 4
        hidden = max(in_channels, 32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.scene_adapter = None
        if int(scene_embedding_dim) > 0:
            self.scene_adapter = nn.Linear(int(scene_embedding_dim), input_dim)
            nn.init.zeros_(self.scene_adapter.weight)
            nn.init.zeros_(self.scene_adapter.bias)
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, experts),
        )
        self.expert_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, 8),
                )
                for _ in range(experts)
            ]
        )

    def forward(
        self,
        roi_features: torch.Tensor,
        boxes: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = build_roi_context_features(roi_features, boxes)
        if scene_embedding is not None and self.scene_adapter is not None:
            features = features + self.scene_adapter(scene_embedding)
        router_logits = self.router(features)
        gates = torch.softmax(router_logits, dim=-1)
        expert_outputs = torch.stack([head(features) for head in self.expert_heads], dim=1)
        roi_points = torch.sigmoid(torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)).view(-1, 4, 2)
        top_left = boxes[:, None, 0:2]
        span = (boxes[:, None, 2:4] - boxes[:, None, 0:2]).clamp(min=1e-3)
        final_quad = torch.clamp(top_left + roi_points * span, 0.0, 1.0)
        return final_quad, router_logits


class ResidualQuadRefineHead(nn.Module):
    def __init__(self, in_channels: int, layers: int = 2, scene_embedding_dim: int = 8) -> None:
        super().__init__()
        input_dim = in_channels + 4
        hidden = max(in_channels, 32)
        self.scene_adapter = None
        if int(scene_embedding_dim) > 0:
            self.scene_adapter = nn.Linear(int(scene_embedding_dim), input_dim)
            nn.init.zeros_(self.scene_adapter.weight)
            nn.init.zeros_(self.scene_adapter.bias)
        trunk_layers: list[nn.Module] = []
        depth = max(int(layers), 1)
        current_dim = input_dim
        for _ in range(depth):
            trunk_layers.append(nn.Linear(current_dim, hidden))
            trunk_layers.append(nn.ReLU(inplace=True))
            current_dim = hidden
        self.trunk = nn.Sequential(*trunk_layers)
        self.delta_head = nn.Linear(hidden, 8)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.blend_logit = nn.Parameter(torch.full((1,), -2.2))

    def forward(
        self,
        roi_features: torch.Tensor,
        boxes: torch.Tensor,
        base_quad: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = build_roi_context_features(roi_features, boxes)
        if scene_embedding is not None and self.scene_adapter is not None:
            features = features + self.scene_adapter(scene_embedding)
        hidden = self.trunk(features)
        delta = torch.tanh(self.delta_head(hidden)).view(-1, 4, 2)
        span = (boxes[:, None, 2:4] - boxes[:, None, 0:2]).clamp(min=1e-3)
        scale = torch.tanh(self.residual_scale)
        residual_quad = torch.clamp(base_quad + scale * delta * span, 0.0, 1.0)
        blend_weight = torch.sigmoid(self.blend_logit).view(1, 1, 1).expand(base_quad.shape[0], 1, 1)
        final_quad = torch.clamp(base_quad + blend_weight * (residual_quad - base_quad), 0.0, 1.0)
        return residual_quad, final_quad, blend_weight


class CoarseResidualHead(nn.Module):
    def __init__(self, in_channels: int, layers: int = 2, scene_embedding_dim: int = 8) -> None:
        super().__init__()
        input_dim = in_channels + 8
        hidden = max(in_channels, 32)
        self.scene_adapter = None
        if int(scene_embedding_dim) > 0:
            self.scene_adapter = nn.Linear(int(scene_embedding_dim), input_dim)
            nn.init.zeros_(self.scene_adapter.weight)
            nn.init.zeros_(self.scene_adapter.bias)
        trunk_layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(max(int(layers), 1)):
            trunk_layers.append(nn.Linear(current_dim, hidden))
            trunk_layers.append(nn.ReLU(inplace=True))
            current_dim = hidden
        self.trunk = nn.Sequential(*trunk_layers)
        self.delta_head = nn.Linear(hidden, 8)
        self.gate_head = nn.Linear(hidden, 4)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -8.0)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        feature_map: torch.Tensor,
        coarse_quad: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
        features = torch.cat([pooled, coarse_quad.view(coarse_quad.shape[0], -1)], dim=-1)
        if scene_embedding is not None and self.scene_adapter is not None:
            features = features + self.scene_adapter(scene_embedding)
        hidden = self.trunk(features)
        delta = torch.tanh(self.delta_head(hidden)).view(-1, 4, 2)
        gate = torch.sigmoid(self.gate_head(hidden)).view(-1, 4, 1)
        span = (torch.amax(coarse_quad, dim=1, keepdim=True) - torch.amin(coarse_quad, dim=1, keepdim=True)).clamp(min=1e-3)
        scale = torch.tanh(self.residual_scale)
        refined_quad = torch.clamp(coarse_quad + scale * gate * delta * span, 0.0, 1.0)
        return refined_quad, gate


class SpatialResidualRefineHead(nn.Module):
    def __init__(self, in_channels: int, roi_size: int, layers: int = 2) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for _ in range(max(int(layers), 1)):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.tower = nn.Sequential(*blocks)
        self.heatmap_head = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.offset_head = nn.Conv2d(in_channels, 8, kernel_size=1)
        nn.init.zeros_(self.heatmap_head.weight)
        nn.init.zeros_(self.heatmap_head.bias)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.roi_size = roi_size

    def forward(self, roi_features: torch.Tensor, boxes: torch.Tensor, base_quad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.tower(roi_features)
        heatmaps = self.heatmap_head(features)
        offsets = self.offset_head(features).view(roi_features.shape[0], 4, 2, self.roi_size, self.roi_size)
        roi_points = decode_model_output((heatmaps, offsets), decode_mode="soft_argmax_offset", head_mode="heatmap_offset")
        top_left = boxes[:, None, 0:2]
        span = (boxes[:, None, 2:4] - boxes[:, None, 0:2]).clamp(min=1e-3)
        spatial_quad = torch.clamp(top_left + roi_points * span, 0.0, 1.0)
        scale = torch.tanh(self.residual_scale)
        refined_quad = torch.clamp(base_quad + scale * (spatial_quad - base_quad), 0.0, 1.0)
        return spatial_quad, refined_quad


class StrictSpatialRefineHead(nn.Module):
    def __init__(self, in_channels: int, roi_size: int, layers: int = 2, scene_embedding_dim: int = 8) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for _ in range(max(int(layers), 1)):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.tower = nn.Sequential(*blocks)
        self.scene_adapter = None
        if int(scene_embedding_dim) > 0:
            self.scene_adapter = nn.Linear(int(scene_embedding_dim), in_channels)
            nn.init.zeros_(self.scene_adapter.weight)
            nn.init.zeros_(self.scene_adapter.bias)
        self.heatmap_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.gate_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
        )
        nn.init.zeros_(self.heatmap_head.weight)
        nn.init.zeros_(self.heatmap_head.bias)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.gate_head[2].weight)
        nn.init.constant_(self.gate_head[2].bias, -8.0)
        self.roi_size = roi_size
        self.patch_radius_ratio = 0.18

    def _sample_corner_patches(
        self,
        roi_features: torch.Tensor,
        boxes: torch.Tensor,
        base_quad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels, _, _ = roi_features.shape
        top_left = boxes[:, None, 0:2]
        span = (boxes[:, None, 2:4] - boxes[:, None, 0:2]).clamp(min=1e-3)
        base_rel = torch.clamp((base_quad - top_left) / span, 0.0, 1.0)
        radius = torch.full_like(base_rel, self.patch_radius_ratio)
        patch_min = torch.clamp(base_rel - radius, 0.0, 1.0)
        patch_max = torch.clamp(base_rel + radius, 0.0, 1.0)
        xs = torch.linspace(0.0, 1.0, self.roi_size, device=roi_features.device, dtype=roi_features.dtype)
        ys = torch.linspace(0.0, 1.0, self.roi_size, device=roi_features.device, dtype=roi_features.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, self.roi_size, self.roi_size, 2)
        patch_grid = patch_min[:, :, None, None, :] + base_grid * (patch_max - patch_min)[:, :, None, None, :]
        expanded_features = roi_features[:, None, :, :, :].expand(batch, 4, channels, roi_features.shape[-2], roi_features.shape[-1])
        patch_features = F.grid_sample(
            expanded_features.reshape(batch * 4, channels, roi_features.shape[-2], roi_features.shape[-1]),
            patch_grid.reshape(batch * 4, self.roi_size, self.roi_size, 2) * 2.0 - 1.0,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return patch_features, patch_min, patch_max

    def forward(
        self,
        roi_features: torch.Tensor,
        boxes: torch.Tensor,
        base_quad: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = roi_features
        if scene_embedding is not None and self.scene_adapter is not None:
            scene_bias = self.scene_adapter(scene_embedding).unsqueeze(-1).unsqueeze(-1)
            features = features + scene_bias
        patch_features, patch_min, patch_max = self._sample_corner_patches(features, boxes, base_quad)
        patch_features = self.tower(patch_features)
        heatmaps = self.heatmap_head(patch_features).view(roi_features.shape[0], 4, self.roi_size, self.roi_size)
        offsets = self.offset_head(patch_features).view(roi_features.shape[0], 4, 2, self.roi_size, self.roi_size)
        blend_weight = torch.sigmoid(self.gate_head(patch_features)).view(roi_features.shape[0], 4, 1)
        patch_points = decode_model_output(
            (
                heatmaps.view(roi_features.shape[0] * 4, 1, self.roi_size, self.roi_size),
                offsets.view(roi_features.shape[0] * 4, 1, 2, self.roi_size, self.roi_size),
            ),
            decode_mode="soft_argmax_offset",
            head_mode="heatmap_offset",
        )
        patch_points = patch_points.view(roi_features.shape[0], 4, 2)
        roi_points = torch.clamp(patch_min + patch_points * (patch_max - patch_min), 0.0, 1.0)
        top_left = boxes[:, None, 0:2]
        span = (boxes[:, None, 2:4] - boxes[:, None, 0:2]).clamp(min=1e-3)
        strict_quad = torch.clamp(top_left + roi_points * span, 0.0, 1.0)
        refined_quad = torch.clamp(base_quad + blend_weight * (strict_quad - base_quad), 0.0, 1.0)
        return heatmaps, offsets, strict_quad, refined_quad, blend_weight


class CandidateSelectionHead(nn.Module):
    def __init__(self, in_channels: int, candidate_count: int = 3, scene_embedding_dim: int = 8) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        input_dim = in_channels + 4 + 8
        hidden = max(in_channels, 32)
        self.scene_adapter = None
        if int(scene_embedding_dim) > 0:
            self.scene_adapter = nn.Linear(int(scene_embedding_dim), input_dim)
            nn.init.zeros_(self.scene_adapter.weight)
            nn.init.zeros_(self.scene_adapter.bias)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(
        self,
        roi_features: torch.Tensor,
        boxes: torch.Tensor,
        candidate_quads: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = build_roi_context_features(roi_features, boxes)
        box_top_left = boxes[:, None, None, 0:2]
        box_span = (boxes[:, None, None, 2:4] - boxes[:, None, None, 0:2]).clamp(min=1e-3)
        candidate_rel = torch.clamp((candidate_quads - box_top_left) / box_span, 0.0, 1.0).view(candidate_quads.shape[0], candidate_quads.shape[1], -1)
        features = torch.cat([context[:, None, :].expand(-1, candidate_quads.shape[1], -1), candidate_rel], dim=-1)
        if scene_embedding is not None and self.scene_adapter is not None:
            features = features + self.scene_adapter(scene_embedding)[:, None, :]
        scores = self.head(features).squeeze(-1)
        candidate_bias = torch.linspace(-0.5, 0.0, steps=candidate_quads.shape[1], device=scores.device, dtype=scores.dtype)
        return scores + candidate_bias.view(1, -1)


class StateAwareCandidateHead(nn.Module):
    def __init__(self, in_channels: int, scene_embedding_dim: int = 8) -> None:
        super().__init__()
        input_dim = in_channels + 4
        hidden = max(in_channels, 32)
        self.scene_adapter = None
        if int(scene_embedding_dim) > 0:
            self.scene_adapter = nn.Linear(int(scene_embedding_dim), input_dim)
            nn.init.zeros_(self.scene_adapter.weight)
            nn.init.zeros_(self.scene_adapter.bias)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Linear(hidden, 8)
        self.corner_state_head = nn.Linear(hidden, 4)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.corner_state_head.weight)
        nn.init.zeros_(self.corner_state_head.bias)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        roi_features: torch.Tensor,
        boxes: torch.Tensor,
        coarse_quad: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = build_roi_context_features(roi_features, boxes)
        if scene_embedding is not None and self.scene_adapter is not None:
            features = features + self.scene_adapter(scene_embedding)
        hidden = self.trunk(features)
        delta = torch.tanh(self.delta_head(hidden)).view(-1, 4, 2)
        corner_state_logits = self.corner_state_head(hidden).view(-1, 4)
        span = (boxes[:, None, 2:4] - boxes[:, None, 0:2]).clamp(min=1e-3)
        state_gate = torch.sigmoid(corner_state_logits).unsqueeze(-1)
        scale = torch.tanh(self.residual_scale)
        candidate_quad = torch.clamp(coarse_quad + scale * state_gate * delta * span, 0.0, 1.0)
        return candidate_quad, corner_state_logits


class DeepScreenV1Net(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        roi_size: int = 16,
        experts: int = 3,
        expand_ratio: float = 0.08,
        roi_adapter_layers: int = 0,
        spatial_refine_layers: int = 0,
        residual_quad_head_layers: int = 0,
        coarse_residual_head_layers: int = 0,
        strict_spatial_refine_layers: int = 0,
        candidate_selection_enabled: bool = False,
        state_aware_candidate_enabled: bool = False,
        internal_candidate_names: list[str] | tuple[str, ...] | None = None,
        final_output_mode: str = "base_final",
        scene_classes: int = 4,
        scene_embedding_dim: int = 8,
        coarse_visibility_refine_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.expand_ratio = expand_ratio
        self.final_output_mode = str(final_output_mode or "base_final")
        self.coarse_visibility_refine_enabled = bool(coarse_visibility_refine_enabled)
        self.internal_candidate_names = tuple(
            item
            for item in (internal_candidate_names or ("coarse_quad", "roi_stage_quad", "base_final_quad"))
            if str(item).strip()
        )
        self.stem = ConvBlock(in_channels, base_channels, stride=2)
        self.stage2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.stage3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        fpn_channels = base_channels * 2
        self.lat2 = nn.Conv2d(base_channels * 2, fpn_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(base_channels * 4, fpn_channels, kernel_size=1)
        self.coarse_heatmap_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, 4, kernel_size=1),
        )
        self.coarse_offset_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, 8, kernel_size=1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.coarse_offset_head[2].weight)
        nn.init.zeros_(self.coarse_offset_head[2].bias)
        self.coarse_scene_adapter = CoarseSceneAdapter(fpn_channels, scene_embedding_dim=scene_embedding_dim)
        adapter_blocks: list[nn.Module] = []
        for _ in range(max(int(roi_adapter_layers), 0)):
            adapter_blocks.append(ResidualRoiAdapterBlock(fpn_channels))
        self.roi_adapter = nn.Identity() if not adapter_blocks else nn.Sequential(*adapter_blocks)
        self.scene_context_head = SceneContextHead(fpn_channels, scene_classes=scene_classes, embedding_dim=scene_embedding_dim)
        self.roi_stage_head = RoiStageHead(fpn_channels, scene_embedding_dim=scene_embedding_dim)
        self.local_refine_head = LocalRefineMoEHead(fpn_channels, experts=experts, scene_embedding_dim=scene_embedding_dim)
        self.process_head = ProcessDistillationHead(fpn_channels, scene_embedding_dim=scene_embedding_dim)
        self.spatial_refine_head = (
            None
            if int(spatial_refine_layers) <= 0
            else SpatialResidualRefineHead(fpn_channels, roi_size=roi_size, layers=spatial_refine_layers)
        )
        self.residual_quad_head = (
            None
            if int(residual_quad_head_layers) <= 0
            else ResidualQuadRefineHead(fpn_channels, layers=residual_quad_head_layers, scene_embedding_dim=scene_embedding_dim)
        )
        self.coarse_residual_head = (
            None
            if int(coarse_residual_head_layers) <= 0
            else CoarseResidualHead(fpn_channels, layers=coarse_residual_head_layers, scene_embedding_dim=scene_embedding_dim)
        )
        self.strict_spatial_refine_head = (
            None
            if int(strict_spatial_refine_layers) <= 0
            else StrictSpatialRefineHead(
                fpn_channels,
                roi_size=roi_size,
                layers=strict_spatial_refine_layers,
                scene_embedding_dim=scene_embedding_dim,
            )
        )
        self.candidate_selection_head = (
            CandidateSelectionHead(fpn_channels, candidate_count=3, scene_embedding_dim=scene_embedding_dim)
            if candidate_selection_enabled
            else None
        )
        self.state_aware_candidate_head = (
            StateAwareCandidateHead(fpn_channels, scene_embedding_dim=scene_embedding_dim)
            if state_aware_candidate_enabled
            else None
        )

    @staticmethod
    def _candidate_tensor(output: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        if name not in output:
            raise KeyError(f"candidate '{name}' not found in model output")
        return output[name]

    def _internal_candidate_quads(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([self._candidate_tensor(output, name) for name in self.internal_candidate_names], dim=1)

    def assemble_candidate_pool(
        self,
        output: dict[str, torch.Tensor],
        external_candidate_quads: torch.Tensor | None = None,
        external_candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        internal_candidates = self._internal_candidate_quads(output)
        batch_size = internal_candidates.shape[0]
        internal_mask = torch.ones((batch_size, internal_candidates.shape[1]), dtype=torch.bool, device=internal_candidates.device)
        if external_candidate_quads is None:
            return {"candidate_quads": internal_candidates, "candidate_mask": internal_mask}
        if external_candidate_quads.dim() == 3:
            external_candidate_quads = external_candidate_quads.unsqueeze(1)
        external_candidate_quads = external_candidate_quads.to(device=internal_candidates.device, dtype=internal_candidates.dtype)
        if external_candidate_mask is None:
            external_candidate_mask = torch.ones(
                (batch_size, external_candidate_quads.shape[1]),
                dtype=torch.bool,
                device=internal_candidates.device,
            )
        else:
            external_candidate_mask = external_candidate_mask.to(device=internal_candidates.device, dtype=torch.bool)
            if external_candidate_mask.dim() == 1:
                external_candidate_mask = external_candidate_mask.unsqueeze(1)
        return {
            "candidate_quads": torch.cat([external_candidate_quads, internal_candidates], dim=1),
            "candidate_mask": torch.cat([external_candidate_mask, internal_mask], dim=1),
        }

    def score_candidate_pool(
        self,
        output: dict[str, torch.Tensor],
        candidate_quads: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.candidate_selection_head is None:
            raise RuntimeError("candidate selection head is disabled")
        candidate_scores = self.candidate_selection_head(
            output["roi_features"],
            output["roi_boxes"],
            candidate_quads,
            scene_embedding=output.get("scene_embedding"),
        )
        if candidate_mask is not None:
            candidate_scores = candidate_scores.masked_fill(~candidate_mask, -1e9)
        return candidate_scores

    def select_candidate_pool(
        self,
        output: dict[str, torch.Tensor],
        external_candidate_quads: torch.Tensor | None = None,
        external_candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pool = self.assemble_candidate_pool(
            output,
            external_candidate_quads=external_candidate_quads,
            external_candidate_mask=external_candidate_mask,
        )
        if self.candidate_selection_head is None:
            candidate_scores = torch.zeros(
                (pool["candidate_quads"].shape[0], pool["candidate_quads"].shape[1]),
                dtype=pool["candidate_quads"].dtype,
                device=pool["candidate_quads"].device,
            )
            if pool["candidate_mask"].shape[1] > 0:
                candidate_scores = candidate_scores.masked_fill(~pool["candidate_mask"], -1e9)
            candidate_selected_index = torch.argmax(candidate_scores, dim=-1)
        else:
            candidate_scores = self.score_candidate_pool(
                output,
                pool["candidate_quads"],
                candidate_mask=pool["candidate_mask"],
            )
            candidate_selected_index = torch.argmax(candidate_scores, dim=-1)
        batch_indices = torch.arange(pool["candidate_quads"].shape[0], device=pool["candidate_quads"].device)
        selected_quad = pool["candidate_quads"][batch_indices, candidate_selected_index]
        return {
            "candidate_quads": pool["candidate_quads"],
            "candidate_mask": pool["candidate_mask"],
            "candidate_scores": candidate_scores,
            "candidate_selected_index": candidate_selected_index,
            "selected_quad": selected_quad,
        }

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        c1 = self.stem(image)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        scene_logits, scene_embedding = self.scene_context_head(p3)
        coarse_p2 = self.coarse_scene_adapter(p2, scene_embedding=scene_embedding)
        coarse_heatmaps = self.coarse_heatmap_head(coarse_p2)
        coarse_offsets = self.coarse_offset_head(coarse_p2).view(image.shape[0], 4, 2, p2.shape[-2], p2.shape[-1])
        coarse_quad = decode_model_output(
            (coarse_heatmaps, coarse_offsets),
            decode_mode="soft_argmax_offset",
            head_mode="heatmap_offset",
        )
        roi_boxes = build_roi_boxes_from_quads(coarse_quad.detach(), expand_ratio=self.expand_ratio)
        roi_features = sample_roi_features(coarse_p2, roi_boxes, self.roi_size)
        adapted_roi_features = self.roi_adapter(roi_features)
        roi_stage_quad = self.roi_stage_head(adapted_roi_features, roi_boxes, scene_embedding=scene_embedding)
        base_final_quad, router_logits = self.local_refine_head(adapted_roi_features, roi_boxes, scene_embedding=scene_embedding)
        process_delta, process_visibility, process_edge, process_fallback_logits = self.process_head(
            adapted_roi_features,
            roi_boxes,
            scene_embedding=scene_embedding,
        )
        visibility_refined_quad = coarse_quad
        visibility_refine_gate = torch.zeros_like(process_fallback_logits)
        if self.coarse_visibility_refine_enabled:
            visibility_refined_quad, visibility_refine_gate = apply_visibility_guided_process_delta(
                coarse_quad,
                roi_boxes,
                process_delta,
                process_visibility,
                process_fallback_logits,
            )
        output = {
            "p2": p2,
            "coarse_p2": coarse_p2,
            "coarse_heatmaps": coarse_heatmaps,
            "coarse_offsets": coarse_offsets,
            "coarse_quad": coarse_quad,
            "visibility_refined_quad": visibility_refined_quad,
            "visibility_refine_gate": visibility_refine_gate,
            "roi_boxes": roi_boxes,
            "roi_features": adapted_roi_features,
            "roi_stage_quad": roi_stage_quad,
            "scene_logits": scene_logits,
            "scene_embedding": scene_embedding,
            "base_final_quad": base_final_quad,
            "final_quad": base_final_quad,
            "router_logits": router_logits,
            "process_delta": process_delta,
            "process_visibility": process_visibility,
            "process_edge": process_edge,
            "process_fallback_logits": process_fallback_logits,
        }
        if self.coarse_residual_head is not None:
            coarse_residual_quad, coarse_residual_gate = self.coarse_residual_head(
                coarse_p2,
                coarse_quad,
                scene_embedding=scene_embedding,
            )
            output["coarse_residual_quad"] = coarse_residual_quad
            output["coarse_residual_gate"] = coarse_residual_gate
        if self.state_aware_candidate_head is not None:
            state_aware_quad, corner_state_logits = self.state_aware_candidate_head(
                adapted_roi_features,
                roi_boxes,
                coarse_quad,
                scene_embedding=scene_embedding,
            )
            output["state_aware_quad"] = state_aware_quad
            output["corner_state_logits"] = corner_state_logits
        if self.spatial_refine_head is not None:
            spatial_quad, refined_quad = self.spatial_refine_head(adapted_roi_features, roi_boxes, base_final_quad)
            output["spatial_quad"] = spatial_quad
            output["final_quad"] = refined_quad
        if self.residual_quad_head is not None:
            residual_base = output["final_quad"]
            residual_quad, refined_quad, residual_blend_weight = self.residual_quad_head(
                adapted_roi_features,
                roi_boxes,
                residual_base,
                scene_embedding=scene_embedding,
            )
            output["residual_quad"] = residual_quad
            output["residual_blend_weight"] = residual_blend_weight
            output["final_quad"] = refined_quad
        if self.strict_spatial_refine_head is not None:
            strict_base = output["coarse_quad"] if self.final_output_mode == "coarse_strict" else output["final_quad"]
            (
                strict_point_heatmaps,
                strict_point_offsets,
                strict_point_quad,
                refined_quad,
                strict_point_blend_weight,
            ) = self.strict_spatial_refine_head(
                adapted_roi_features,
                roi_boxes,
                strict_base,
                scene_embedding=scene_embedding,
            )
            output["strict_point_heatmaps"] = strict_point_heatmaps
            output["strict_point_offsets"] = strict_point_offsets
            output["strict_point_base_quad"] = strict_base
            output["strict_point_quad"] = strict_point_quad
            output["strict_point_blend_weight"] = strict_point_blend_weight
            output["final_quad"] = refined_quad
        if self.final_output_mode == "coarse":
            output["final_quad"] = output["visibility_refined_quad"] if self.coarse_visibility_refine_enabled else output["coarse_quad"]
        elif self.final_output_mode == "coarse_residual" and "coarse_residual_quad" in output:
            output["final_quad"] = output["coarse_residual_quad"]
        elif self.final_output_mode == "coarse_strict" and "strict_point_quad" in output:
            output["final_quad"] = refined_quad
        elif self.final_output_mode == "state_aware" and "state_aware_quad" in output:
            output["final_quad"] = output["state_aware_quad"]
        elif self.candidate_selection_head is not None:
            selection = self.select_candidate_pool(output)
            output["candidate_quads"] = selection["candidate_quads"]
            output["candidate_mask"] = selection["candidate_mask"]
            output["candidate_scores"] = selection["candidate_scores"]
            output["candidate_selected_index"] = selection["candidate_selected_index"]
            output["final_quad"] = selection["selected_quad"]
        return output
