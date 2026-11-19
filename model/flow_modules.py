"""Stage 2-4 modules: coarse flow, local matching, event refinement, and fusion."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.blocks import DenseBlock, up
from model.raft import RAFT, default_raft_args
from model.submodules import conv_resblock_one
from model.utils.utils import coords_grid
from model.warp import backwarp, bgr_to_gray, resize_flow


def _strip_prefix(state_dict, prefix="module."):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


class CoarseFlowEstimator(nn.Module):
    """Cost-volume flow between the reference at t and the key frames.

    Inputs are grayscale or BGR. Both pairs are stacked on the batch axis for
    one RAFT call, yielding F_{t->0} and F_{t->1} on the grid of time t.
    """

    def __init__(self, iters: int = 12, freeze_bn: bool = True):
        super().__init__()
        # input_scale=1.0: inputs are in [0, 1]; RAFT maps them to [-1, 1] as in the original.
        self.raft = RAFT(default_raft_args(), input_scale=1.0)
        self.iters = int(iters)
        self.freeze_bn = bool(freeze_bn)

    def load_pretrained(self, path: str) -> None:
        """Load official RAFT weights (e.g. raft-things.pth); 'module.' prefixes are stripped."""
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 1.13 has no weights_only argument
            checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.raft.load_state_dict(_strip_prefix(checkpoint), strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        # RAFT's context encoder uses BatchNorm; with batch size 1 its statistics are
        # meaningless, so (like RAFT fine-tuning) keep BN layers in eval mode.
        if mode and self.freeze_bn:
            self.raft.freeze_bn()
        return self

    def forward(self, ref_t: torch.Tensor, frame0: torch.Tensor, frame1: torch.Tensor,
                iters: Optional[int] = None, *, return_quarter: bool = False):
        batch = ref_t.shape[0]
        channels = ref_t.shape[1]
        if channels not in (1, 3) or frame0.shape[1] != channels or frame1.shape[1] != channels:
            raise ValueError("RAFT reference and key frames must all be grayscale or all be BGR")
        image1 = torch.cat((ref_t, ref_t), dim=0)
        image2 = torch.cat((frame0, frame1), dim=0)
        if channels == 1:
            image1 = image1.repeat(1, 3, 1, 1)
            image2 = image2.repeat(1, 3, 1, 1)
        else:
            image1 = image1[:, [2, 1, 0]]
            image2 = image2[:, [2, 1, 0]]
        # RAFT runs its encoders with autocast disabled, so feed fp32 even under --amp.
        result = self.raft(image1.float(), image2.float(), iters=iters or self.iters,
                           return_quarter=return_quarter)
        if return_quarter:
            predictions, flow_eighth, quarter_features = result
        else:
            predictions = result
        flows_t0 = [flow[:batch] for flow in predictions]
        flows_t1 = [flow[batch:] for flow in predictions]
        if return_quarter:
            return flows_t0, flows_t1, (flow_eighth, *quarter_features)
        return flows_t0, flows_t1


class QuarterFlowMatcher(nn.Module):
    """Search the original key-frame features around x + upsampled RAFT flow(x)."""

    def __init__(self, radius: int):
        super().__init__()
        self.radius = radius
        axis = torch.arange(-radius, radius + 1)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((x, y), dim=-1).reshape(-1, 2)
        self.register_buffer("offsets", offsets, persistent=False)
        # RAFT BasicEncoder.layer2 provides 96 channels at quarter resolution.
        self.project = nn.Conv2d(96, 32, 1)
        self.update = nn.Sequential(
            conv_resblock_one(len(offsets) + 32 + 2, 64),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    @staticmethod
    def _correlation_chunk(query, source, centers, offsets):
        batch, channels, height, width = query.shape
        points = centers[:, None] + offsets[None, :, None, None]
        x, y = points.unbind(dim=-1)
        valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
        grid = torch.stack((2 * x / max(width - 1, 1) - 1,
                            2 * y / max(height - 1, 1) - 1), dim=-1)
        candidates = F.grid_sample(source, grid.reshape(batch, -1, width, 2),
                                   mode="bilinear", padding_mode="zeros", align_corners=True)
        candidates = candidates.reshape(batch, channels, len(offsets), height, width)
        similarity = (query.unsqueeze(2) * candidates).sum(dim=1)
        # Valid cosine scores lie between -1 and 1.
        return similarity.masked_fill(~valid, -2.0)

    def forward(self, flow_eighth, ref_features, key_features):
        batch, _, height, width = ref_features.shape
        flow = resize_flow(flow_eighth.float(), size=(height, width))
        centers = (coords_grid(batch, height, width, device=flow.device) + flow).permute(0, 2, 3, 1)
        query = F.normalize(self.project(ref_features).float(), dim=1, eps=1e-6)
        source = F.normalize(self.project(key_features).float(), dim=1, eps=1e-6)
        correlation = []
        for offsets in self.offsets.split(9):
            if self.training and torch.is_grad_enabled():
                # Save scores and recompute sampled features during backward.
                scores = checkpoint(self._correlation_chunk, query, source, centers, offsets,
                                    use_reentrant=False, preserve_rng_state=False)
            else:
                scores = self._correlation_chunk(query, source, centers, offsets)
            correlation.append(scores)
        correlation = torch.cat(correlation, dim=1)
        residual = self.radius * torch.tanh(self.update(torch.cat((query, correlation, flow / 5.0), dim=1)))
        return residual * (correlation.amax(dim=1, keepdim=True) > -2.0)


class EventFlowRefiner(nn.Module):
    """Refine a coarse flow F_{t->k} with the events between t and k.

    Inputs per direction (weights are shared by both directions because the events are
    always given in the t -> k direction, see ``reverse_voxel``):
        coarse flow, I_k warped to t, the reference at t, |gray(warp) - ref|,
        voxel E_{t->k}, event mask.
    Outputs: refined flow (coarse + residual) and an occlusion logit for frame k.
    """

    def __init__(self, bins: int, flow_norm: float = 20.0, iters: int = 1):
        super().__init__()
        in_ch = 2 + 3 + 1 + 1 + bins + 1
        self.flow_norm = float(flow_norm)
        self.iters = int(iters)
        self.enc1 = conv_resblock_one(in_ch, 32)
        self.enc2 = conv_resblock_one(32, 64, stride=2)
        self.enc3 = conv_resblock_one(64, 96, stride=2)
        self.dec2 = up(96, 64)
        self.dec1 = up(64, 32)
        self.head = nn.Conv2d(32, 3, 3, 1, 1)
        nn.init.zeros_(self.head.weight)  # start exactly from the coarse flow
        nn.init.zeros_(self.head.bias)

    def _step(self, flow, image_k, ref_t, events_tk, mask):
        warped = backwarp(image_k, flow)
        error = (bgr_to_gray(warped) - ref_t).abs()
        x = torch.cat((flow / self.flow_norm, warped, ref_t, error, events_tk, mask), dim=1)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        y = self.dec1(self.dec2(s3, s2), s1)
        out = self.head(y)
        return flow + out[:, :2], out[:, 2:3]

    def forward(self, coarse_flow, image_k, ref_t, events_tk, mask):
        flow = coarse_flow
        # iters == 0 disables the refinement (ablation): coarse flow, neutral occlusion logit.
        occ_logit = coarse_flow.new_zeros(coarse_flow.shape[0], 1, *coarse_flow.shape[-2:])
        for _ in range(self.iters):
            flow, occ_logit = self._step(flow, image_k, ref_t, events_tk, mask)
        return flow, occ_logit


class SelectiveFusion(nn.Module):
    """Pixel-wise selection between the two warped frames: I_t = M * W_0 + (1 - M) * W_1."""

    def __init__(self, flow_norm: float = 20.0):
        super().__init__()
        # warp0(3) + warp1(3) + ref(1) + vis0(1) + vis1(1) + flows(4) + event mask(1) + t(1)
        self.net = DenseBlock(15, 1)
        self.flow_norm = float(flow_norm)

    def forward(self, warp0, warp1, ref_t, vis0, vis1, flow_t0, flow_t1, mask, t_map):
        x = torch.cat((warp0, warp1, ref_t, vis0, vis1,
                       flow_t0 / self.flow_norm, flow_t1 / self.flow_norm, mask, t_map), dim=1)
        weight0 = self.net(x)  # sigmoid output in (0, 1)
        fused = weight0 * warp0 + (1.0 - weight0) * warp1
        return fused, weight0
