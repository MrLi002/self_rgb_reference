"""Evaluate a checkpoint on BS-ERGB (PSNR/SSIM, optional LPIPS).

Timestamp-corrected version matching the updated datasets/bsergb.py:
  * images/timestamp.txt contains one absolute timestamp per RGB frame.
  * events/000000.npz covers [T_0, T_1], events/000001.npz covers [T_1, T_2], etc.
  * voxelization uses exact frame timestamps, NOT the surviving events' min/max.
  * normalized target t = (T_target - T_start) / (T_end - T_start).

OF-AF protocol and metric definitions are unchanged from the original evaluator:
  * --protocol ofaf (default): windows stride=skip; skip_inds exclusions;
    truncating uint8 conversion; skimage-equivalent PSNR and 7x7 SSIM.
  * nonoverlap: stride=skip+1; all: stride=1; 11x11 Gaussian SSIM.
  * optional LPIPS, output CSV/JSON, full-resolution evaluation, optional PNGs.

Requires the timestamp-corrected datasets/bsergb.py previously provided.

Example:
  python eval_bsergb.py --data-root /path/BSERGB/1_TEST \\
      --checkpoint train/bsergb_full_rgb/last.pth --skips 1 3
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.size_adapter import ImgAndEventSizeAdapter
from datasets.bsergb import (
    _event_channels_or_zeros,
    _list_files,
    _load_event_chunks,
    _read_frame_timestamps,
    _sequence_from_chunks,
    _voxel_or_zeros,
)
from model.ourNet import MyNet
from util.utils_func import process_mask

# OF-AF BSERGB_Test.skip_inds (0-based image indexes).
OFAF_SKIP_INDS = {
    "basket_09": [31, 32, 33, 34],
    "may29_rooftop_handheld_02": [17, 70],
    "may29_rooftop_handheld_03": [306],
    "may29_rooftop_handheld_05": [21],
}
PROTOCOLS = ("ofaf", "nonoverlap", "all")

# Defaults for older checkpoints without stored args.
MODEL_DEFAULTS = dict(
    bins=8, stage="full", beta=0.1, mask_patch_size=32, hcs_beta_mode="sum",
    ref_mode="composite", rec_use_frames=False, rec_rgb=False,
    detach_rec_for_flow=False, raft_iters=12, refine_iters=1,
    texture_window=3, no_freeze_raft_bn=False,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on the BS-ERGB test split")
    parser.add_argument("--data-root", required=True,
                        help="BS-ERGB test root, with scene/images/timestamp.txt")
    parser.add_argument("--checkpoint", required=True, help="checkpoint from train_bsergb.py")
    parser.add_argument("--skips", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--protocol", default="ofaf", choices=PROTOCOLS,
                        help="ofaf: OF-AF windows / exclusions / metric definitions (default)")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--exclude-json", default=None,
                        help="optional extra {scene: [excluded frame indices]}")
    parser.add_argument("--max-windows-per-scene", type=int, default=None)
    parser.add_argument("--stage", default=None, choices=("flow", "full"),
                        help="override checkpoint stage; flow scores coarse result")
    parser.add_argument("--raft-iters", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=1,
                        help="number of targets in each forward pass")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


# ------------------------------------------------------------------ metrics

def quantize(image: torch.Tensor) -> torch.Tensor:
    return torch.round(image.clamp(0.0, 1.0) * 255.0) / 255.0


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in [0,1]."""
    mse = ((pred - gt) ** 2).flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def _gaussian_window(channels: int, size: int = 11, sigma: float = 1.5,
                     device=None, dtype=torch.float32):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11,
         sigma: float = 1.5) -> torch.Tensor:
    """Per-image SSIM with Gaussian valid filter, as in original evaluator."""
    channels = pred.shape[1]
    pred, gt = pred.double(), gt.double()
    window = _gaussian_window(channels, window_size, sigma, pred.device, pred.dtype)
    filt = lambda x: F.conv2d(x, window, groups=channels)
    mu1, mu2 = filt(pred), filt(gt)
    sigma1 = filt(pred * pred) - mu1 ** 2
    sigma2 = filt(gt * gt) - mu2 ** 2
    sigma12 = filt(pred * gt) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
    )
    return ssim_map.flatten(1).mean(1).float()


def to_uint8_truncate(image: torch.Tensor) -> torch.Tensor:
    """OF-AF uses `(x * 255).astype(np.uint8)`, i.e. truncation."""
    return (image.float().clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def psnr_skimage(pred8: torch.Tensor, gt8: torch.Tensor) -> torch.Tensor:
    mse = ((pred8.double() - gt8.double()) ** 2).flatten(1).mean(1)
    return (10.0 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))).float()


def ssim_skimage(pred8: torch.Tensor, gt8: torch.Tensor, win_size: int = 7) -> torch.Tensor:
    """skimage structural_similarity defaults: uint8, uniform 7x7, sample covariance."""
    x, y = pred8.double(), gt8.double()
    channels = x.shape[1]
    kernel = torch.full((channels, 1, win_size, win_size), 1.0 / win_size ** 2,
                        dtype=x.dtype, device=x.device)
    filt = lambda img: F.conv2d(img, kernel, groups=channels)
    ux, uy = filt(x), filt(y)
    uxx, uyy, uxy = filt(x * x), filt(y * y), filt(x * y)
    cov_norm = win_size ** 2 / (win_size ** 2 - 1.0)
    vx, vy, vxy = (
        cov_norm * (uxx - ux * ux),
        cov_norm * (uyy - uy * uy),
        cov_norm * (uxy - ux * uy),
    )
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    s_map = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
        (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
    )
    return s_map.flatten(2).mean(2).mean(1).float()


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor,
                    protocol: str, quantize_output: bool = True):
    """Return (PSNR, SSIM, scored_pred) for (B,3,H,W) tensors in [0,1]."""
    if protocol == "ofaf":
        if not quantize_output:
            return psnr(pred, gt), ssim_skimage(pred * 255.0, gt * 255.0), pred
        pred8, gt8 = to_uint8_truncate(pred), to_uint8_truncate(gt)
        return psnr_skimage(pred8, gt8), ssim_skimage(pred8, gt8), pred8.float() / 255.0
    scored = quantize(pred) if quantize_output else pred
    return psnr(scored, gt), ssim(scored, gt), scored


# ------------------------------------------------------------------ data

def _first(batch):
    """Use one temporal window per DataLoader item."""
    return batch[0]


class BSERGBEvalWindows(Dataset):
    """Full-resolution test windows, with all K=skip intermediate GTs per item.

    This class matches the revised training dataset's timestamp semantics.
    Image, timestamp, and event counts are verified once at initialization.
    """

    def __init__(self, data_root, skips, bins, protocol="ofaf", scenes=None,
                 excluded=None, max_windows_per_scene=None):
        self.root = Path(data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"BS-ERGB test root does not exist: {self.root}")
        if protocol not in PROTOCOLS:
            raise ValueError(f"Invalid protocol: {protocol}")
        self.bins = int(bins)
        names = list(scenes) if scenes else sorted(p.name for p in self.root.iterdir() if p.is_dir())
        merged = {k: set(v) for k, v in (OFAF_SKIP_INDS.items() if protocol == "ofaf" else [])}
        for key, value in (excluded or {}).items():
            merged.setdefault(key, set()).update(value)
        excluded = merged
        self.dropped_windows = 0
        self.scenes, self.items = [], []
        for name in names:
            image_dir = self.root / name / "images"
            event_dir = self.root / name / "events"
            images = _list_files(image_dir, (".png", ".jpg", ".jpeg"))
            events = _list_files(event_dir, (".npz",))
            if len(images) < 3 or len(events) < len(images) - 1:
                print(f"[warn] skip scene {name}: {len(images)} images, {len(events)} event files")
                continue
            if len(events) != len(images) - 1:
                raise ValueError(f"{name}: expected images-1 event files; "
                                 f"found {len(images)} images, {len(events)} event files")
            timestamps = _read_frame_timestamps(
                image_dir / "timestamp.txt", expected_images=len(images)
            )
            scene_id = len(self.scenes)
            self.scenes.append({
                "name": name, "images": images, "events": events,
                "timestamps": timestamps,
            })
            for skip in skips:
                if skip < 1:
                    raise ValueError("skip must be positive")
                offset = skip + 1
                step = {"ofaf": skip, "nonoverlap": offset, "all": 1}[protocol]
                count = 0
                for start in range(0, len(images) - offset, step):
                    end = start + offset
                    if any(f in excluded.get(name, ()) for f in range(start, end + 1)):
                        self.dropped_windows += 1
                        continue
                    if max_windows_per_scene is not None and count >= max_windows_per_scene:
                        break
                    self.items.append((scene_id, start, end, skip))
                    count += 1
        if not self.items:
            raise RuntimeError("No evaluation windows found; check --data-root / --skips / --scenes")

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _read(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index):
        scene_id, start, end, skip = self.items[index]
        scene = self.scenes[scene_id]
        timestamps = scene["timestamps"]
        img0, img1 = self._read(scene["images"][start]), self._read(scene["images"][end])
        height, width = img0.shape[-2:]
        targets = list(range(start + 1, end))
        gts = torch.stack([self._read(scene["images"][k]) for k in targets])
        chunks = _load_event_chunks(
            scene["events"][start:end],
            (0, 0, height, width),
            frame_timestamps=timestamps,
            start_interval=start,
        )
        voxels, masks = [], []
        for target in targets:
            split = target - start
            # Explicit frame-time boundaries, identical to datasets/bsergb.py train loader.
            ev_0t = _sequence_from_chunks(
                chunks[:split], height, width,
                start_time=timestamps[start], end_time=timestamps[target],
            )
            ev_t1 = _sequence_from_chunks(
                chunks[split:], height, width,
                start_time=timestamps[target], end_time=timestamps[end],
            )
            counts = torch.cat((
                _event_channels_or_zeros(ev_0t, height, width),
                _event_channels_or_zeros(ev_t1, height, width)
            ), dim=0)
            masks.append(process_mask(counts).float())
            ev_1t = None if ev_t1 is None else ev_t1.copy().reverse()
            voxels.append(torch.cat((
                _voxel_or_zeros(ev_0t, self.bins, height, width),
                _voxel_or_zeros(ev_t1, self.bins, height, width),
                _voxel_or_zeros(ev_1t, self.bins, height, width),
            ), dim=0).float())
        # Use physical target timestamps rather than equal-frame-index spacing.
        t0, t1 = float(timestamps[start]), float(timestamps[end])
        times = torch.tensor([
            (float(timestamps[k]) - t0) / (t1 - t0) for k in targets
        ], dtype=torch.float32)
        return {
            "img0": img0, "img1": img1, "gts": gts,
            "voxels": torch.stack(voxels), "masks": torch.stack(masks),
            "t": times,
            "scene": scene["name"], "start": start, "end": end,
            "skip": skip, "targets": targets,
        }


# ------------------------------------------------------------------ model

def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(checkpoint_path: str, device: torch.device, stage: Optional[str] = None,
               raft_iters: Optional[int] = None):
    checkpoint = _torch_load(checkpoint_path)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    config = dict(MODEL_DEFAULTS)
    config.update({k: v for k, v in saved_args.items() if k in MODEL_DEFAULTS})
    if stage is not None:
        config["stage"] = stage
    if raft_iters is not None:
        config["raft_iters"] = raft_iters
    if config["stage"] == "rec":
        raise ValueError("This checkpoint is from the rec stage, which only produces the reference frame. "
                         "Evaluate a flow/full checkpoint, or pass --stage flow/full explicitly.")
    model_args = argparse.Namespace(**config)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("net", "model", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    state = {(k[7:] if k.startswith("module.") else k): v
             for k, v in state.items() if torch.is_tensor(v)}
    model = MyNet(model_args)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, model_args, checkpoint.get("epoch") if isinstance(checkpoint, dict) else None


@torch.no_grad()
def predict_window(model, sample, device, bins, chunk=1, amp=False):
    """Predict K target frames. Output (K,3,H,W) in [0,1]."""
    num_targets = sample["voxels"].shape[0]
    img0 = sample["img0"].to(device)[None].expand(num_targets, -1, -1, -1)
    img1 = sample["img1"].to(device)[None].expand(num_targets, -1, -1, -1)
    imgs = torch.cat((img0, torch.zeros_like(img0), img1), dim=1)
    voxels = sample["voxels"].to(device)
    masks = sample["masks"].to(device)
    times = sample["t"].to(device)
    adapter = ImgAndEventSizeAdapter(minimum_size=64)
    imgs, voxels, masks = adapter.pad(imgs, voxels, masks)
    outputs = []
    use_amp = amp and device.type == "cuda"
    for begin in range(0, num_targets, max(1, chunk)):
        sl = slice(begin, begin + max(1, chunk))
        with torch.amp.autocast("cuda", enabled=use_amp):
            result = model(imgs[sl], voxels[sl], masks[sl], bins, times[sl])
        outputs.append(adapter.unpad(result[3][0]).float().clamp(0.0, 1.0))
    return torch.cat(outputs, dim=0)


# ------------------------------------------------------------------ evaluation

def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, model_args, epoch = load_model(args.checkpoint, device, args.stage, args.raft_iters)
    excluded = json.loads(Path(args.exclude_json).read_text()) if args.exclude_json else None
    dataset = BSERGBEvalWindows(args.data_root, args.skips, model_args.bins, args.protocol,
                                args.scenes, excluded, args.max_windows_per_scene)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=_first)
    out_dir = Path(args.output_dir or Path(args.checkpoint).resolve().parent / "eval_bsergb")
    out_dir.mkdir(parents=True, exist_ok=True)

    lpips_metric = None
    if args.lpips:
        import lpips
        lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    print(f"Checkpoint: {args.checkpoint} (stage={model_args.stage}, epoch={epoch}) | device={device} | "
          f"windows={len(dataset)} (dropped {dataset.dropped_windows}) | scenes={len(dataset.scenes)} | "
          f"skips={args.skips} | protocol={args.protocol}")
    rows: List[Dict[str, object]] = []
    gpu_time, start_time = 0.0, time.time()
    for index, sample in enumerate(loader):
        t0 = time.time()
        preds = predict_window(model, sample, device, model_args.bins, args.chunk, args.amp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time += time.time() - t0
        gts = sample["gts"].to(device)
        frame_psnr, frame_ssim, scored = compute_metrics(preds, gts, args.protocol, not args.no_quantize)
        frame_lpips = None
        if lpips_metric is not None:
            frame_lpips = lpips_metric(scored[:, [2, 1, 0]] * 2 - 1,
                                      gts[:, [2, 1, 0]] * 2 - 1).flatten()
        for k, target in enumerate(sample["targets"]):
            row = {"scene": sample["scene"], "skip": sample["skip"], "start": sample["start"],
                   "end": sample["end"], "target": target, "t": round(float(sample["t"][k]), 4),
                   "psnr": float(frame_psnr[k]), "ssim": float(frame_ssim[k])}
            if frame_lpips is not None:
                row["lpips"] = float(frame_lpips[k])
            rows.append(row)
            if args.save_images:
                path = out_dir / "images" / f"skip{sample['skip']}" / sample["scene"] / f"{target:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image = (preds[k].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
                cv2.imwrite(str(path), image)
        group_finished = index + 1 == len(dataset)
        if not group_finished:
            next_scene_id, _, _, next_skip = dataset.items[index + 1]
            next_scene = dataset.scenes[next_scene_id]["name"]
            group_finished = next_scene != sample["scene"] or next_skip != sample["skip"]
        if group_finished:
            done = [r for r in rows if r["scene"] == sample["scene"] and r["skip"] == sample["skip"]]
            print(f"  [{index + 1}/{len(dataset)}] scene={sample['scene']} skip={sample['skip']} "
                  f"frames={len(done)} PSNR={np.mean([r['psnr'] for r in done]):.4f} "
                  f"SSIM={np.mean([r['ssim'] for r in done]):.4f}", flush=True)

    summary = summarize(rows, lpips_metric is not None)
    summary.update({
        "checkpoint": str(args.checkpoint), "stage": model_args.stage,
        "epoch": epoch, "protocol": args.protocol,
        "quantized": not args.no_quantize,
        "seconds_per_frame_model": gpu_time / max(1, len(rows)),
        "seconds_total": time.time() - start_time,
    })
    write_outputs(out_dir, rows, summary)
    print_summary(summary)
    print(f"Per-frame results: {out_dir / 'per_frame.csv'} | summary: {out_dir / 'summary.json'}")
    return summary


def summarize(rows, with_lpips):
    metrics = ["psnr", "ssim"] + (["lpips"] if with_lpips else [])

    def mean_of(subset):
        return {m: float(np.mean([r[m] for r in subset])) for m in metrics}

    by_skip = defaultdict(list)
    for row in rows:
        by_skip[row["skip"]].append(row)
    summary = {"metrics": metrics, "skips": {}}
    for skip, subset in sorted(by_skip.items()):
        by_t, by_scene = defaultdict(list), defaultdict(list)
        for row in subset:
            by_t[row["t"]].append(row)
            by_scene[row["scene"]].append(row)
        scene_means = [mean_of(v) for v in by_scene.values()]
        summary["skips"][str(skip)] = {
            "frames": len(subset),
            "windows": len({(r["scene"], r["start"]) for r in subset}),
            "frame_mean": mean_of(subset),
            "scene_mean": {m: float(np.mean([s[m] for s in scene_means])) for m in metrics},
            "by_t": {str(t): dict(mean_of(v), frames=len(v)) for t, v in sorted(by_t.items())},
            "by_scene": {name: dict(mean_of(v), frames=len(v)) for name, v in sorted(by_scene.items())},
        }
    return summary


def write_outputs(out_dir, rows, summary):
    with open(out_dir / "per_frame.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def print_summary(summary):
    metrics = summary["metrics"]
    print("\n================ BS-ERGB results ================")
    print(f"stage={summary['stage']} epoch={summary['epoch']} quantized={summary['quantized']} "
          f"protocol={summary['protocol']}")
    header = f"{'skip':>4} {'frames':>7} " + " ".join(f"{m.upper():>8}" for m in metrics)
    print(header)
    for skip, info in summary["skips"].items():
        values = " ".join(f"{info['frame_mean'][m]:8.4f}" for m in metrics)
        print(f"{skip:>4} {info['frames']:>7} {values}")
        for t, info_t in info["by_t"].items():
            values_t = " ".join(f"{info_t[m]:8.4f}" for m in metrics)
            print(f"{'':>4} {'t=' + t:>7} {values_t}")
    print(f"model time per frame: {summary['seconds_per_frame_model']:.3f} s "
          f"(total {summary['seconds_total'] / 60:.1f} min)")


if __name__ == "__main__":
    evaluate(parse_args())
