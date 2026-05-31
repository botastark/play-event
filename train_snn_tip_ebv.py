import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ebv_dataset import EventRecording, infer_sensor_size, parse_ebv_file


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_delta: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(membrane_delta)
        return (membrane_delta > 0).to(membrane_delta.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (membrane_delta,) = ctx.saved_tensors
        grad_window = (1.0 - membrane_delta.abs()).clamp(min=0.0)
        return grad_output * grad_window


class LIFCell(nn.Module):
    def __init__(self, beta: float = 0.95, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, current: torch.Tensor, membrane: torch.Tensor):
        membrane = self.beta * membrane + current
        spike = SurrogateSpike.apply(membrane - self.threshold)
        membrane = membrane - spike * self.threshold
        return spike, membrane


class TipTrackerSNN(nn.Module):
    def __init__(self, height: int, width: int, hidden_size: int):
        super().__init__()
        self.fc_in = nn.Linear(height * width, hidden_size)
        self.lif = LIFCell(beta=0.95, threshold=1.0)
        self.fc_out = nn.Linear(hidden_size, 2)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        # events shape: [batch, time, height, width]
        batch_size, time_steps, _, _ = events.shape
        membrane = events.new_zeros((batch_size, self.fc_in.out_features))
        readout = events.new_zeros((batch_size, 2))

        for t in range(time_steps):
            current = self.fc_in(events[:, t].reshape(batch_size, -1))
            spikes, membrane = self.lif(current, membrane)
            readout = readout + self.fc_out(spikes)

        return torch.sigmoid(readout / time_steps)


@dataclass
class BuildConfig:
    dataset_dir: Path
    include_pen3bg: bool
    time_steps: int
    window_steps: int
    stride_steps: int
    top_band: int
    target_method: str
    target_smooth_alpha: float


def build_recordings(cfg: BuildConfig):
    recordings = [
        EventRecording(path=cfg.dataset_dir / "pen1.EBV", label=0),
        EventRecording(path=cfg.dataset_dir / "pen2.EBV", label=0),
    ]

    if cfg.include_pen3bg:
        recordings.append(EventRecording(path=cfg.dataset_dir / "pen3BG.EBV", label=0))

    for rec in recordings:
        if not rec.path.exists():
            raise FileNotFoundError(f"Missing recording: {rec.path}")

    return recordings


def events_to_frames_and_tip_targets(
    events: np.ndarray,
    time_steps: int,
    height: int,
    width: int,
    top_band: int,
    target_method: str = "hybrid",
    target_smooth_alpha: float = 0.7,
):
    frames = np.zeros((time_steps, height, width), dtype=np.float32)
    targets = np.zeros((time_steps, 2), dtype=np.float32)

    if events.size == 0:
        targets[:, 0] = 0.5
        targets[:, 1] = 0.5
        return torch.from_numpy(frames), torch.from_numpy(targets)

    in_bounds = (
        (events[:, 0] >= 0)
        & (events[:, 0] < height)
        & (events[:, 1] >= 0)
        & (events[:, 1] < width)
    )
    events = events[in_bounds]

    if events.size == 0:
        targets[:, 0] = 0.5
        targets[:, 1] = 0.5
        return torch.from_numpy(frames), torch.from_numpy(targets)

    y = events[:, 0].astype(np.int64)
    x = events[:, 1].astype(np.int64)
    ts = events[:, 3]

    t_min = ts.min()
    t_max = ts.max()
    duration = max(t_max - t_min, 1.0)
    bin_idx = np.floor((ts - t_min) / duration * (time_steps - 1)).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, time_steps - 1)

    np.add.at(frames, (bin_idx, y, x), 1.0)
    frames = np.clip(frames, 0.0, 1.0)

    last_x = (width - 1) / 2.0
    last_y = (height - 1) / 2.0

    def _tip_from_top_band(x_t: np.ndarray, y_t: np.ndarray):
        min_y = int(y_t.min())
        band_mask = y_t <= (min_y + top_band)
        x_candidates = x_t[band_mask]
        if x_candidates.size == 0:
            return None
        return float(np.median(x_candidates)), float(min_y)

    def _tip_from_linefit(x_t: np.ndarray, y_t: np.ndarray):
        if x_t.size < 10:
            return None
        pts = np.column_stack([x_t, y_t]).astype(np.float32)
        vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        p1 = np.array([x0 + t.min() * vx, y0 + t.min() * vy], dtype=np.float32)
        p2 = np.array([x0 + t.max() * vx, y0 + t.max() * vy], dtype=np.float32)
        tip = p1 if p1[1] < p2[1] else p2
        return float(tip[0]), float(tip[1])

    def _tip_from_hough(x_t: np.ndarray, y_t: np.ndarray):
        if x_t.size < 10:
            return None
        img = np.zeros((height, width), dtype=np.uint8)
        img[y_t, x_t] = 255
        lines = cv.HoughLinesP(
            img,
            rho=1,
            theta=np.pi / 180,
            threshold=max(8, int(0.02 * x_t.size)),
            minLineLength=max(6, int(0.05 * min(height, width))),
            maxLineGap=6,
        )
        if lines is None:
            return None

        best = None
        best_y = float("inf")
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in line]
            if y1 < best_y:
                best = (x1, y1)
                best_y = y1
            if y2 < best_y:
                best = (x2, y2)
                best_y = y2
        return best

    for t in range(time_steps):
        idx = np.where(bin_idx == t)[0]
        if idx.size > 0:
            y_t = y[idx]
            x_t = x[idx]
            candidate_top = _tip_from_top_band(x_t, y_t)
            candidate_line = _tip_from_linefit(x_t, y_t)
            candidate_hough = _tip_from_hough(x_t, y_t)

            if target_method == "top":
                candidate = candidate_top
            elif target_method == "linefit":
                candidate = candidate_line or candidate_top
            elif target_method == "hough":
                candidate = candidate_hough or candidate_top
            else:
                # Hybrid model-based target: prefer Hough, then line fit, then top-band heuristic.
                candidate = candidate_hough or candidate_line or candidate_top

            if candidate is not None:
                c_x = float(np.clip(candidate[0], 0, width - 1))
                c_y = float(np.clip(candidate[1], 0, height - 1))
                alpha = float(np.clip(target_smooth_alpha, 0.0, 1.0))
                last_x = alpha * c_x + (1.0 - alpha) * last_x
                last_y = alpha * c_y + (1.0 - alpha) * last_y

        targets[t, 0] = last_x / max(width - 1, 1)
        targets[t, 1] = last_y / max(height - 1, 1)

    return torch.from_numpy(frames), torch.from_numpy(targets)


def split_windows_with_targets(
    frames: torch.Tensor,
    targets: torch.Tensor,
    window_steps: int,
    stride_steps: int,
):
    total_steps = int(frames.shape[0])
    windows = []
    window_targets = []

    center_offset = window_steps // 2

    for start in range(0, max(total_steps - window_steps + 1, 1), stride_steps):
        end = start + window_steps
        if end > total_steps:
            break
        windows.append(frames[start:end])
        window_targets.append(targets[start + center_offset])

    if not windows:
        padded = torch.zeros((window_steps, frames.shape[1], frames.shape[2]), dtype=frames.dtype)
        keep = min(total_steps, window_steps)
        padded[:keep] = frames[:keep]
        windows = [padded]
        window_targets = [targets[min(center_offset, total_steps - 1)]]

    x = torch.stack(windows)
    y = torch.stack(window_targets)
    return x, y


def build_dataset(cfg: BuildConfig):
    recordings = build_recordings(cfg)
    height, width = infer_sensor_size(recordings)

    all_x = []
    all_y = []

    for rec in recordings:
        events = parse_ebv_file(rec.path)
        frames, tip_targets = events_to_frames_and_tip_targets(
            events=events,
            time_steps=cfg.time_steps,
            height=height,
            width=width,
            top_band=cfg.top_band,
            target_method=cfg.target_method,
            target_smooth_alpha=cfg.target_smooth_alpha,
        )
        x_rec, y_rec = split_windows_with_targets(
            frames=frames,
            targets=tip_targets,
            window_steps=cfg.window_steps,
            stride_steps=cfg.stride_steps,
        )
        all_x.append(x_rec)
        all_y.append(y_rec)

    x = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0)

    perm = torch.randperm(x.shape[0])
    x = x[perm]
    y = y[perm]

    split = int(0.8 * x.shape[0])
    split = max(1, min(split, x.shape[0] - 1))

    return (x[:split], y[:split]), (x[split:], y[split:]), (height, width)


def mae_pixels(pred: torch.Tensor, target: torch.Tensor, width: int, height: int) -> float:
    dx = (pred[:, 0] - target[:, 0]).abs() * max(width - 1, 1)
    dy = (pred[:, 1] - target[:, 1]).abs() * max(height - 1, 1)
    return float((dx + dy).mean().item() / 2.0)


def train(args):
    if args.hidden_size >= 1000:
        raise ValueError(
            f"hidden_size must be < 1000 for this setup, got {args.hidden_size}"
        )

    device = torch.device(args.device)
    cfg = BuildConfig(
        dataset_dir=Path(args.dataset_dir),
        include_pen3bg=args.include_pen3bg,
        time_steps=args.time_steps,
        window_steps=args.window_steps,
        stride_steps=args.stride_steps,
        top_band=args.top_band,
        target_method=args.target_method,
        target_smooth_alpha=args.target_smooth_alpha,
    )

    (train_x, train_y), (test_x, test_y), (height, width) = build_dataset(cfg)

    print(f"sensor size inferred: {height}x{width}")
    print(f"train samples: {train_x.shape[0]} | test samples: {test_x.shape[0]}")
    print(f"hidden neurons: {args.hidden_size}")

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

    model = TipTrackerSNN(height=height, width=width, hidden_size=args.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_mae = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_mae += mae_pixels(pred.detach(), yb, width=width, height=height)

        model.eval()
        test_mae = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                test_mae += mae_pixels(pred, yb, width=width, height=height)

        train_loss /= max(1, len(train_loader))
        train_mae /= max(1, len(train_loader))
        test_mae /= max(1, len(test_loader))

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.5f} "
            f"train_mae_px={train_mae:.2f} "
            f"test_mae_px={test_mae:.2f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae_px": train_mae,
                "test_mae_px": test_mae,
            }
        )

    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "task": "tip_tracking",
            "model_state": model.state_dict(),
            "height": height,
            "width": width,
            "hidden_size": args.hidden_size,
            "time_steps": args.time_steps,
            "window_steps": args.window_steps,
            "stride_steps": args.stride_steps,
            "top_band": args.top_band,
            "target_method": args.target_method,
            "target_smooth_alpha": args.target_smooth_alpha,
            "history": history,
        }
        torch.save(checkpoint, save_path)
        print(f"saved model: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a basic SNN pen-tip tracker on EBV recordings")
    parser.add_argument("--dataset-dir", type=str, default="/home/bota/Desktop/SNN-Example")
    parser.add_argument("--include-pen3bg", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--time-steps", type=int, default=240)
    parser.add_argument("--window-steps", type=int, default=24)
    parser.add_argument("--stride-steps", type=int, default=8)
    parser.add_argument("--top-band", type=int, default=2)
    parser.add_argument(
        "--target-method",
        type=str,
        default="hybrid",
        choices=["top", "linefit", "hough", "hybrid"],
    )
    parser.add_argument("--target-smooth-alpha", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-model", type=str, default="models/snn_tip_ebv.pt")

    train(parser.parse_args())
