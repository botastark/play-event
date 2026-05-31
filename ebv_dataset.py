from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass
class EventRecording:
    path: Path
    label: int


def parse_ebv_file(path: Path) -> np.ndarray:
    """Load EBV text data encoded as: y x p timestamp (one event per row)."""
    events = np.loadtxt(path, dtype=np.float64)
    if events.ndim == 1:
        events = events.reshape(1, 4)

    if events.shape[1] != 4:
        raise ValueError(f"Expected 4 columns in {path}, got shape {events.shape}")

    return events


def infer_sensor_size(recordings: Iterable[EventRecording]) -> tuple[int, int]:
    max_y = 0
    max_x = 0
    for rec in recordings:
        events = parse_ebv_file(rec.path)
        max_y = max(max_y, int(events[:, 0].max()))
        max_x = max(max_x, int(events[:, 1].max()))

    return max_y + 1, max_x + 1


def _clip_events_to_sensor(events: np.ndarray, height: int, width: int) -> np.ndarray:
    in_bounds = (
        (events[:, 0] >= 0)
        & (events[:, 0] < height)
        & (events[:, 1] >= 0)
        & (events[:, 1] < width)
    )
    return events[in_bounds]


def events_to_frames(
    events: np.ndarray,
    time_steps: int,
    height: int,
    width: int,
    polarity_mode: str = "binary",
) -> torch.Tensor:
    """Convert asynchronous events into a dense tensor [T, H, W]."""
    if events.size == 0:
        return torch.zeros((time_steps, height, width), dtype=torch.float32)

    events = _clip_events_to_sensor(events, height=height, width=width)
    if events.size == 0:
        return torch.zeros((time_steps, height, width), dtype=torch.float32)

    ts = events[:, 3]
    t_min = ts.min()
    t_max = ts.max()
    duration = max(t_max - t_min, 1.0)

    # Map timestamps to discrete bins [0, T-1].
    bin_idx = np.floor((ts - t_min) / duration * (time_steps - 1)).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, time_steps - 1)

    y = events[:, 0].astype(np.int64)
    x = events[:, 1].astype(np.int64)

    if polarity_mode == "signed":
        p = np.where(events[:, 2] > 0, 1.0, -1.0)
    elif polarity_mode == "count":
        p = np.ones_like(events[:, 2], dtype=np.float64)
    else:
        p = np.ones_like(events[:, 2], dtype=np.float64)

    frames = np.zeros((time_steps, height, width), dtype=np.float32)
    np.add.at(frames, (bin_idx, y, x), p.astype(np.float32))

    if polarity_mode in {"binary", "count"}:
        frames = np.clip(frames, 0.0, 1.0)
    else:
        max_abs = np.maximum(np.abs(frames).max(), 1.0)
        frames = frames / max_abs

    return torch.from_numpy(frames)


def split_windows(
    frames: torch.Tensor,
    label: int,
    window_steps: int,
    stride_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a long sequence [T, H, W] into overlapping windows for training."""
    total_steps = frames.shape[0]
    windows = []

    for start in range(0, max(total_steps - window_steps + 1, 1), stride_steps):
        end = start + window_steps
        if end > total_steps:
            break
        windows.append(frames[start:end])

    if not windows:
        # Pad short sequences to at least one sample.
        padded = torch.zeros((window_steps, frames.shape[1], frames.shape[2]), dtype=frames.dtype)
        keep = min(total_steps, window_steps)
        padded[:keep] = frames[:keep]
        windows = [padded]

    x = torch.stack(windows)
    y = torch.full((x.shape[0],), int(label), dtype=torch.long)
    return x, y
