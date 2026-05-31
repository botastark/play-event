import argparse
import csv
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from ebv_dataset import EventRecording, parse_ebv_file
from train_snn_tip_ebv import TipTrackerSNN, build_recordings, events_to_frames_and_tip_targets, split_windows_with_targets


def frame_to_rgb(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    gray = (frame * 255.0).clip(0, 255).astype(np.uint8)
    color = cv.applyColorMap(gray, cv.COLORMAP_BONE)
    return cv.resize(color, (out_w, out_h), interpolation=cv.INTER_NEAREST)


def px_from_norm(xy: np.ndarray, width: int, height: int):
    x = int(round(float(xy[0]) * max(width - 1, 1)))
    y = int(round(float(xy[1]) * max(height - 1, 1)))
    return x, y


def inspect_recording(
    model: TipTrackerSNN,
    device: torch.device,
    rec: EventRecording,
    checkpoint: dict,
    out_dir: Path,
    fps: int,
    scale: int,
    target_method: str,
    target_smooth_alpha: float,
):
    events = parse_ebv_file(rec.path)
    height = int(checkpoint["height"])
    width = int(checkpoint["width"])

    frames, targets = events_to_frames_and_tip_targets(
        events=events,
        time_steps=int(checkpoint["time_steps"]),
        height=height,
        width=width,
        top_band=int(checkpoint.get("top_band", 2)),
        target_method=target_method,
        target_smooth_alpha=target_smooth_alpha,
    )

    windows, window_targets = split_windows_with_targets(
        frames=frames,
        targets=targets,
        window_steps=int(checkpoint["window_steps"]),
        stride_steps=int(checkpoint["stride_steps"]),
    )

    model.eval()
    with torch.no_grad():
        pred = model(windows.to(device)).cpu().numpy()

    tgt = window_targets.numpy()
    out_w = width * scale
    out_h = height * scale

    video_path = out_dir / f"{rec.path.stem}_tip_track.mp4"
    csv_path = out_dir / f"{rec.path.stem}_tip_track.csv"

    writer = cv.VideoWriter(
        str(video_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video {video_path}")

    center = int(checkpoint["window_steps"]) // 2

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "window_idx",
            "pred_x",
            "pred_y",
            "target_x",
            "target_y",
            "error_px",
        ]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        errors = []

        for idx in range(windows.shape[0]):
            frame = windows[idx, center].numpy()
            img = frame_to_rgb(frame, out_w=out_w, out_h=out_h)

            pred_x, pred_y = px_from_norm(pred[idx], width=width, height=height)
            tgt_x, tgt_y = px_from_norm(tgt[idx], width=width, height=height)

            err = float(np.hypot(pred_x - tgt_x, pred_y - tgt_y))
            errors.append(err)

            sx = scale
            cv.circle(img, (pred_x * sx, pred_y * sx), 5, (0, 0, 255), 2)
            cv.circle(img, (tgt_x * sx, tgt_y * sx), 5, (0, 255, 0), 2)

            cv.putText(
                img,
                f"{rec.path.stem} | window {idx + 1}/{windows.shape[0]}",
                (10, 26),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                img,
                f"err_px: {err:.1f}",
                (10, 52),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv.LINE_AA,
            )

            writer.write(img)

            csv_writer.writerow(
                {
                    "window_idx": idx,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "target_x": tgt_x,
                    "target_y": tgt_y,
                    "error_px": f"{err:.4f}",
                }
            )

    writer.release()

    mean_err = float(np.mean(errors)) if errors else 0.0
    print(f"[{rec.path.name}] mean pixel error: {mean_err:.2f}")
    print(f"[{rec.path.name}] video: {video_path}")
    print(f"[{rec.path.name}] csv: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect SNN pen-tip tracking on EBV recordings")
    parser.add_argument("--model", type=str, default="models/snn_tip_ebv.pt")
    parser.add_argument("--dataset-dir", type=str, default="/home/bota/Desktop/SNN-Example")
    parser.add_argument("--include-pen3bg", action="store_true")
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="viz_out")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--target-method",
        type=str,
        default="",
        choices=["", "top", "linefit", "hough", "hybrid"],
    )
    parser.add_argument("--target-smooth-alpha", type=float, default=-1.0)
    args = parser.parse_args()

    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    if checkpoint.get("task") != "tip_tracking":
        raise ValueError("Checkpoint task is not tip_tracking. Train using train_snn_tip_ebv.py")

    model = TipTrackerSNN(
        height=int(checkpoint["height"]),
        width=int(checkpoint["width"]),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model = model.to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_method = args.target_method or str(checkpoint.get("target_method", "hybrid"))
    target_smooth_alpha = (
        args.target_smooth_alpha
        if args.target_smooth_alpha >= 0.0
        else float(checkpoint.get("target_smooth_alpha", 0.7))
    )

    if args.file:
        fp = Path(args.file)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")
        recordings = [EventRecording(path=fp, label=0)]
    else:
        cfg_like = type("Cfg", (), {
            "dataset_dir": Path(args.dataset_dir),
            "include_pen3bg": args.include_pen3bg,
        })
        recordings = build_recordings(cfg_like)

    for rec in recordings:
        inspect_recording(
            model=model,
            device=device,
            rec=rec,
            checkpoint=checkpoint,
            out_dir=out_dir,
            fps=args.fps,
            scale=args.scale,
            target_method=target_method,
            target_smooth_alpha=target_smooth_alpha,
        )


if __name__ == "__main__":
    main()
