import argparse
import csv
from pathlib import Path

from ebv_dataset import EventRecording, parse_ebv_file
from train_snn_tip_ebv import build_recordings, events_to_frames_and_tip_targets


def export_labels_for_file(
    file_path: Path,
    out_csv: Path,
    time_steps: int,
    height: int,
    width: int,
    top_band: int,
    target_method: str,
    target_smooth_alpha: float,
):
    events = parse_ebv_file(file_path)
    _, targets = events_to_frames_and_tip_targets(
        events=events,
        time_steps=time_steps,
        height=height,
        width=width,
        top_band=top_band,
        target_method=target_method,
        target_smooth_alpha=target_smooth_alpha,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_bin",
                "x_norm",
                "y_norm",
                "x_px",
                "y_px",
                "method",
                "smooth_alpha",
            ],
        )
        writer.writeheader()

        for t in range(targets.shape[0]):
            x_norm = float(targets[t, 0].item())
            y_norm = float(targets[t, 1].item())
            x_px = int(round(x_norm * max(width - 1, 1)))
            y_px = int(round(y_norm * max(height - 1, 1)))
            writer.writerow(
                {
                    "time_bin": t,
                    "x_norm": f"{x_norm:.7f}",
                    "y_norm": f"{y_norm:.7f}",
                    "x_px": x_px,
                    "y_px": y_px,
                    "method": target_method,
                    "smooth_alpha": f"{target_smooth_alpha:.4f}",
                }
            )


def infer_sensor_size_from_events(recordings):
    max_y = 0
    max_x = 0
    for rec in recordings:
        ev = parse_ebv_file(rec.path)
        if ev.size == 0:
            continue
        max_y = max(max_y, int(ev[:, 0].max()))
        max_x = max(max_x, int(ev[:, 1].max()))
    return max_y + 1, max_x + 1


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-label CSVs for pen-tip tracking")
    parser.add_argument("--dataset-dir", type=str, default="/home/bota/Desktop/SNN-Example")
    parser.add_argument("--out-dir", type=str, default="labels")
    parser.add_argument("--include-pen3bg", action="store_true")
    parser.add_argument("--time-steps", type=int, default=240)
    parser.add_argument("--top-band", type=int, default=2)
    parser.add_argument(
        "--target-method",
        type=str,
        default="hybrid",
        choices=["top", "linefit", "hough", "hybrid"],
    )
    parser.add_argument("--target-smooth-alpha", type=float, default=0.7)
    parser.add_argument("--file", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        fp = Path(args.file)
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")
        recordings = [EventRecording(path=fp, label=0)]
    else:
        cfg_like = type(
            "Cfg",
            (),
            {
                "dataset_dir": Path(args.dataset_dir),
                "include_pen3bg": args.include_pen3bg,
            },
        )
        recordings = build_recordings(cfg_like)

    height, width = infer_sensor_size_from_events(recordings)
    print(f"sensor size inferred: {height}x{width}")

    for rec in recordings:
        out_csv = out_dir / f"{rec.path.stem}_tip_labels.csv"
        export_labels_for_file(
            file_path=rec.path,
            out_csv=out_csv,
            time_steps=args.time_steps,
            height=height,
            width=width,
            top_band=args.top_band,
            target_method=args.target_method,
            target_smooth_alpha=args.target_smooth_alpha,
        )
        print(f"[{rec.path.name}] labels: {out_csv}")


if __name__ == "__main__":
    main()
