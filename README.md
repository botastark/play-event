# play-event

Event camera utilities and early SNN experimentation.

## Train SNN On Recorded EBV Files

You can train directly on your recorded files in:

- `/home/bota/Desktop/SNN-Example/pen1.EBV`
- `/home/bota/Desktop/SNN-Example/pen2.EBV`
- `/home/bota/Desktop/SNN-Example/pen3BG.EBV` (optional harder class)

Each row is parsed as: `<y> <x> <p> <timestamp>`.

## Use Java Viewer

The viewer can be launched on Linux with:
The script mirrors the original Windows bat command and uses:

```bash
 /usr/lib/jvm/java-25-openjdk-amd64/bin/java   --enable-native-access=ALL-UNNAMED   -jar /home/bota/Desktop/SNN-Example/viewPSGX320.jar
# java --enable-native-access=ALL-UNNAMED -jar /home/bota/Desktop/SNN-Example/viewPSGX320.jar
```

### Java version requirement

`viewPSGX320.jar` was compiled for Java class version `66` (Java 22).
If you see `UnsupportedClassVersionError`, install a newer runtime:

```bash
sudo apt install openjdk-25-jre
```

## Basic SNN Tip Tracking

For pen-tip tracking, train a regression model that predicts tip coordinates `(x, y)`.

### 1) Generate model-based pseudo labels (good-enough GT)

This exports one CSV per recording with `(x,y)` tip targets per time bin.

```bash
/home/bota/miniconda3/envs/eventcam/bin/python generate_tip_labels.py \
	--dataset-dir /home/bota/Desktop/SNN-Example \
	--out-dir /home/bota/repos/play-event/labels \
	--target-method hybrid \
	--target-smooth-alpha 0.7
```

Try alternatives for pseudo-GT quality:

- `--target-method top`
- `--target-method linefit`
- `--target-method hough`
- `--target-method hybrid` (recommended default)

### 2) Train simple SNN tip tracker

```bash
/home/bota/miniconda3/envs/eventcam/bin/python train_snn_tip_ebv.py \
	--dataset-dir /home/bota/Desktop/SNN-Example \
	--target-method hybrid \
	--target-smooth-alpha 0.7 \
	--save-model /home/bota/repos/play-event/models/snn_tip_ebv.pt \
	--epochs 20
```

### 3) Visualize tracking results

```bash
/home/bota/miniconda3/envs/eventcam/bin/python inspect_snn_tip_ebv.py \
	--model /home/bota/repos/play-event/models/snn_tip_ebv.pt \
	--dataset-dir /home/bota/Desktop/SNN-Example \
	--target-method hybrid \
	--out-dir /home/bota/repos/play-event/viz_out
```

Outputs:

- `*_tip_track.mp4`: overlay video (red = predicted tip, green = pseudo-target)
- `*_tip_track.csv`: per-window predicted and target coordinates with pixel error

Notes:

- Useful knobs: `--time-steps`, `--window-steps`, `--stride-steps`, `--hidden-size`, `--top-band`.
- Ground-truth generation methods: `--target-method top|linefit|hough|hybrid`.
- Temporal smoothing: `--target-smooth-alpha` in [0, 1].
- Data loading and conversion helpers are in `ebv_dataset.py`.

Compare pseudo-ground-truth strategies quickly:

```bash
/home/bota/miniconda3/envs/eventcam/bin/python train_snn_tip_ebv.py \
	--dataset-dir /home/bota/Desktop/SNN-Example \
	--target-method hough \
	--target-smooth-alpha 0.8 \
	--save-model /home/bota/repos/play-event/models/snn_tip_ebv_hough.pt \
	--epochs 5
```

```bash
/home/bota/miniconda3/envs/eventcam/bin/python inspect_snn_tip_ebv.py \
	--model /home/bota/repos/play-event/models/snn_tip_ebv_hough.pt \
	--dataset-dir /home/bota/Desktop/SNN-Example \
	--target-method hough \
	--out-dir /home/bota/repos/play-event/viz_out
```
