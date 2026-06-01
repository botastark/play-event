import dv_processing as dv
import cv2 as cv
from datetime import timedelta

capture = dv.io.camera.open()

if not capture.isEventStreamAvailable():
    raise RuntimeError("This camera does not provide an event stream.")

resolution = capture.getEventResolution()

# Filter parameters
background_duration_ms = 1
decay_half_life_ms = 20
decay_subdivision = 10
decay_threshold = 1.5

# Initialize a background activity noise filter with 1-millisecond activity period
filter_background = dv.noise.BackgroundActivityNoiseFilter(
    resolution,
    backgroundActivityDuration=timedelta(milliseconds=background_duration_ms),
)

# Initialize a background activity noise filter with 10-millisecond half life decay, resolution subdivision
# factor of 4 and noise threshold of 1. Half life decay and noise threshold values controls the quality of
# filtering, while subdivision factor is used for resolution downsizing for internal event representation.
filter_decay = dv.noise.FastDecayNoiseFilter(
    resolution,
    halfLife=timedelta(milliseconds=decay_half_life_ms),
    subdivisionFactor=decay_subdivision,
    noiseThreshold=decay_threshold,
)

# Use a visualizer instance to preview the events
visualizer = dv.visualization.EventVisualizer(resolution)
slicer = dv.EventStreamSlicer()

cv.namedWindow("Event Preview", cv.WINDOW_NORMAL)

# Toggle for showing noise
show_noise = [False]  # Use list to allow modification in callback
exit_flag = [False]  # Use list to allow modification in callback

print("Event Camera Noise Filter Comparison")
print("=====================================")
print("Controls:")
print("  'n' - Toggle noise visualization")
print("  'q' or ESC - Quit")
print("  Close window - Quit")
print()


def process_events(events: dv.EventStore):
    # Pass events to the filter and generate filtered output
    filter_background.accept(events)
    filtered_b = filter_background.generateEvents()
    filter_decay.accept(events)
    filtered_d = filter_decay.generateEvents()

    # Generate images for original and filtered
    frame_original = visualizer.generateImage(events)
    frame_filtered_background = visualizer.generateImage(filtered_b)
    frame_filtered_decay = visualizer.generateImage(filtered_d)

    # Add text labels to filtered frames
    cv.putText(
        frame_original,
        "Original",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    cv.putText(
        frame_filtered_background,
        f"Background ({background_duration_ms}ms)",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    cv.putText(
        frame_filtered_decay,
        f"Fast Decay ({decay_half_life_ms}ms, {decay_subdivision}x, {decay_threshold})",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    if show_noise[0]:
        # Create event stores for noise (filtered out events)
        noise_b = dv.EventStore()
        noise_d = dv.EventStore()

        # Get indices of filtered events to identify noise
        filtered_b_coords = set((e.x(), e.y(), e.timestamp()) for e in filtered_b)
        filtered_d_coords = set((e.x(), e.y(), e.timestamp()) for e in filtered_d)

        for e in events:
            coord = (e.x(), e.y(), e.timestamp())
            if coord not in filtered_b_coords:
                noise_b.push_back(e)
            if coord not in filtered_d_coords:
                noise_d.push_back(e)

        # Generate noise images
        frame_noise_original = visualizer.generateImage(dv.EventStore())  # Empty
        frame_noise_background = visualizer.generateImage(noise_b)
        frame_noise_decay = visualizer.generateImage(noise_d)

        # Add text labels to noise frames
        cv.putText(
            frame_noise_original,
            "Original",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        cv.putText(
            frame_noise_background,
            f"BG Noise ({background_duration_ms}ms)",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        cv.putText(
            frame_noise_decay,
            f"FD Noise ({decay_half_life_ms}ms, {decay_subdivision}x, {decay_threshold})",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        # Create 2 rows: filtered top, noise bottom
        row_filtered = cv.hconcat(
            [frame_original, frame_filtered_background, frame_filtered_decay]
        )
        row_noise = cv.hconcat(
            [frame_noise_original, frame_noise_background, frame_noise_decay]
        )
        preview = cv.vconcat([row_filtered, row_noise])
    else:
        # Single row with filtered events only
        preview = cv.hconcat(
            [frame_original, frame_filtered_background, frame_filtered_decay]
        )

    cv.imshow("Event Preview", preview)

    # Check for key press to toggle noise
    key = cv.waitKey(1) & 0xFF
    if key == ord("n"):
        show_noise[0] = not show_noise[0]
        print(f"Noise visualization: {'ON' if show_noise[0] else 'OFF'}")
    elif key == ord("q") or key == 27:  # 'q' or ESC
        exit_flag[0] = True


slicer.doEveryTimeInterval(timedelta(milliseconds=33), process_events)

while capture.isRunning() and not exit_flag[0]:
    events = capture.getNextEventBatch()
    if events is not None:
        # Process events through slicer - check if exit was requested
        slicer.accept(events)

    # Check if window is still open
    try:
        if cv.getWindowProperty("Event Preview", cv.WND_PROP_VISIBLE) < 1:
            break
    except cv.error:
        break

cv.destroyAllWindows()
