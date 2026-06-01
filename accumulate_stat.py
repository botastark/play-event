import cv2 as cv
import dv_processing as dv
from datetime import timedelta

capture = dv.io.camera.open()

if not capture.isEventStreamAvailable():
    raise RuntimeError("This camera does not provide an event stream.")

resolution = capture.getEventResolution()
print("Camera opened.")
print("Event resolution:", resolution)

accumulator = dv.Accumulator(resolution)
accumulator.setMinPotential(0.0)
accumulator.setMaxPotential(1.0)
accumulator.setNeutralPotential(0.5)
accumulator.setEventContribution(0.15)
accumulator.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
accumulator.setDecayParam(1e6)
accumulator.setIgnorePolarity(False)
accumulator.setSynchronousDecay(False)

slicer = dv.EventStreamSlicer()
cv.namedWindow("Accumulated Events", cv.WINDOW_NORMAL)


def on_slice(events: dv.EventStore):
    if events is None or events.size() == 0:
        return

    accumulator.accept(events)
    frame = accumulator.generateFrame()
    image = frame.image.copy()

    event_count = events.size()
    duration_ms = events.duration().total_seconds() * 1000.0
    rate_kevs = event_count / max(events.duration().total_seconds(), 1e-9) / 1000.0

    cv.putText(
        image,
        f"events: {event_count}",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.putText(
        image,
        f"slice: {duration_ms:.2f} ms",
        (20, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.putText(
        image,
        f"rate: {rate_kevs:.1f} kevs/s",
        (20, 90),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.imshow("Accumulated Events", image)
    cv.waitKey(1)


slicer.doEveryTimeInterval(timedelta(milliseconds=33), on_slice)

while capture.isRunning():
    events = capture.getNextEventBatch()
    if events is not None:
        slicer.accept(events)

    if cv.getWindowProperty("Accumulated Events", cv.WND_PROP_VISIBLE) < 1:
        break

cv.destroyAllWindows()
