import cv2 as cv
import dv_processing as dv
from datetime import timedelta

capture = dv.io.camera.open()

if not capture.isEventStreamAvailable():
    raise RuntimeError("This camera does not provide an event stream.")

resolution = capture.getEventResolution()
print("Camera opened.")
print("Event resolution:", resolution)

visualizer = dv.visualization.EventVisualizer(resolution)
slicer = dv.EventStreamSlicer()

cv.namedWindow("Event Preview", cv.WINDOW_NORMAL)

def on_slice(events: dv.EventStore):
    frame = visualizer.generateImage(events)
    cv.imshow("Event Preview", frame)
    cv.waitKey(1)

slicer.doEveryTimeInterval(timedelta(milliseconds=33), on_slice)

while capture.isRunning():
    events = capture.getNextEventBatch()
    if events is not None:
        slicer.accept(events)

    if cv.getWindowProperty("Event Preview", cv.WND_PROP_VISIBLE) < 1:
        break

cv.destroyAllWindows()
