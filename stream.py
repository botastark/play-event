import time
import cv2 as cv
import dv_processing as dv

capture = dv.io.camera.open()
resolution = capture.getEventResolution()

visualizer = dv.visualization.EventVisualizer(resolution)

cv.namedWindow("Events", cv.WINDOW_NORMAL)

while capture.isRunning():
    events = capture.getNextEventBatch()

    if events is not None:
        preview = visualizer.generateImage(events)
        cv.imshow("Events", preview)

    if cv.waitKey(1) == 27:
        break

    if events is None:
        time.sleep(0.001)