import dv_processing as dv

cameras = dv.io.camera.discover()
print("Found cameras:", cameras)

capture = dv.io.camera.open()
print("Opened:", capture.getCameraName())

if capture.isEventStreamAvailable():
    resolution = capture.getEventResolution()
    print("Event resolution:", resolution)

if capture.isFrameStreamAvailable():
    print("Frame resolution:", capture.getFrameResolution())

if capture.isImuStreamAvailable():
    print("IMU available")