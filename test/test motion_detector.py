from src import MotionDetector
from imutils.video import VideoStream
import datetime
import imutils
import time
import cv2

vs = VideoStream(src=0).start()
time.sleep(2.0)

md = MotionDetector(accumWeight=0.2)
total = 0

while True:
    #! WARNING about vs.read() function
    #! vs.read() will return fixed memory block
    #! hard copy returned value before doing anything
    originalFrame = vs.read()
    frame = originalFrame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    timestamp = datetime.datetime.now()
    cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    if total > 35:
        motion = md.detect(gray)

        if motion is not None:
            (thresh, (minX, minY, maxX, maxY)) = motion
            cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)

    md.update(gray)
    total += 1

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

# quit
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
