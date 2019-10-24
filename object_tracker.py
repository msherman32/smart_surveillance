# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker, Http_processor
from imutils.video import VideoStream, FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from queue import Queue
from threading import Thread
# import basehash as bhash

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
       help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
       help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,
       help="path to Video file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
       help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
q = Queue()
ct = CentroidTracker(q) # ct is now a thread
ht = Http_processor(q)
ct.start()
ht.start()
# ct.join()
# ht.join()
# q.join()

(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
# https://github.com/C-Aniruddh/realtime_object_recognition
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# Set video to load
# for more videos, look at Campus Videos on Google Drive
videoPath = args["video"]

# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

time.sleep(2.0)

fps = FPS().start()

frame_id = 0

# hash_fn = bhash.base36()  # you can initialize a 36, 52, 56, 58, 62 and 94 base fn
# hash_value = hash_fn.hash((2,3)) # returns 'M8YZRZ'
# print(hash_value)
# unhashed = hash_fn.unhash('M8YZRZ') # returns 1

# loop over the frames from the video stream
while True:
       # read the next frame from the video stream and resize it
       frame = cap.read()
       # Read first frame
       success, frame = cap.read()
       # quit if unable to read the video file
       if not success:
        print('Failed to read video')
        exit(1)
       frame = imutils.resize(frame, width=300) #400

       # if the frame dimensions are None, grab them
       if W is None or H is None:
               (H, W) = frame.shape[:2]

       # construct a blob from the frame, pass it through the network,
       # obtain our output predictions, and initialize the list of
       # bounding box rectangles
#        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
       blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
       net.setInput(blob)
       detections = net.forward()
       rects = []

       # loop over the detections
       for i in range(0, detections.shape[2]):
               # filter out weak detections by ensuring the predicted
               # probability is greater than a minimum threshold
               if detections[0, 0, i, 2] > args["confidence"]:
                       # compute the (x, y)-coordinates of the bounding box for
                       # the object, then update the bounding box rectangles list
                       box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                       rects.append(box.astype("int"))
                #        print(detections[0,0,i,0:7])

                       # draw a bounding box surrounding the object so we can
                       # visualize it
                       (startX, startY, endX, endY) = box.astype("int")
                       cv2.rectangle(frame, (startX, startY), (endX, endY),
                               (0, 255, 0), 2)

       # update our centroid tracker using the computed set of bounding
       # box rectangles
       objects = ct.update(rects, frame_id, frame)

       # loop over the tracked objects
       for (objectID, centroid) in objects.items():
               # draw both the ID of the object and the centroid of the
               # object on the output frame
               text = "ID {}".format(objectID)
               cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
               cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
               # if ct.is_new(objectID):
            # cv2.imwrite("frame_captures/frame%d.jpg" % frame_id, frame) 

       # update the FPS counter
       fps.update()
       frame_id += 1

       # show the output frame
       cv2.imshow("Frame", frame)
       key = cv2.waitKey(1) & 0xFF

       # if the `q` key was pressed, break from the loop
       if key == ord("q"):
               break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# print(ct.id_to_frame)
# todo: figure out some of this stuff:
# print("BEGIN:")
# for k, v in ct.begin.items():
#     print (k, '-->', v)
# print("END:")
# for k, v in ct.end.items():
#     print (k, '-->', v)
# net.dumpToFile('static/dump.txt')
# print(net.dump())

# do a bit of cleanup
cv2.destroyAllWindows()
# vc.stop()