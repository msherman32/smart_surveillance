# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
from threading import Thread
from queue import Queue
import numpy as np
import cv2 as cv
import requests
import time
import base64


class Packet():
    def __init__(self, payload, time):
        self.payload = payload
        self.time = time

    def getPayload(self):
        return self.payload

    def getTime(self):
        return self.time

class Http_processor(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self): 
        #check for a condition here?
        url = 'http://localhost:5000/submit/'
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        while (True):
            element = self.queue.get()
            payload = element[0]
            # handle converting to json serializable here:
            retval, buf = cv.imencode('.jpg', payload["image"])
            jpg_as_text = base64.b64encode(buf).decode('ascii')
            payload["image"] = jpg_as_text
            start = time.time()
            requests.post(url, json=payload, headers=headers)
            end = time.time()
            delay = (time.time() - element[1]) # * 1000 in milliseconds
            post_delay = end - start
            self.queue.task_done()
            print("Delay for %d POST is %f seconds" %(payload["object_id"], post_delay))
            print("Delay for %d is %f seconds" %(payload["object_id"], delay))

class CentroidTracker(Thread):
    def __init__(self, queue, maxDisappeared=1): # 50
        Thread.__init__(self)
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.queue = queue

        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.id_to_frame = OrderedDict()
        self.begin = OrderedDict()
        self.end = OrderedDict()
        self.old_ids = tuple()
        self.frame = None

        self.old_object_id = 0

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, frame_id, bbox):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.id_to_frame[self.nextObjectID] = [frame_id] # begin a new list of frame numbers 
        self.disappeared[self.nextObjectID] = 0
        startY = bbox[1]
        endY = bbox[3]
        startX = bbox[0]
        endX = bbox[2]
        cropped = self.frame[startY:endY, startX:endX]
        # print(type(cropped))
        # take away writing to file
        # cv.imwrite("static/object%d.jpg" % self.nextObjectID, cropped)  #todo: must be static/image for flask...
     
        # retval, buf = cv.imencode('.jpg', cropped)
        # jpg_as_text = base64.b64encode(buf).decode('ascii')
        # img_str = cv.imencode('.jpg', cropped)[1].toString()
        # Put this into http_processor?
        # image_string = ""
        # with open("static/object%d.jpg" % self.nextObjectID, "rb") as imageFile:
        #     # imageFile.write(img_str.tobytes())
        #     print(imageFile)
            # image_string = base64.b64encode(imageFile.read()).decode('ascii') # must decode to ascii so that it is JSON serializable
        start_time = time.time()
        payload = {"object_id":self.nextObjectID, "image":cropped}
        element = (payload, start_time)
        self.queue.put_nowait(element)
   
        # add before and after time?
        # do this multithreaded (to get these operations out of Critical Path)
        # use quicker encoding for images
        # structure of multithread with mutex lock and condition variables:
        # 1. get/acquire mutex
        # 2. check queue for !empty
        # 3. if !empty, send htttp post, release mutex
        # 4. else, wait on condition variable
        # if woken up, return to step 1

        # if res.ok:
        #     print(res)
        # make a post request to /submit (which will be an endpoint replaced with whisk) and then display which displays everything that has been submitted
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    # def cmpTuple(self, t1, t2):
      # 	return sorted(t1) == sorted(t2)

    def is_new(self, object_ID):
        return (object_ID > self.old_object_id) and (object_ID <= self.nextObjectID)

    def update(self, rects, frame_id, frame):
        # check to see if the list of input bounding box rectangles
        # is empty
        self.frame = frame
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        detected_objects = []
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], frame_id, rects[i])
                detected_objects.append(self.nextObjectID)
            
            # old_ids = tuple(detected_objects)
            # begin[old_ids] = frame_id
            # end[old_ids] = frame_id			
            # ????	

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            curr_ids = tuple(objectIDs)
            
            if (curr_ids not in self.begin) and (curr_ids not in self.end):
                self.begin[curr_ids] = frame_id
                self.end[curr_ids] = frame_id
            else:
                if sorted(self.old_ids) != sorted(curr_ids):
                    self.end[self.old_ids] = frame_id
                else:
                    self.end[curr_ids] = frame_id

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.id_to_frame[objectID].append(frame_id)
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], frame_id, rects[col]) # indexing into rects with col gets the unregistered bboxes

        old_ids = tuple(self.objects.keys())
        # self.old_object_id = self.nextObjectID #update old to current id

        # return the set of trackable objects
        return self.objects