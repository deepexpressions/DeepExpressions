from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
from numpy import array
from .face_detector_utils import maybe_download_model


class FaceDetector():
    """
    Face Detector based on OpenCV's dnn module. 
    It utilizes the Single Shot Detector (SSD) framework with a ResNet as the base network.

    Methods:
        + compute -- Compute bounding boxes from detected faces within an image using OpenCV dnn module.

    Attributes:
        + detector -- Caffe model to detect faces.
    """
    

    def __init__(self):
        """FaceDetector constructor."""

        # Download OpenCV model if it does not exists
        maybe_download_model()

        # Files path
        model_path = os.path.join(os.environ['HOME'], ".keras/opencv/face_detector/")
        model_caffe_path = os.path.join(model_path, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        proto_caffe_path = os.path.join(model_path, "deploy.prototxt")

        # Loading Caffe model to OpenCV's deep learning face detector
        print("[INFO] Loading face detector model...", end="", flush=True)
        self.detector = cv2.dnn.readNetFromCaffe(proto_caffe_path, model_caffe_path)
        print("Done.")


    def compute(self, image, size=(300,300), threshold=0.5):
        """
        Compute bounding boxes from detected faces within an image using OpenCV dnn module.

        Arguments:
            + image (np.ndarray) -- Image from which the faces will be detected.
            + size (tuple) -- New size to resize input image before detections.
            + threshold (float) -- Confidence level in detections.

        Output:
            + List of bounding boxes with shape [x0, y0, x1, y1].
        """

        # Construct input blob
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, size), 1.0, 
                                     size, (104.0, 177.0, 123.0))

        # Forward pass
        self.detector.setInput(blob)
        detections = self.detector.forward()

        # Track bounding boxes
        bb_list = []

        # Loop over detections:
        for i in range(0, detections.shape[2]):
            # Extract confidence probability
            confidence = detections[0,0,i,2]

            # Filter weak detections
            if confidence > threshold:
                box = detections[0, 0, i, 3:7] * array([w, h, w, h])
                bb_list.append(box.astype("int"))

        return bb_list


