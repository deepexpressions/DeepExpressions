from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from DeepExpressions import FaceDetector, Model
from .wrapper_utils import update_style, draw_bounding_box


class VideoDetector():
    """
    DeepExpressions wrapper for Facial Expressions Recognition in images and videos.
    
    Arguments:
        + model_name (str) = name of the model. Options available at: 
        + mode (str) = Input data format. It can be either an 'image' or 'video'
        + face_alignment (bool) = Perform Face Alignment process before feed the classifier
        + style (dict) = API output style. Options available at:

    Methods:

    Functions:
        + compute = compute facial expressions from image or video
    """

    def __init__(self, model_name, mode="image", face_alignment=False, style=None):
        """API constructor."""

        self.model = Model(model_name)
        self.face_detector = FaceDetector()

        assert mode in ["image", "video"], "{} id an invalid mode! Try 'image' or 'video'.".format(mode)
        self.mode = mode
        self.face_alignment = face_alignment
        self.style = update_style(style)


    def compute(self, image=None, cam_id=0, fps=10, skip_frame=False, filename=None, display=False, title="DeepExpressions API"):
        """
        Compute facial expressions from input_data.
        
        Arguments:
            + image = Image to compute facial expressions. It can be: np.ndarray, tf.Tensor or str(path to image) (only for mode='image')
            + cam_id (int) = Camera ID (only for mode='video')
            + fps (int) = Camera FPS (only for mode='video')
            + skip_frame (bool) = Process one in every two frames (only for mode='video')
            + filename (str) = Name to save computed image.
            + display (bool) = Display API output using cv2.imshow()
            + title (str) = OpenCV window's name if display=True
        """

        if self.mode == "video":
            self._compute_video(cam_id=cam_id, fps=fps, skip_frame=skip_frame, display=display, title=title)
        else:
            self._compute_image(image=image, filename=filename, display=display, title=title)


    def _compute_image(self, image, filename=None, display=False, title="DeepExpressions API"):
        """Compute emotions for images.
        
        Arguments:
            + image = Image to compute facial expressions. It can be: np.ndarray, tf.Tensor or str(path to image)
            + display (bool) = Display API output using cv2.imshow()
            + title (str) = OpenCV window's name if display=True
        """

        if isinstance(image, str):
            image = cv2.imread(image)
            
        # Compute faces in image
        bb_list = self.face_detector.compute(image)

        # Loop over computed faces
        images = []
        for i in range(len(bb_list)):
            x0, y0, x1, y1 = bb_list[i]
            # add square crop options here!
            images.append(image[y0:y1, x0:x1])

        # predict labels from all faces detected in the image
        predictions = self.model.predict(images, labeled_output=True)
        # draw bounding boxes and labels on the image
        draw_bounding_box(image, bb_list, predictions, self.style)

        # save image
        if filename is not None:
            cv2.imwrite(filename, image)


        if display:
            cv2.imshow(title, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image
        
    
    def _compute_video(self, cam_id, fps=10, skip_frame=False, display=False, title="DeepExpressions API"):
        """Compute emotions for videos.
        
        Arguments:
            + cam_id (int) = Camera ID (only for mode='video')
            + fps (int) = Camera FPS (only for mode='video')
            + skip_frame (bool) = Process one in every two frames (only for mode='video')
            + display (bool) = Display API output using cv2.imshow()
            + title (str) = OpenCV window's name if display=True
        """

        # cap = cv2.VideoCapture(cam_id)
        # self.frame = ...
        pass