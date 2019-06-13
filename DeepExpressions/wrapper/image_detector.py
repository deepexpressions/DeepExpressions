from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from DeepExpressions import FaceDetector, Model
from .wrapper_utils import update_style, draw_bounding_box


class ImageDetector():
    """
    DeepExpressions wrapper in order to facilitate Facial Expressions Recognitions in images.

    Arguments:
        + model_name (str) -- Name of the model. Options available at : http://github.com/deepexpressions/models.

    Methods:
        + compute -- Detect Facial Expressions within the image.
    """

    def __init__(self, model_name="ce-xception-512-256"):
        """ImageDetector constructor."""

        # Load models
        self.model = Model(model_name)
        self.face_detector = FaceDetector()


    def compute(self, images=None, size=(300,300), face_detection_threshold=0.5, 
                display_package=None, display_title=None, figsize=None, style=None, filename=None):
        """
        Compute Facial Expression within the input images.

        Arguments:
            + images (str) -- Image to compute expressions. It can be a string (path to image), an array or a list of arrays.
            + size (tuple) -- New size to resize input image before detections.
            + face_detection_threshold (float) -- Confidence level in detections.
            + display_package (str) -- Name of the package used to display the output. It can be either matplotlib (mlp, plt) or opencv (cv2).
            + display_title (str) -- Image title.
            + figsize (tuple) -- Figure size with matplotlib.
            + style (dict) -- Display settings about the bounding box and text in the output image.
            + filename (str) -- Filename to save the output. If filename is None, the image will not be saved.

        Output:
            + Images with bounding boxes with labels written on it expressing the facial expressions detected.
        """

        assert display_package in [None, "matplotlib", "mlp", "plt", "opencv", "cv2"], """
            Package display must be 'matplotlib' or 'opencv'!"""

        # Load output style
        self.style = update_style(style)

        # preprocess input data
        if isinstance(images, str):
            images = [cv2.imread(images)]

        elif isinstance(images, np.ndarray):
            images = [images]
        
        if filename is not None:
            # images counter
            images_count, ic = len(images), 0
            # editing filename
            filename_splited = filename.split(".")
            filename_main = "".join(filename_splited[:-1])
            filename_ext = "." + filename_splited[-1]
        
        # Loop over images
        for image in images:
            # Compute faces in image
            bb_list = self.face_detector.compute(image, size=size, threshold=face_detection_threshold)
            # init the list with images from detected faces
            facial_images = []

            # Loop over detections
            for i in range(len(bb_list)):
                x0, y0, x1, y1 = bb_list[i]
                facial_images.append(image[y0:y1, x0:x1])

            # predict labels from all faces detected in the image
            predictions = self.model.predict(facial_images, labeled_output=True)
            print(predictions)
            # draw bounding boxes and labels on the image
            draw_bounding_box(image, bb_list, predictions, self.style)

            # display image with OpenCV
            if display_package in ["opencv", "cv2"]:
                cv2.imshow(display_title, image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # display image with matplotlib.pyplot
            elif display_package in ["matplotlib", "plt", "pyplot"]:
                if figsize is not None:
                    plt.figure(figsize=figsize)
                plt_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                plt.imshow(plt_image)
                plt.grid(False)
                plt.axis("off")
                plt.title(display_title)
                plt.show()

            # saving image
            if filename is not None:
                if images_count == 1:
                    cv2.imwrite(filename, image)
                else:
                    cv2.imwrite(filename_main+ "_{}".format(ic) + filename_ext, image)
                    ic += 1