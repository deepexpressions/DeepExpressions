from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from math import floor


def _extract_square_roi_by_min(box):
    # Calculate box shape and differences
    xmin, ymin, xmax, ymax = box
    box_height = ymax - ymin
    box_width = xmax - xmin
    diff = floor(abs(box_height - box_width) / 2)
    
    # Make the width and height even
    xmax = xmax if box_width % 2 == 0 else xmax - 1
    ymax = ymax if box_height % 2 == 0 else ymax - 1
    
    # Decreases box height
    if box_height > box_width:
        ymin += diff
        ymax -= diff
        
    # Decreases box width
    elif box_height < box_width:    
        xmin += diff
        xmax -= diff
        
    return [xmin, ymin, xmax, ymax]


def _extract_square_roi_by_max(box, height, width):
    # Calculate box shape and differences
    xmin, ymin, xmax, ymax = box    
    box_height = ymax - ymin
    box_width = xmax - xmin
    diff = floor(abs(box_height - box_width) / 2)
    
    # Make the width and height even
    xmax = xmax if box_width % 2 == 0 else xmax - 1
    ymax = ymax if box_height % 2 == 0 else ymax - 1

    # Increases box width
    if box_height > box_width:
        xmin = xmin - diff if (xmin - diff) > 0 else 0
        xmax = xmax + diff if (xmin + diff) < width else width - 1
        
    # Increases box height
    elif box_height < box_width:    
        ymin = ymin - diff if (ymin - diff) > 0 else 0
        ymax = ymax + diff if (ymax + diff) < height else height - 1
        
    # If box is not square, it means the box exceed the image bordes
    # Therefore, the box will be recalculated based on the new values
    # using the 'min' method
    if (ymax - ymin) != (xmax - xmin):
        xmin, ymin, xmax, ymax = _extract_square_roi_by_min([xmin, ymin, xmax, ymax])  

    return [xmin, ymin, xmax, ymax]


def extract_rois(image, boxes, method="min", square_box=False):
    """
    Extracts crops from image based on boxes of detection.

    Arguments:
        + image (np.ndarray, tf.Tensor) -- Image to be cropped
        + boxes (list) -- Coordinates from the boxes. [[xmin, ymin, xmax, ymax], ...].
        + method (str) -- Method used to get square boxes.
        + square_box (bool) -- Square the boxes before crop the image.

    Output:
        + list of cropped images based on input image and detections boxes
    """

    # Get image shape
    h, w = image.shape[:2]
    rois = []
    
    # Loop over boxes
    for box in boxes:
        # If box must be square
        if square_box:
            if method == "min":
                box = _extract_square_roi_by_min(box)
                
            elif method == "max":
                box = _extract_square_roi_by_max(box, h, w)
        
        # Crop image and append it to list
        xmin, ymin, xmax, ymax = box
        roi = image[ymin:ymax, xmin:xmax]
        rois.append(roi)
                
    return rois