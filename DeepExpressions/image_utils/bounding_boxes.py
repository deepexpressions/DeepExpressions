import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


STANDARD_COLORS = list(ImageColor.colormap)


def draw_bounding_box_from_array(image, box, scores=None, labels=None, colors="red", font_color="black", 
                                 thickness=2, font_size=24, use_normalized_coordinates=True):
    """
    Adds a bounding box to an image (numpy array).
    
    Arguments:
        + image (np.array) -- Image to draw the bounding box with shape [height, width, 3].
        + box (list) -- Coordinates from the box. [xmin, ymin, xmax, ymax].
        + scores (list, np.array) -- Output scores from the FERModel.
        + labels (list, dict) -- Labels from each score.
        + colors (str, dict) -- Colors to draw bounding box. Default is 'red'. 
            A dict can be passed to draw boxes of different colors for each label.
        + font_color -- Color to draw the text.
        + thickness (int) -- Line thickness. Default value is 4.
        + font_size (int) -- Font size to draw the text.
        + use_normalized_coordinates (bool) -- If True (default), treat coordinates
            [xmin, ymin, xmax, ymax] as relative to the image.  Otherwise treat
            coordinates as absolute.
    """

    pil_image = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_bounding_box(pil_image, box, scores, labels, colors, font_color, 
                      thickness, font_size, use_normalized_coordinates)
    np.copyto(image, np.array(pil_image))


def draw_bounding_box(image, box, scores=None, labels=None, colors="red", font_color="black", 
                      thickness=2, font_size=24, use_normalized_coordinates=True):
    """
    Adds a bounding box to an image.
    
    Arguments:
        + image (PIL.Image) -- A PIL.Image object.
        + box (list) -- Coordinates from the box. [xmin, ymin, xmax, ymax].
        + scores (list, np.array) -- Output scores from the FERModel.
        + labels (list, dict) -- Labels from each score.
        + colors (str, dict) -- Colors to draw bounding box. Default is 'red'. 
            A dict can be passed to draw boxes of different colors for each label.
        + font_color -- Color to draw the text.
        + thickness (int) -- Line thickness. Default value is 4.
        + font_size (int) -- Font size to draw the text.
        + use_normalized_coordinates (bool) -- If True (default), treat coordinates
            [xmin, ymin, xmax, ymax] as relative to the image.  Otherwise treat
            coordinates as absolute.
    """

    if isinstance(colors, dict):
        assert scores is not None, "If scores is None, colors must be a color name (str) rather than a dictionary."
    else:
        color = colors

    # Init drawing tool
    draw = ImageDraw.Draw(image)

    # labels list to dict
    if isinstance(labels, list):
        labels = {i:l for i, l in enumerate(labels)} 

    # Retrieve info from scores
    if scores is not None:
        # most likely label
        max_score = np.argmax(scores)
        # Select color and text based on scores
        prob = scores[max_score]
        if labels is not None:
            text = labels[max_score] + ": {:.2f}%".format(prob*100)
        else:
            text = str(max_score) + ": {:.2f}%".format(prob*100)
        color = colors if not isinstance(colors, dict) else colors[max_score]

    else:
        text = None

    # Unpack image shape and box coordinates
    w, h = image.size
    # ymin, xmin, ymax, xmax = box
    xmin, ymin, xmax, ymax = box

    # Get box coordinates
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    # Draw box
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

    # Write label
    if text is not None:
        draw_text(image, text, left, right, top, bottom, color, font_color, font_size)


def draw_bounding_boxes_from_array(image, boxes, scores=None, labels=None, colors="red", font_color="black",
                                   thickness=2, font_size=24, use_normalized_coordinates=True):
    """
    Adds multiples bounding boxes to an image (numpy array).
    
    Arguments:
        + image (np.array) -- Image to draw the bounding box with shape [height, width, 3].
        + boxes (list) -- Coordinates from the boxes. [[xmin, ymin, xmax, ymax], ...].
        + scores (list, np.array) -- Output scores from the FERModel.
        + labels (list, dict) -- Labels from each score.
        + colors (str, dict) -- Colors to draw bounding box. Default is 'red'. 
            A dict can be passed to draw boxes of different colors for each label.
        + font_color -- Color to draw the text.
        + thickness (int) -- Line thickness. Default value is 4.
        + font_size (int) -- Font size to draw the text.
        + use_normalized_coordinates (bool) -- If True (default), treat coordinates
            [[xmin, ymin, xmax, ymax], ...] as relative to the image.  Otherwise treat
            coordinates as absolute.
    """

    pil_image = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_bounding_boxes(pil_image, boxes, scores, labels, colors, font_color, 
                        thickness, font_size, use_normalized_coordinates)
    np.copyto(image, np.array(pil_image))


def draw_bounding_boxes(image, boxes, scores=None, labels=None, colors="red", font_color="black", 
                        thickness=2, font_size=24, use_normalized_coordinates=True):
    """
    Adds multiples bounding boxes to an image.
    
    Arguments:
        + image (PIL.Image) -- A PIL.Image object.
        + boxes (list) -- Coordinates from the boxes. [[xmin, ymin, xmax, ymax], ...].
        + scores (list, np.array) -- Output scores from the FERModel.
        + labels (list, dict) -- Labels from each score.
        + colors (str, dict) -- Colors to draw bounding box. Default is 'red'. 
            A dict can be passed to draw boxes of different colors for each label.
        + font_color -- Color to draw the text.
        + thickness (int) -- Line thickness. Default value is 4.
        + font_size (int) -- Font size to draw the text.
        + use_normalized_coordinates (bool) -- If True (default), treat coordinates
            [[xmin, ymin, xmax, ymax], ...] as relative to the image.  Otherwise treat
            coordinates as absolute.
    """

    assert np.array(
        boxes).shape[1] == 4, "boxes must be a list of shape (N, 4)."

    if scores is not None:
        assert len(
            boxes) == scores.shape[0], "The number of boxes and scores must be the same."

    for i, box in enumerate(boxes):
        score = None if scores is None else scores[i]
        draw_bounding_box(image, box, score, labels, colors, font_color, thickness, 
                          font_size, use_normalized_coordinates)


def draw_text_from_array(image, text, left, right, top, bottom, color="red", font_color="black", font_size=24):
    """
    Adds a text box to an image (numpy array).
    
    Arguments:
        + image (np.array) -- Image to draw a text with shape [height, width, 3]
        + text (str) -- Text to draw on the image.
        + left (int) -- left point in the rectangle containg the text.
        + right (int) -- left point in the rectangle containg the text.
        + top (int) -- left point in the rectangle containg the text.
        + bottom (int) -- left point in the rectangle containg the text.
        + color (str) -- Color to draw the text box. Default is 'red'. 
        + font_color -- Color to draw the text.
        + font_size (int) -- Font size to draw the text.
    """

    pil_image = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_text(pil_image, text, left, right, top, bottom, color, font_color, font_size)
    np.copyto(image, np.array(pil_image))


def draw_text(image, text, left, right, top, bottom, color="red", font_color="black", font_size=24):
    """
    Adds a text box to an image (numpy array).
    
    Arguments:
        + image (PIL.Image) --  A PIL.Image object.
        + text (str) -- Text to draw on the image.
        + left (int) -- left point in the rectangle containg the text.
        + right (int) -- left point in the rectangle containg the text.
        + top (int) -- left point in the rectangle containg the text.
        + bottom (int) -- left point in the rectangle containg the text.
        + color (str) -- Color to draw the text box. Default is 'red'. 
        + font_color -- Color to draw the text.
        + font_size (int) -- Font size to draw the text.
    """

    # Init drawing tool
    draw = ImageDraw.Draw(image)
    
    # Load font
    try:
        font = ImageFont.truetype('arial.ttf', font_size)
    except IOError:
        font = ImageFont.load_default()
        
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    text_width, text_height = font.getsize(text)
    # Each display_str has a top and bottom margin of 0.05x.
    total_text_height = (1 + 2 * 0.05) *  text_height
    text_bottom = top if top > total_text_height else bottom + total_text_height
    margin = np.ceil(0.05 * text_height)
    
    # Draw rectangle
    x0, x1 = left, left + text_width
    y0, y1 = text_bottom - text_height - 2 * margin, text_bottom
    draw.rectangle([(x0, y0), (x1, y1)], fill=color)
    
    # Draw text
    draw.text((left + margin, text_bottom - text_height - margin), 
              text, fill=font_color, font=font)