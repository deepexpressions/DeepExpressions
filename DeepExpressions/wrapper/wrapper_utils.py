import cv2
import collections
from copy import deepcopy

DEFAULT_STYLE = {
        "image":{
            "ratio":1.0,
            "text":{
                "fontFace":cv2.FONT_HERSHEY_SIMPLEX,
                "fontScale":1,
                "color":(255,255,255),
                "thickness":1,
                "lineType":1,
            },
        },

        "bounding_box":{
            "color":(20,186,59),
            "thickness":1,
            "offset":"0%",
            "text":{
                "fontFace":cv2.FONT_HERSHEY_SIMPLEX,
                "fontScale":1.0,
                "color":(255,255,255),
                "thickness":1,
                "lineType":1,
            },
        }
    }


def update_style(style=None):
    """Update output data style."""

    def update_dict(d, u):
        """Update dict d based on new values in u."""
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    if style is None:
        return DEFAULT_STYLE
    else:
        return update_dict(deepcopy(DEFAULT_STYLE), style)


def draw_bounding_box(image, bb_list, labels, style):
    """Draw bounding box in image"""

    # Check bb_list and labels list lenght
    assert len(bb_list) == len(labels), "Bouding Boxes and Labels incompatibles"

    # Loop over detections
    for i in range(len(bb_list)):
        x0, y0, x1, y1 = bb_list[i]

        # Draw bb
        cv2.rectangle(image, (x0, y0), (x1, y1), style["bounding_box"]["color"], style["bounding_box"]["thickness"])
        cv2.rectangle(image, (x0, y1-35), (x1, y1), style["bounding_box"]["color"], cv2.FILLED)
        # Write label
        cv2.putText(image, labels[i], (x0+6, y1-6), style["bounding_box"]["text"]["fontFace"], style["bounding_box"]["text"]["fontScale"], 
                    style["bounding_box"]["text"]["color"], style["bounding_box"]["text"]["thickness"], style["bounding_box"]["text"]["lineType"])

    return image