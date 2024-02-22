import cv2
from PIL import Image


def set_size(img, size: int, set_height=True):
    """
    Compresses an image to a desired height.
    :param img: Image to compress
    :param size: Size in pixels to set image to. Height set by default. Width scales.
    :param set_height: Whether to set height to size. If False, set width instead.
    :return: img
    """
    if set_height:
        height = size
        width = int(img.shape[1] * height/img.shape[0])
    else:
        width = size
        height = int(img.shape[0] * width/img.shape[1])

    img = cv2.resize(
        img,
        (width, height),
    )
    return img


def display(img, name="Output Window", hold=True):

    cv2.imshow(name, img)
    if hold:
        cv2.waitKey(0)


def display_with_pillow(img):
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()


def save(img, file):
    cv2.imwrite(file, img)


def load(file):
    return cv2.imread(file)


def draw_landmark(img, lm, color=(0, 0, 0)):
    cv2.drawMarker(img, (int(lm.x), int(lm.y)), color)


def draw_vector(img, v, org, color=(0, 0, 0), thickness=3, head_length=8):
    p1 = org
    p2 = p1 + v

    cv2.arrowedLine(img,
                    (int(p1.x), int(p1.y)),
                    (int(p2.x), int(p2.y)),
                    color,
                    thickness=thickness,
                    tipLength=1 / v.norm * head_length
                    )
