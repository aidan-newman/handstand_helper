import cv2
from pathlib import Path
from numpy import ndarray
from PIL import Image as pillowImage
from tasks.file import get_safe_path


EXIT_KEY = 27


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


def rotate(img, rotation):
    if rotation == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Invalid rotation value.")
    return img


def compress_file(file, width=None, height=None, ignore_if_smaller=True, quality=90):
    img = pillowImage.open(file)

    if width:
        if img.size[0] <= width and ignore_if_smaller:
            return
        img = img.resize((width, int(width/img.size[0] * img.size[1])))
    elif height:
        if img.size[1] <= height and ignore_if_smaller:
            return
        img = img.resize((int(height/img.size[1] * img.size[0]), height))

    try:
        img.save(file, quality=quality, optimize=True)
    except OSError:
        img = img.convert("RGB")
        img.save(file, quality=quality, optimize=True)


def display(img, name="Output Window", hold=True):

    if hold:
        while True:
            cv2.imshow(name, img)
            key = cv2.waitKey(0)
            if key & 0xFF == EXIT_KEY:
                break
        cv2.destroyAllWindows()
        return
    cv2.imshow(name, img)
    cv2.waitKey(1)


def display_with_pillow(img, name="Output Window"):
    if isinstance(img, ndarray):
        pillowImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show(name)
    elif isinstance(img, Path):
        ary = load(img)
        pillowImage.fromarray(cv2.cvtColor(ary, cv2.COLOR_BGR2RGB)).show(img.name)
    else:
        raise ValueError("Invalid image type. Pass an np.ndarray or a pathlib.Path.")


def save(img, path):
    #  save image, if the same file already exists add a valid copy number (ex. image(#).png)
    cv2.imwrite(str(get_safe_path(path)), img)


def load(file):
    return cv2.imread(str(file))


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
