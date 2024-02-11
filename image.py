import cv2
import numpy


def compress_image(img, height: int):
    """
    Compresses an image to a desired height.
    :param img: Image to compress
    :param height: Desired height in pixels, width will scale appropriately
    :return: img
    """
    width = int(img.shape[1] * height/img.shape[0])
    img = cv2.resize(
        img,
        (width, height),
    )
    return img
