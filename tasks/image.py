import cv2
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing_utils
from tasks import landmark_analysis


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


def annotate_image(file,
                   save_file=None,
                   output_window=True,
                   pose_options=mp_pose.Pose(
                       static_image_mode=True,
                       model_complexity=2,
                       min_tracking_confidence=0.6
                   )):
    """
    Displays a window with landmarks drawn on the prominent figure.
    :param file: The image file directory to read from. If unspecified, starts live webcam feed.
    :param save_file: File directory to save the output image to. If unspecified no image file is saved.
    :param output_window: Display an output window of the annotated image. Default: True
    :param pose_options: mediapipe.python.solutions.pose.Pose() object - define custom options.
    """
    if not output_window and not save_file:
        raise ValueError("output_window and save_file can't both be False")

    img = cv2.imread(file)
    img = compress_image(img, 500)

    pose_results = pose_options.process(img)
    pose_landmarks = pose_results.pose_landmarks
    mp_drawing_utils.draw_landmarks(img, pose_landmarks, mp_pose.POSE_CONNECTIONS)

    landmark_analysis.check_form(pose_landmarks)

    if output_window:
        cv2.imshow("Annotated Image", img)

    if save_file:
        cv2.imwrite(filename=save_file, img=img)

    cv2.waitKey(0)
