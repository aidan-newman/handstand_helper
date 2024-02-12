import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing_utils
from mediapipe.python.solutions import pose as mp_pose
from tasks import image
from tasks import landmark_analysis


def annotate_video(file=None, exit_key=27, pause_key=32):
    """
    Displays a window with landmarks drawn on the prominent figure.
    :param file: The video file directory to read from. If unspecified, starts live webcam feed.
    :param exit_key: ASCII code to close output window. Default: escape key
    :param pause_key: ASCII code to pause/unpause video in output window. Default: space key
    """
    pose = mp_pose.Pose(
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    )

    if file:
        cap = cv2.VideoCapture(file)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    pause = False
    pause_frame = 0
    run = True
    ms = 0
    while run:

        ret, frame = cap.read()
        ch = cv2.waitKey(1)

        if pause:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pause_frame)
        elif ret:
            frame = image.compress_image(frame, 500)

            pose_results = pose.process(frame)
            pose_landmarks = pose_results.pose_landmarks
            mp_drawing_utils.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmark_analysis.check_form(pose_landmarks)
            cv2.imshow("Annotated Output", frame)

        if ch & 0xFF == exit_key or not ret:  # escape key
            run = False
        elif ch & 0xFF == pause_key:
            pause = not pause
            pause_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ms += 1

    cap.release()
    cv2.destroyAllWindows()
