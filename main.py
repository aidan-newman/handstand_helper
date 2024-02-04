
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draws landmarks on an input image that contains a person.
    """
    pose_landmarks_list = detection_result.pose_landmarks
    print(pose_landmarks_list)
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for landmark in pose_landmarks:
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def compress_image(img, height):
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


# create a PoseLandmarker object.
base_options = mp_python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
)
detector = vision.PoseLandmarker.create_from_options(options)

# load image
image = mp.Image.create_from_file("input_images/image.jpg")

# detect pose landmarks from the input image
detection_result = detector.detect(image)

# create output image (draw landmarks on input)
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# compress and recolour
annotated_image = compress_image(annotated_image, 1000)
annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

# show window and save to file
cv2.imshow("Output Window", annotated_image)
cv2.imwrite("output_images/output_image.jpg", annotated_image)
cv2.waitKey(0)
