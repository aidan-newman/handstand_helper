import numpy as np
from mediapipe.tasks.python.components.containers import NormalizedLandmark

LANDMARK_NAMES = (
    "nose",
    "left eye (inner)",
    "left eye",
    "left eye (outer)",
    "right eye (inner)",
    "right eye",
    "right eye (outer)",
    "left ear",
    "right ear",
    "mouth (left)",
    "mouth (right)",
    "left shoulder",  # 11
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left pinky",
    "right pinky",
    "left index",
    "right index",
    "left thumb",
    "right thumb",
    "left hip",  # 23
    "right hip",
    "left knee",
    "right knee",
    "left ankle",  # 27
    "right ankle",
    "left heel",
    "right heel",
    "left foot index",
    "right foot index",
)

CORRECTIONS = (
    "Neutral Head",
    "Rotate Pelvis",
    "Hips Above Shoulders",
    "Straighten Legs",
)

VISIBILITY_THRESHOLD = 0.2


def get_vector(l1, l2, pose_landmarks):

    v1 = np.array([pose_landmarks[l1].x, 1-pose_landmarks[l1].y, pose_landmarks[l1].z])
    v2 = np.array([pose_landmarks[l2].x, 1-pose_landmarks[l2].y, pose_landmarks[l2].z])

    return np.subtract(v2, v1)


def check_form(pose_landmarks):

    pose_landmarks = list(pose_landmarks.landmark)

    if isinstance(pose_landmarks[0], NormalizedLandmark):
        print("No landmarks found.")
        return None

    should_to_hip = get_vector(11, 23, pose_landmarks)
    hip_to_ank = get_vector(23, 27, pose_landmarks)
    should_to_ank = get_vector(11, 27, pose_landmarks)

    # for i in desired_outputs:
    #     if pose_landmarks[i].visibility > VISIBILITY_THRESHOLD:
    #         print(LANDMARK_NAMES[i] + " - x: " + str(pose_landmarks[i].x))
    #         print(LANDMARK_NAMES[i] + " - y: " + str(pose_landmarks[i].y))
    #         print(LANDMARK_NAMES[i] + " - z: " + str(pose_landmarks[i].z))

