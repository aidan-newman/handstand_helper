from __future__ import annotations

import math
import cv2

import mediapipe.python.solutions.pose as mp_pose
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.python.solutions import drawing_utils as mp_drawing_utils

from tasks import image
from neural_network import predict


LANDMARK_NAMES = ("nose",
    "left eye (inner)",
    "left eye",
    "left eye (outer)",
    "right eye (inner)",
    "right eye",
    "right eye (outer)",
    "left ear",  # 7
    "right ear",
    "mouth (left)",
    "mouth (right)",
    "left shoulder",  # 11
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",  # 15
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
    "right foot index",)

INTERESTED_FEATURES = (0, 2, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)

CORRECTIONS = {
    "neutral_head",
    "rotate_pelvis",
    "hips_forward",
    "straighten_knees"
}

CORRECTION_TRIPLETS = ((11, 23, 27),
    (15, 11, 23),
    (11, 23, 25),
    (23, 25, 27),
    (7, 11, 23))


class Vector:

    def __init__(self, x=float(0), y=float(0), z=float(0)):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, v: Vector) -> Vector:
        return Vector(
            self.x + v.x,
            self.y + v.y,
            self.z + v.z
        )

    def __neg__(self):
        return Vector(
            self.x * -1,
            self.y * -1,
            self.z * -1
        )

    def __sub__(self, v: Vector) -> Vector:
        return self + -v

    def __mul__(self, s: float) -> Vector:
        return Vector(
            self.x * s,
            self.y * s,
            self.z * s
        )

    def __rmul__(self, other) -> Vector:
        return self * other

    def __str__(self):
        return "[" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + "]"

    @property
    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def to_list(self) -> list:
        return [self.x, self.y, self.z]

    def dot(self, v: Vector) -> float:
        return self.x*v.x + self.y*v.y + self.z*v.z

    def cross(self, v: Vector) -> Vector:
        return Vector(
            self.y*v.z - self.z*v.y,
            self.z*v.x - self.x*v.z,
            self.x*v.y - self.y*v.x
        )

    def unit(self):
        norm = self.norm
        return Vector(
            self.x / norm,
            self.y / norm,
            self.z / norm
        )

    def normal(self, plane=0):
        """
        :param plane: Plane of resulting normal vector (0=xy, 1=xz, 2=yz).
        """
        if plane == 0:
            return Vector(
                x=self.y,
                y=-self.x
            )
        elif plane == 1:
            return Vector(
                x=-self.z,
                z=self.x
            )
        elif plane == 2:
            return Vector(
                y=self.z,
                z=-self.y
            )
        raise ValueError("Invalid plane option.")

    def angle(self, v: Vector) -> float:
        return math.acos(self.dot(v)/(self.norm * v.norm))

    def project(self, v: Vector) -> Vector:
        return self.dot(v.unit()) * v.unit()


def get_direction_vector(i1, i2, shape, pose_landmarks) -> Vector:
    """
    Returns the direction vector from pose_landmarks[i1] to pose_landmarks[i2].
    :param i1: Index of first pose landmark.
    :param i2: Index of second pose landmark.
    :param shape: Image shape.
    :param pose_landmarks: List of pose landmarks.
    :return: direction_vector
    """
    l1 = Vector(pose_landmarks[i1].x * shape[1], pose_landmarks[i1].y*shape[0])
    l2 = Vector(pose_landmarks[i2].x * shape[1], pose_landmarks[i2].y*shape[0])

    return l2 - l1


def get_position_vector(i, shape, pose_landmarks) -> Vector:
    """
    Returns the position vector for a given pose landmark.
    :param i: Pose landmark index.
    :param shape: Image shape.
    :param pose_landmarks: List of pose landmarks.
    :return: position_vector
    """
    return Vector(pose_landmarks[i].x * shape[1], pose_landmarks[i].y * shape[0], 0)


def get_form_vectors(triplets, shape, pose_landmarks) -> list:

    form_vecs = []
    for triplet in triplets:
        v1 = get_direction_vector(triplet[0], triplet[1], shape, pose_landmarks)
        v2 = get_direction_vector(triplet[1], triplet[2], shape, pose_landmarks)
        v3 = v1 + v2

        # p = get_position_vector(triplet[1], shape, pose_landmarks)
        form_vec = v2.project(v3.normal())
        form_vecs.append(form_vec)
        # print(LANDMARK_NAMES[i1] + "->" + LANDMARK_NAMES[i2] + "->" + LANDMARK_NAMES[i3] + ": " + str(d))

    return form_vecs


#  NEW FORMAT
def analyze_image(img,
                  window=False,
                  annotate=1,
                  hold=True,
                  save_file=None,
                  pose_options=mp_pose.Pose(
                      static_image_mode=True,
                      model_complexity=2,
                      min_tracking_confidence=0.6
                  )):
    # annotate: 0=none, 1=just interested, 2=mediapipe
    img = image.set_size(img, 800)

    pose_results = pose_options.process(img)
    pose_landmarks = list(pose_results.pose_landmarks.landmark)

    if isinstance(pose_landmarks[0], NormalizedLandmark):
        print("No landmarks found.")
        return

    shape = img.shape
    form_vectors = get_form_vectors(CORRECTION_TRIPLETS, shape, pose_landmarks)
    vectors_lists = []
    for vec in form_vectors:
        vectors_lists.append(vec.to_list())

    # predict form corrections with neural network by inputting form vectors
    corrections = predict.predict(vectors_lists)

    # handle annotations
    if window or save_file:

        if annotate == 1:
            for i in INTERESTED_FEATURES:
                image.draw_landmark(img, get_position_vector(i, shape, pose_landmarks), (0, 255, 0))
                i = 0
                for vec in form_vectors:
                    image.draw_vector(
                        img,
                        vec,
                        get_position_vector(CORRECTION_TRIPLETS[i][1], shape, pose_landmarks),
                        (0, 0, 255)
                    )
                    i += 1

        elif annotate == 2:
            mp_drawing_utils.draw_landmarks(img, pose_landmarks, mp_pose.POSE_CONNECTIONS)

        elif annotate != 0:
            raise ValueError("Invalid annotate parameter value.")

        if window:
            image.display(img, hold=hold)
        else:
            image.save(img, save_file)

    cv2.destroyAllWindows()
    return form_vectors


def analyze_video(file=None,
                  window=False,
                  annotate=1,
                  save_file=None,
                  exit_key=27,
                  pause_key=32,
                  pose_options=mp_pose.Pose(
                      model_complexity=2,
                      min_tracking_confidence=0.6
                  )):
    # annotate: 0=none, 1=just interested, 2=mediapipe

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
            frame = image.set_size(frame, 800)

            analyze_image(frame, window, annotate, False, False, pose_options)
            # HANDLE SAVING FILE

        if ch & 0xFF == exit_key or not ret:  # escape key
            run = False
        elif ch & 0xFF == pause_key:
            pause = not pause
            pause_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ms += 1

    cap.release()
    cv2.destroyAllWindows()
