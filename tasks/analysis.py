from __future__ import annotations

import math
import cv2

import mediapipe.python.solutions.pose as mp_pose
import numpy as np
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.python.solutions import drawing_utils as mp_drawing_utils

import paths
from tasks import image
from tasks import file
from neural_network.predict import predict as predict_corrections


LANDMARK_NAMES = (
    "nose",
    "left eye (inner)",
    "left eye",  # 2
    "left eye (outer)",
    "right eye (inner)",
    "right eye",  # 5
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
    "left pinky",  # 17
    "right pinky",
    "left index",  # 19
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
    "right foot index"
)

LEFT_FEATURES  = (0, 2, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)
RIGHT_FEATURES = (0, 5, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32)

CORRECTIONS = (
    "lift_head",
    "lower_head",
    "hips_forward",
    "hips_backward",
    "ankles_forward",
    "ankles_backward",
    "rotate_pelvis",
    "straighten_knees"
)

LEFT_CORRECTION_TRIPLETS = (
    (11, 23, 27),
    (15, 11, 23),
    (11, 23, 25),
    (23, 25, 27),
    (7,  11, 23)
)
RIGHT_CORRECTION_TRIPLETS = (
    (12, 24, 28),
    (16, 12, 24),
    (12, 24, 28),
    (24, 26, 28),
    (8,  12, 24)
)


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

    def flip_x(self):
        return Vector(
            x=-self.x,
            y=self.y
        )


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

    return form_vecs


def find_significant_correction(cor_arys, threshold=0.7):

    sums = np.array([])
    for i in range(len(cor_arys[0])):
        col_sum = 0
        for ary in cor_arys:
            col_sum += float(ary[i])
    avgs = sums / len(cor_arys)

    indices = []
    i = 0
    for avg in avgs:
        if avg >= threshold:
            indices.append(i)
    return indices


#  NEW FORMAT
def analyze_image(img,
                  predict=True,
                  window=False,
                  annotate=1,
                  hold=True,
                  save_file=None,
                  pose_options=mp_pose.Pose(
                      static_image_mode=True,
                      model_complexity=2,
                      min_tracking_confidence=0.6
                  ),
                  destroy_windows=True):
    # annotate: 0=none, 1=just interested, 2=mediapipe
    if not isinstance(img, np.ndarray):
        raise ValueError("Invalid image object. Pass a np.ndarray object.")
    img = image.set_size(img, 800)

    pose_results = pose_options.process(img)
    if not pose_results.pose_landmarks:
        print("No landmarks found.")
        return None, None
    pose_landmarks = list(pose_results.pose_landmarks.landmark)

    if isinstance(pose_landmarks[0], NormalizedLandmark):
        print("No landmarks found.")
        return
    pose_landmarks = list(pose_results.pose_landmarks.landmark)

    shape = img.shape

    # determine which side of the person is prominent
    left_sum = 0
    right_sum = 0
    for i, j in zip(LEFT_FEATURES, RIGHT_FEATURES):
        left_sum += pose_landmarks[i].visibility
        right_sum += pose_landmarks[j].visibility

    # get appropriate vectors based on side visible
    if left_sum >= right_sum:
        form_vectors = get_form_vectors(LEFT_CORRECTION_TRIPLETS, shape, pose_landmarks)
    else:
        form_vectors = get_form_vectors(RIGHT_CORRECTION_TRIPLETS, shape, pose_landmarks)

    # convert vectors to a list of lists for ease of access in other modules
    vectors_list = []
    for vec in form_vectors:
        #  normalize vectors left side of a person (simply flip x components)
        if right_sum > left_sum:
            vectors_list.append(vec.flip_x().to_list())
        else:
            vectors_list.append(vec.to_list())

    # predict form corrections with neural network by inputting form vectors
    corrections = None
    if predict:
        corrections = predict_corrections(vectors_list)
        i = 0
        for correction in CORRECTIONS:
            if float(corrections[i]) > 0.09:
                print(correction + ": " + str(corrections[i]))
            i += 1

    # handle annotations
    if window or save_file:
        if annotate == 1:  # only show relevant features and form vectors

            if left_sum >= right_sum:
                features = LEFT_FEATURES
                triplets = LEFT_CORRECTION_TRIPLETS
            else:
                features = RIGHT_FEATURES
                triplets = RIGHT_CORRECTION_TRIPLETS

            for i in features:
                image.draw_landmark(img, get_position_vector(i, shape, pose_landmarks), (0, 255, 0))
                j = 0
                for vec in form_vectors:
                    image.draw_vector(
                        img,
                        vec,
                        get_position_vector(triplets[j][1], shape, pose_landmarks),
                        (0, 0, 255)
                    )
                    j += 1

        elif annotate == 2:  # use mediapipe's drawing util to draw landmarks and connections
            mp_drawing_utils.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        elif annotate != 0:  # don't do any annotations if 0, if other value: error
            raise ValueError("Invalid annotate parameter value.")

        if window:
            image.display(img, "Output", hold)
            # image.display_with_pillow(img)
        if save_file:
            image.save(img, paths.OUTPUT_IMAGES, "output_image")

    if destroy_windows:
        cv2.destroyAllWindows()

    return corrections, vectors_list


def analyze_video(filepath=None,
                  predict=True,
                  window=False,
                  annotate=1,
                  save_file=None,
                  exit_key=27,
                  pause_key=32,
                  pose_options=mp_pose.Pose(
                      static_image_mode=False,
                      smooth_landmarks=True,
                      model_complexity=2,
                      min_tracking_confidence=0.6),
                  period=1000,
                  ):
    # annotate: 0=none, 1=just interested, 2=mediapipe

    if filepath:
        cap = cv2.VideoCapture(str(filepath))
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    pause = False
    pause_frame = 0
    run = True
    target_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) + period
    cor_arys = []
    while run:

        ret, frame = cap.read()
        ch = cv2.waitKey(1)

        if pause:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pause_frame)
        elif ret:
            frame = image.set_size(frame, 800)

            cors, _ = analyze_image(frame, predict, window, annotate, False, False, pose_options, destroy_windows=False)
            cor_arys.append(cors)
            if int(cap.get(cv2.CAP_PROP_POS_MSEC)) >= target_ms:
                sig_cors = find_significant_correction(cor_arys)
                cor_lbls = ""
                for i in sig_cors:
                    cor_lbls += CORRECTIONS[i] + "\n"
                print("Corrections:\n" + cor_lbls)
                cor_arys.clear()
                target_ms += period

        if ch & 0xFF == exit_key or not ret:  # escape key
            run = False
        elif ch & 0xFF == pause_key:
            pause = not pause
            pause_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    cap.release()
    cv2.destroyAllWindows()
