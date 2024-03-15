from __future__ import annotations

import math
import cv2

import mediapipe.python.solutions.pose as mp_pose
import numpy as np
from mediapipe.tasks.python.components.containers import NormalizedLandmark

import paths
from tasks import image
from tasks import video
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

LEFT_LANDMARKS  = (0, 2, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)
RIGHT_LANDMARKS = (0, 5, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32)

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

STATIC_POSE_OPTIONS = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_tracking_confidence=0.6
)
NONSTATIC_POSE_OPTIONS = mp_pose.Pose(
    static_image_mode=False,
    smooth_landmarks=True,
    model_complexity=2,
    min_tracking_confidence=0.6
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


class HandstandFeatures:

    def __init__(self, img, static):

        self.pose_landmarks = self.get_pose_landmarks(img, static)
        self.form_vectors = None
        self.left_visible = True

        if self.pose_landmarks:
            self.form_vectors, self.left_visible = self.get_form_vectors(self.pose_landmarks, img.shape)

    @staticmethod
    def get_pose_landmarks(img, static):
        #  note: compresses image to a height of 800 -- image object is mutable, don't repeat
        if static:
            pose_options = STATIC_POSE_OPTIONS
        else:
            pose_options = NONSTATIC_POSE_OPTIONS

        if not isinstance(img, np.ndarray):
            raise ValueError("Invalid image object. Pass a np.ndarray object.")
        img = image.set_size(img, 800)

        # get pose landmarks from image
        # make sure this works !!
        pose_results = pose_options.process(img)
        if pose_results.pose_landmarks:
            pose_landmarks = list(pose_results.pose_landmarks.landmark)
            if not isinstance(pose_landmarks[0], NormalizedLandmark):  # double check this makes sense !!
                return pose_landmarks
        print("No landmarks found.")
        return None

    @staticmethod
    def get_form_vectors(pose_landmarks, shape):
        # determine which side of person is visible
        left_sum = 0
        right_sum = 0
        for i, j in zip(LEFT_LANDMARKS, RIGHT_LANDMARKS):
            left_sum += pose_landmarks[i].visibility
            right_sum += pose_landmarks[j].visibility

        left_visible = True
        if left_sum >= right_sum:
            triplets = LEFT_CORRECTION_TRIPLETS
        else:
            triplets = RIGHT_CORRECTION_TRIPLETS
            left_visible = False

        # create list of Vectors from landmarks
        form_vectors = []
        for triplet in triplets:
            v1 = get_direction_vector(triplet[0], triplet[1], shape, pose_landmarks)
            v2 = get_direction_vector(triplet[1], triplet[2], shape, pose_landmarks)
            v3 = v1 + v2

            # p = get_position_vector(triplet[1], shape, pose_landmarks)
            vector = v2.project(v3.normal())
            form_vectors.append(vector)

        return form_vectors, left_visible


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


def find_significant_correction(cors_ary, threshold=0.7):

    sums = np.array([])
    for i in range(len(cors_ary[0])):
        col_sum = 0
        for ary in cors_ary:
            col_sum += float(ary[i])
    avgs = sums / len(cors_ary)

    indices = []
    i = 0
    for avg in avgs:
        if avg >= threshold:
            indices.append(i)
    return indices


#  NEW FORMAT
def analyze_image(img,
                  model=None,
                  static=False,
                  predict=True,
                  window=False,
                  annotate=False,
                  hold=True,
                  save_file=None
                  ):

    features = HandstandFeatures(img, static)

    if not features.pose_landmarks:
        return img, None

    shape = img.shape

    # predict form corrections with neural network by inputting form vectors
    corrections = None
    if predict:
        corrections = predict_corrections(model, features.form_vectors, features.left_visible)
        i = 0
        for correction in CORRECTIONS:
            if float(corrections[i]) > 0.1:
                print(correction + ": " + str(corrections[i]))
            i += 1

    # annotate image
    if annotate:
        if features.left_visible:
            landmarks = LEFT_LANDMARKS
            triplets = LEFT_CORRECTION_TRIPLETS
        else:
            landmarks = RIGHT_LANDMARKS
            triplets = RIGHT_CORRECTION_TRIPLETS

        for i in landmarks:
            image.draw_landmark(img, get_position_vector(i, shape, features.pose_landmarks), (0, 255, 0))
            j = 0
            for vec in features.form_vectors:
                image.draw_vector(
                    img,
                    vec,
                    get_position_vector(triplets[j][1], shape, features.pose_landmarks),
                    (0, 0, 255)
                )
                j += 1

    # display output window
    if window:
        image.display(img, "Output", hold)
        # image.display_with_pillow(img)

    # save file
    if save_file:
        image.save(img, paths.OUTPUT_IMAGES, "output_image")

    del features
    return corrections


def analyze_video(filepath=None,
                  model=None,
                  predict=True,
                  window=False,
                  annotate=True,
                  save_file=None,
                  period=1000,
                  ):

    if filepath:
        vid_thread = video.VideoThread(filepath)
    else:
        vid_thread = video.VideoThread()

    target_ms = period
    cors_ary = []
    while vid_thread.run:
        if vid_thread.status:
            frame = image.set_size(vid_thread.input_frame, 800)

            corrections = analyze_image(img=frame,
                                        model=model,
                                        static=False,
                                        predict=predict,
                                        window=False,
                                        annotate=annotate,
                                        hold=False)

            vid_thread.set_frame(frame)

            if corrections is not None:
                cors_ary.append(corrections)
                if int(vid_thread.capture.get(cv2.CAP_PROP_POS_MSEC)) >= target_ms:
                    sig_cors = find_significant_correction(cors_ary)
                    cor_lbls = ""
                    for i in sig_cors:
                        cor_lbls += CORRECTIONS[i] + "\n"
                    print("Corrections:\n" + cor_lbls)
                    cors_ary.clear()
                    target_ms += vid_thread.capture.get(cv2.CAP_PROP_POS_MSEC) + period
