from __future__ import annotations
import math
from mediapipe.tasks.python.components.containers import NormalizedLandmark
import cv2
import numpy as np
from pathlib import Path
from tasks import audio
import os

LANDMARK_NAMES = (
    "nose",
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
    "right foot index",
)

INTERESTED_FEATURES = (0, 2, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)

CORRECTIONS = {
    "neutral_head",
    "rotate_pelvis",
    "hips_forward",
    "straighten_knees"
}

VISIBILITY_THRESHOLD = 0.2


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

    def draw(self, org: Vector, img, color=(0, 0, 0)):

        v = self
        p1 = org
        p2 = p1+v

        cv2.arrowedLine(img,
                        (int(p1.x), int(p1.y)),
                        (int(p2.x), int(p2.y)),
                        color,
                        thickness=3,
                        tipLength=1 / self.norm * 10
                        )


def get_vector(i1, i2, shape, pose_landmarks) -> Vector:

    l1 = Vector(pose_landmarks[i1].x * shape[1], pose_landmarks[i1].y*shape[0])
    l2 = Vector(pose_landmarks[i2].x * shape[1], pose_landmarks[i2].y*shape[0])

    return l2 - l1


def get_cords(i, shape, pose_landmarks) -> Vector:
    return Vector(pose_landmarks[i].x * shape[1], pose_landmarks[i].y * shape[0], 0)


def draw_opening_vector(i1, i2, i3, img, shape, pose_landmarks) -> Vector:

    v1 = get_vector(i1, i2, shape, pose_landmarks)
    v2 = get_vector(i2, i3, shape, pose_landmarks)
    v3 = v1 + v2

    p = get_cords(i2, shape, pose_landmarks)

    d = v2.project(v3.normal())
    print(LANDMARK_NAMES[i1] + "->" + LANDMARK_NAMES[i2] + "->" + LANDMARK_NAMES[i3] + ": " + str(d))

    d.draw(p, img, (50, 50, 255))
    return d


def check_form(pose_landmarks, img):

    pose_landmarks = list(pose_landmarks.landmark)

    if isinstance(pose_landmarks[0], NormalizedLandmark):
        print("No landmarks found.")
        return None

    shape = img.shape

    for i in INTERESTED_FEATURES:
        cv2.drawMarker(img, (int(pose_landmarks[i].x * shape[1]), int(pose_landmarks[i].y * shape[0])), (0, 255, 0))

    vecs = (
            draw_opening_vector(11, 23, 27, img, shape, pose_landmarks),
            draw_opening_vector(15, 11, 23, img, shape, pose_landmarks),
            draw_opening_vector(11, 23, 25, img, shape, pose_landmarks),
            draw_opening_vector(23, 25, 27, img, shape, pose_landmarks),
            draw_opening_vector(7, 11, 23, img, shape, pose_landmarks)
            )

    # p = Path().resolve() / "audio/hips_forward.mp3"
    # audio.play_audio(p)
