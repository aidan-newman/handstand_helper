from __future__ import annotations
import math
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


def get_vector(i1, i2, pose_landmarks) -> Vector:

    l1 = Vector(pose_landmarks[i1].x, 1-pose_landmarks[i1].y, pose_landmarks[i1].z)
    l2 = Vector(pose_landmarks[i2].x, 1-pose_landmarks[i2].y, pose_landmarks[i2].z)

    return l2 - l1


def check_form(pose_landmarks):

    pose_landmarks = list(pose_landmarks.landmark)

    if isinstance(pose_landmarks[0], NormalizedLandmark):
        print("No landmarks found.")
        return None

    should_to_hip = get_vector(11, 23, pose_landmarks)
    hip_to_ank = get_vector(23, 27, pose_landmarks)
    should_to_ank = should_to_hip + hip_to_ank

    print(should_to_hip.project(should_to_ank.normal()))
    print(hip_to_ank.project(should_to_ank.normal()))
