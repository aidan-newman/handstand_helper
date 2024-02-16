from playsound import playsound
from pathlib import Path


def play_audio(path: Path):
    path.__str__()
    playsound(str(path))
