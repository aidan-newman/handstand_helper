from playsound import playsound
from threading import Thread
import paths
import queue


CORRECTION_AUDIOS = (
    paths.AUDIO / "lift_head.mp3",
    paths.AUDIO / "lower_head.mp3",
    paths.AUDIO / "hips_forward.mp3",
    paths.AUDIO / "hips_backward.mp3",
    paths.AUDIO / "ankles_forward.mp3",
    paths.AUDIO / "ankles_backward.mp3",
    paths.AUDIO / "rotate_pelvis.mp3",
    paths.AUDIO / "straighten_knees.mp3"
)


class AudioQueue:

    def __init__(self):
        self.queue = queue.Queue()
        self.stop = False
        thread = Thread(target=self._worker, args=())
        thread.daemon = True
        thread.start()

    def kill(self):
        while self.queue.get():
            pass
        self.stop = True

    def enqueue(self, task):
        if self.stop:
            raise NotImplementedError("This queue has been killed, can't enqueue.")
        self.queue.put(task)

    def _worker(self):
        while True:
            task = self.queue.get()
            self._play_audio_thread(task)
            self.queue.task_done()
            if self.stop:
                break

    def _play_audio_thread(self, path):
        self.playing = True
        playsound(str(path))
