import time
from threading import Thread
import cv2
from tasks.file import get_safe_path


PAUSE_KEY = 32
EXIT_KEY = 27


class VideoThread:

    def __init__(self, src=None, display=True, msec=10):
        self.status = None
        self.input_frame = None
        self.output_frame = None
        self.capture = None
        self.run = True
        self.pause = False

        if src is None:
            self.pause_frame = None
            src = 0
        else:
            self.pause_frame = 1

        self.read_thread = Thread(target=self.read, args=(src,))
        self.read_thread.daemon = True
        self.read_thread.start()
        if display:
            self.dspl_thread = Thread(target=self.display, args=(msec,))
            self.dspl_thread.daemon = True
            self.dspl_thread.start()

    def read(self, src,):
        if src == 0:
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(str(src))

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        while self.run and self.capture.isOpened():

            (self.status, self.input_frame) = self.capture.read()
            if self.status:
                if self.pause:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.pause_frame)
                time.sleep(0.01)
            else:
                self.run = False

        self.capture.release()
        cv2.destroyAllWindows()

    def display(self, msec):
        while not self.capture:
            pass

        # writer = None
        # if write:
        #     save_location = get_safe_path(path)
        #     writer = cv2.VideoWriter(
        #         str(save_location),
        #         fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
        #         fps=20.0,
        #         frameSize=(int(self.capture.get(3)), int(self.capture.get(4))),
        #         isColor=True
        #     )

        while self.run and self.capture.isOpened():
            if self.status:
                if self.output_frame is not None:
                    cv2.imshow('Output Frame', self.output_frame)
                    # if write:
                    #     writer.write(self.output_frame)
                    self.output_frame = None

                    key = cv2.waitKey(1)

                    if key & 0xFF == EXIT_KEY:
                        self.run = False
                        return
                    elif key & 0xFF == PAUSE_KEY and self.pause_frame:
                        self.pause = not self.pause
                        # add one incase somehow vid is paused on frame 0
                        self.pause_frame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)+1
        # writer.release()

    def set_frame(self, frame):
        self.output_frame = frame
