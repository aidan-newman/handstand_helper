import time
from threading import Thread
import cv2
from tasks import image


PAUSE_KEY = 27
EXIT_KEY = 32


class VideoThread:

    def __init__(self, src=0, msec=100):
        self.status = None
        self.input_frame = None
        self.output_frame = None
        self.capture = None
        self.run = True
        self.pause = False

        if src == 0: self.pause_frame = None
        else: self.pause_frame = 1

        self.read_thread = Thread(target=self.read, args=(src,))
        self.read_thread.daemon = True
        self.read_thread.start()
        self.dspl_thread = Thread(target=self.display, args=(msec,))
        self.dspl_thread.daemon = True
        self.dspl_thread.start()

    def read(self, src):
        if src == 0:
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(src)

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        while self.run and self.capture.isOpened():
            # print("read")
            (self.status, self.input_frame) = self.capture.read()
            self.input_frame = image.set_size(self.input_frame, 800)
            if self.pause:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.pause_frame)
            time.sleep(0.01)

    def display(self, msec):
        while not self.capture:
            pass

        while self.run and self.capture.isOpened():
            if self.status:
                if self.output_frame is not None:
                    cv2.imshow('Output Frame', self.output_frame)
                    self.output_frame = None
                else:
                    cv2.imshow('Output Frame', self.input_frame)

                key = cv2.waitKey(msec)
                # print("show")

                if key & 0xFF == EXIT_KEY:
                    self.run = False
                    self.capture.release()
                    cv2.destroyAllWindows()
                    exit(1)
                elif key & 0xFF == PAUSE_KEY and self.pause_frame:
                    self.pause = not self.pause
                    # add one incase somehow vid is paused on frame 0
                    self.pause_frame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)+1

    def set_frame(self, frame):
        self.output_frame = frame
