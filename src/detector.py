"""Face detection utilities for the Lab 2 emotion-recognition application."""
from pathlib import Path
import cv2

class FaceDetector:
    def __init__(self, method: str = "haar", min_size: int = 30):
        if method != "haar":
            raise ValueError("Supported face detector: haar")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar Cascade: {cascade_path}")
        self.min_size = int(min_size)

    def detect(self, frame):
        """Return (x, y, w, h) boxes. For tiny FER-2013 face crops, use full-frame fallback."""
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.min_size, self.min_size),
        )
        boxes = [tuple(map(int, b)) for b in boxes]
        # FER-2013 images are already tightly cropped faces at 48x48; Haar Cascade
        # is not reliable at that resolution, so treat the small image as a face.
        h, w = gray.shape[:2]
        if not boxes and max(h, w) <= 96:
            boxes = [(0, 0, w, h)]
        return boxes
