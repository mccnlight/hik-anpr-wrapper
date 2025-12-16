import cv2
import numpy as np
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def detect(self, image: np.ndarray, conf: float = 0.25):
        """
        image: np.ndarray BGR (cv2.imread)
        return: list of dicts with bbox, conf, crop
        """
        results = self.model(image, conf=conf, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            crop = image[y1:y2, x1:x2].copy()

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": confidence,
                "crop": crop
            })

        return detections


# тестовая функция
def test_on_image(img_path, weights_path="runs/detect/train4/weights/best.pt"):
    detector = PlateDetector(weights_path)
    img = cv2.imread(img_path)

    detections = detector.detect(img)

    print(f"Found {len(detections)} plates")

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("result", img)
    cv2.waitKey(0)

