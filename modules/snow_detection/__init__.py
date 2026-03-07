# Snow detection via YOLO: classes 0=truck, 1=snow.
# Result: snow_detected (bool), detection_confidence (float).

from modules.snow_detection.inference import load_model, run_detection

__all__ = ["load_model", "run_detection"]
