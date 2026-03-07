#!/usr/bin/env python3
"""
Скрипт обучения YOLOv8 для детекции снега в кузове грузовика.
Классы: 0 = truck, 1 = snow.

Запуск (из корня hik-anpr-wrapper):
  python train_yolo.py

Требуется:
  - dataset/data.yaml с путями к train/val и names: {0: truck, 1: snow}
  - изображения в dataset/images/train, dataset/images/val
  - разметка в YOLO-формате в dataset/labels/train, dataset/labels/val
    (один .txt на изображение: "class_id x_center y_center width height" нормализованные 0-1)

Веса сохраняются в models/training_runs/ (после обучения скопируй best.pt
и укажи путь в SNOW_YOLO_WEIGHTS в app.env).
"""
import os
import sys

# Путь к data.yaml: относительно корня проекта
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(PROJECT_ROOT, "dataset", "data.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "training_runs")


def main():
    if not os.path.isfile(DATA_YAML):
        print(f"[TRAIN] ERROR: dataset config not found: {DATA_YAML}")
        print("Create dataset/data.yaml and fill dataset/images and dataset/labels (train/val).")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[TRAIN] ERROR: ultralytics not installed. pip install ultralytics")
        sys.exit(1)

    # Базовая модель: yolov8n (nano). Для точнее — yolov8s / yolov8m
    # УТОЧНЕНИЕ: если нужна другая база, замени строку ниже
    model_name = os.getenv("YOLO_BASE_MODEL", "yolov8n.pt")
    epochs = int(os.getenv("YOLO_EPOCHS", "100"))
    imgsz = int(os.getenv("YOLO_IMGSZ", "640"))
    batch = int(os.getenv("YOLO_BATCH", "16"))

    print(f"[TRAIN] Loading base model: {model_name}")
    model = YOLO(model_name)

    print(f"[TRAIN] Dataset: {DATA_YAML}")
    print(f"[TRAIN] Output dir: {OUTPUT_DIR}")
    print(f"[TRAIN] Epochs={epochs}, imgsz={imgsz}, batch={batch}")

    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=OUTPUT_DIR,
        name="snow_detection",
        exist_ok=True,
        verbose=True,
    )

    print(f"[TRAIN] Done. Weights: {OUTPUT_DIR}/snow_detection/weights/best.pt")
    print("Set in app.env: SNOW_YOLO_WEIGHTS=models/training_runs/snow_detection/weights/best.pt")


if __name__ == "__main__":
    main()
