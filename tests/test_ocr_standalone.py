# test_ocr_standalone.py
import cv2
from modules.ocr import PlateOCR

if __name__ == "__main__":
    img = cv2.imread("debug_proc_crop.jpg")  # или debug_raw_crop.jpg
    print("shape:", img.shape)
    ocr = PlateOCR()
    text, conf = ocr.recognize(img)
    print("OCR RESULT:", text, conf)
