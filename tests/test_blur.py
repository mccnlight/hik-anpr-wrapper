import cv2
import numpy as np
import sys
import os

# Добавляем корневую директорию проекта в путь
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from modules.blur import variance_of_laplacian


def test_variance_of_laplacian_sharp_vs_blur() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.putText(img, "KZ123ABC", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    blur = cv2.GaussianBlur(img, (9, 9), 2.0)

    sharp_score = variance_of_laplacian(img)
    blur_score = variance_of_laplacian(blur)

    print(f"\n🔍 Тест blur detection:")
    print(f"   Резкое изображение: {sharp_score:.2f}")
    print(f"   Размытое изображение: {blur_score:.2f}")
    print(f"   Разница: {sharp_score - blur_score:.2f}")
    
    if sharp_score > blur_score:
        print(f"   ✅ ТЕСТ ПРОЙДЕН: резкое изображение имеет больший blur_score")
        return True
    else:
        print(f"   ❌ ТЕСТ НЕ ПРОЙДЕН: резкое изображение должно иметь больший blur_score")
        return False


if __name__ == "__main__":
    success = test_variance_of_laplacian_sharp_vs_blur()
    sys.exit(0 if success else 1)
