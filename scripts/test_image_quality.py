"""
Тест улучшения качества изображений:
- Показывает blur_score до и после обработки
- Сохраняет улучшенные версии
- Сравнивает исходное и улучшенное изображение
"""
import argparse
import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cv2
import numpy as np

from modules.blur import variance_of_laplacian
from modules.preprocess_ocr import preprocess_for_ocr


def main():
    parser = argparse.ArgumentParser(
        description="Test image quality improvement (blur detection and preprocessing)",
        epilog="Примеры:\n"
        "  python scripts/test_image_quality.py tests/plate_frames/b7e52cbc-3302-4cea-bf92-409c5f4e1315.jpg\n"
        "  python scripts/test_image_quality.py debug_raw_crop.jpg\n"
        "  python scripts/test_image_quality.py tests/output/b7e52cbc-3302-4cea-bf92-409c5f4e1315_orig.jpg",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Path to input image (JPEG/PNG)")
    parser.add_argument("--save", action="store_true", help="Save improved images to output directory")
    args = parser.parse_args()

    # Читаем изображение
    img_path = Path(args.input)
    if not img_path.exists():
        print(f"❌ Файл не найден: {args.input}")
        return

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ Не удалось прочитать изображение: {args.input}")
        return

    h, w = img.shape[:2]
    print(f"\n📸 Исходное изображение: {w}x{h} пикселей")
    print(f"   Размер файла: {img_path.stat().st_size / 1024:.1f} KB")

    # 1. Blur score исходного изображения
    original_blur = variance_of_laplacian(img)
    print(f"\n🔍 Blur Score (Variance of Laplacian):")
    print(f"   Исходное: {original_blur:.2f}")

    # 2. Применяем предобработку
    improved = preprocess_for_ocr(img)
    improved_blur = variance_of_laplacian(improved)
    print(f"   После предобработки: {improved_blur:.2f}")

    # 3. Улучшение
    improvement = ((improved_blur - original_blur) / max(original_blur, 1.0)) * 100
    if improvement > 0:
        print(f"   ✅ Улучшение: +{improvement:.1f}%")
    else:
        print(f"   ⚠️  Изменение: {improvement:.1f}%")

    # 4. Оценка качества
    print(f"\n📊 Оценка качества:")
    if original_blur < 60:
        print(f"   Исходное: {'🔴 Низкое' if original_blur < 30 else '🟡 Среднее'} (порог: 60)")
    else:
        print(f"   Исходное: 🟢 Хорошее")

    if improved_blur < 60:
        print(f"   Улучшенное: {'🔴 Низкое' if improved_blur < 30 else '🟡 Среднее'} (порог: 60)")
    else:
        print(f"   Улучшенное: 🟢 Хорошее")

    # 5. Сохранение результатов
    if args.save:
        output_dir = Path("tests/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = img_path.stem
        original_path = output_dir / f"{base_name}_original.jpg"
        improved_path = output_dir / f"{base_name}_improved.jpg"

        cv2.imwrite(str(original_path), img)
        cv2.imwrite(str(improved_path), improved)

        print(f"\n💾 Сохранено:")
        print(f"   Исходное: {original_path}")
        print(f"   Улучшенное: {improved_path}")
    else:
        print(f"\n💡 Используйте --save для сохранения улучшенных изображений")

    print("\n✅ Тест завершен!")


if __name__ == "__main__":
    main()
