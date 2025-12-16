from pathlib import Path
from typing import List, Tuple, Dict

from paddleocr import PaddleOCR


# === НАСТРОЙКИ ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_BEST_DIR = PROJECT_ROOT / "models" / "infer_best"
MODEL_LATEST_DIR = PROJECT_ROOT / "models" / "infer_latest"
CHAR_DICT_PATH = PROJECT_ROOT / "models" / "plate_dict.txt"

TEST_IMAGES_DIR = PROJECT_ROOT / "img" / "test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
def levenshtein(a: str, b: str) -> int:
    """Простейшая реализация расстояния Левенштейна."""
    a = a.upper()
    b = b.upper()
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # удаление
                dp[i][j - 1] + 1,      # вставка
                dp[i - 1][j - 1] + cost,  # замена
            )

    return dp[n][m]


def normalized_char_similarity(gt: str, pred: str) -> float:
    """1 - (levenshtein / max_len), от 0 до 1."""
    gt = gt.upper()
    pred = pred.upper()
    max_len = max(len(gt), len(pred), 1)
    dist = levenshtein(gt, pred)
    return 1.0 - dist / max_len


def extract_text(result) -> str:
    """
    result = ocr.ocr(path, cls=False)
    Структура: [[ [box points...], (text, score) ], ...]
    Для кропа номера, как правило, один элемент.
    """
    if not result or not result[0]:
        return ""
    # Берём самый уверенный результат
    candidates = []
    for box, (txt, score) in result[0]:
        candidates.append((txt, score))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


# === ОСНОВНОЙ КОД ===
def load_ocr(model_dir: Path) -> PaddleOCR:
    return PaddleOCR(
        use_textline_orientation=False,
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir=str(model_dir),
    )


def gather_test_images(dir_path: Path) -> List[Path]:
    files = []
    for p in sorted(dir_path.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def main():
    if not TEST_IMAGES_DIR.exists():
        print(f"[ERROR] Папка с тестовыми номерами не найдена: {TEST_IMAGES_DIR}")
        return

    print("Загружаю модели...")
    ocr_best = load_ocr(MODEL_BEST_DIR)
    ocr_latest = load_ocr(MODEL_LATEST_DIR)

    images = gather_test_images(TEST_IMAGES_DIR)
    if not images:
        print(f"[ERROR] В папке {TEST_IMAGES_DIR} нет картинок с расширениями: {IMAGE_EXTENSIONS}")
        return

    print(f"Найдено тестовых картинок: {len(images)}")
    print("Запускаю сравнение...\n")

    stats = {
        "best": {"exact": 0, "char_sim_sum": 0.0},
        "latest": {"exact": 0, "char_sim_sum": 0.0},
        "total": len(images),
    }

    # Список примеров, где модели различаются
    diffs: List[Dict] = []

    for img_path in images:
        gt = img_path.stem.upper()  # имя файла = правильный номер, например 001ALA09

        # OCR обеими моделями
        res_best = ocr_best.ocr(str(img_path), det=False, cls=False)
        res_latest = ocr_latest.ocr(str(img_path), det=False, cls=False)


        pred_best = extract_text(res_best).upper()
        pred_latest = extract_text(res_latest).upper()

        sim_best = normalized_char_similarity(gt, pred_best)
        sim_latest = normalized_char_similarity(gt, pred_latest)

        if pred_best == gt:
            stats["best"]["exact"] += 1
        if pred_latest == gt:
            stats["latest"]["exact"] += 1

        stats["best"]["char_sim_sum"] += sim_best
        stats["latest"]["char_sim_sum"] += sim_latest

        if pred_best != pred_latest:
            diffs.append(
                {
                    "image": img_path,
                    "gt": gt,
                    "best": pred_best,
                    "best_sim": sim_best,
                    "latest": pred_latest,
                    "latest_sim": sim_latest,
                }
            )

    total = stats["total"]
    best_acc = stats["best"]["exact"] / total
    latest_acc = stats["latest"]["exact"] / total

    best_char = stats["best"]["char_sim_sum"] / total
    latest_char = stats["latest"]["char_sim_sum"] / total

    print("=== РЕЗУЛЬТАТЫ ===")
    print(f"Всего тестов: {total}")
    print()
    print("*** Модель infer_best ***")
    print(f"Полная точность (номер целиком): {best_acc:.3f}")
    print(f"Средняя точность по символам:    {best_char:.3f}")
    print()
    print("*** Модель infer_latest ***")
    print(f"Полная точность (номер целиком): {latest_acc:.3f}")
    print(f"Средняя точность по символам:    {latest_char:.3f}")
    print()

    print("=== Примеры, где модели различаются ===")
    # выведем до 20 примеров
    for item in diffs[:20]:
        print("-" * 60)
        print(f"Файл:   {item['image'].name}")
        print(f"GT:     {item['gt']}")
        print(f"BEST:   {item['best']} (sim={item['best_sim']:.3f})")
        print(f"LATEST: {item['latest']} (sim={item['latest_sim']:.3f})")

    print("\nГотово.")


if __name__ == "__main__":
    main()
