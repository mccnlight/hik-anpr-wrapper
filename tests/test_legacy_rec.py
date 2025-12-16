from pathlib import Path

from modules.legacy_plate_recognizer import LegacyPlateRecognizer


ROOT = Path(__file__).resolve().parent

MODEL_BEST_DIR = ROOT / "models" / "infer_best"
TEST_IMG = ROOT / "img" / "test" / "235AZ15.jpg"


def main():
    rec = LegacyPlateRecognizer(
        MODEL_BEST_DIR,
        use_gpu=False,
    )

    text, score = rec.predict(TEST_IMG)
    print(f"RESULT: {text}  (score={score:.3f})")


if __name__ == "__main__":
    main()
