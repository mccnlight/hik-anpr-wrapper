import argparse
import time

import cv2
import numpy as np

from modules.upscale_hat import HATUpscaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Test HAT upscale on a plate crop")
    parser.add_argument("input", help="Path to input image (crop)")
    parser.add_argument("output", help="Path to save enhanced output image")
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.input}")

    upscaler = HATUpscaler()
    start = time.time()
    enhanced = upscaler.enhance(img)
    elapsed = time.time() - start

    ok = cv2.imwrite(args.output, enhanced)
    if not ok:
        raise SystemExit(f"Cannot write output: {args.output}")

    print(f"Input:  {img.shape[1]}x{img.shape[0]}")
    print(f"Output: {enhanced.shape[1]}x{enhanced.shape[0]}")
    print(f"Inference time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
