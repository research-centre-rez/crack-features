import argparse
import json
import imageio.v3 as iio
import numpy as np
from skimage.color import rgb2gray
import cv2
import image_logger
from tqdm.auto import tqdm
import logging
import os

import output_generator


def _load_input(file_path):
    img = iio.imread(file_path).astype(int)
    if np.max(img) > 255:
        img = (img // 256).astype(np.uint8)

    # check dimensionality of the input (grayscale, rgb or rgba)
    if len(img.shape) == 2:
        return img.astype(np.uint8)

    if img.shape[2] == 1:
        return img[:, :, 0].astype(np.uint8)

    # from rgba drop a dimension
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return (rgb2gray(img.astype(np.uint8), channel_axis=2) * 255).astype(np.uint8)


def threshold_image(gray_img):
    colored = np.zeros(gray_img.shape + (3,))
    for th in CRACKS_CONFIG["threshold_levels"]:
        colored[np.logical_and(th["range"][0] < gray_img, gray_img <= th["range"][1]), :] = np.array(th["color"])
    return colored


def normalize_hist(img, mid_target):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    img = clahe.apply(img)
    out = np.copy(img)
    mid = np.argmax(np.histogram(img.reshape(-1), bins=256)[0])
    out[out <= mid] = img[img <= mid] / mid * mid_target
    out[out > mid] = (out[out > mid] - mid) / (255 - mid) * (255 - mid_target) + mid_target
    return out


def crack_mask(input_path, output_dir_path):
    # Load input image
    gray_img = _load_input(input_path)
    image_logger.info(gray_img, output_dir_path, "[user]input-grayscale.png")

    # Normalize histogram
    gray_norm = normalize_hist(gray_img, CRACKS_CONFIG["median_level"])
    image_logger.info(gray_norm, output_dir_path, "[user]input-normalized.png")

    # Apply thresholds
    colored = threshold_image(gray_norm)
    image_logger.info(colored, output_dir_path, "[user]colored.png")

    # Create crack mask
    th0 = CRACKS_CONFIG["threshold_levels"][0]["range"]
    th2 = CRACKS_CONFIG["threshold_levels"][1]["range"]
    candidates = (gray_norm < th2[1]).astype(np.uint8) + np.logical_and(th0[0] < gray_norm, gray_norm < th0[1]).astype(np.uint8)
    for y in tqdm(range(candidates.shape[0]), desc="Flood fill forward"):
        for x in range(candidates.shape[1]):
            if candidates[y, x] == 1 and (
                    (y - 1 > 0 and candidates[y - 1, x] > 1) or
                    (x - 1 > 0 and candidates[y, x - 1] > 1)):
                candidates[y, x] += 1

    for y in tqdm(np.arange(candidates.shape[0])[::-1], desc="Flood fill backward"):
        for x in np.arange(candidates.shape[1])[::-1]:
            if candidates[y, x] and (
                    (y + 1 < candidates.shape[0] and candidates[y + 1, x] > 1) or
                    (x + 1 < candidates.shape[1] and candidates[y, x + 1] > 1)):
                candidates[y, x] += 1

    mask = (candidates > 1).astype(np.uint8)
    image_logger.info(mask * 255, output_dir_path, "[user]crack-mask.png")
    image_logger.dump_image(os.path.join(output_dir_path, "crack-mask.png"), mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Preprocessing of the crack scans. Target of this routine is to create crack mask.
    """)
    parser.add_argument(
        "input_image_path",
        type=str,
        help="Path to grayscale input image where cracks should be detected.")
    parser.add_argument(
        "-o",
        "--output_dir_path",
        type=str,
        help="Path to output directory. If directory does not exist, it will be created (but parent directory must exist).",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help=f"Path to the cracks_config.json. In this file are defined crack image processing parameters."
             f"(@see config/cracks_config.json).",
        default="../config/cracks_config.json"
    )
    args = parser.parse_args()

    output_generator.prepare_output_path(args.output_dir_path)

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(args.output_dir_path, "cracks_mask.log"),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    global CRACKS_CONFIG
    CRACKS_CONFIG = json.load(open(args.config))
    json.dump(CRACKS_CONFIG, open(os.path.join(args.output_dir_path, "cracks_config_used.json"), "wt"))

    crack_mask(input_path=args.input_image_path, output_dir_path=args.output_dir_path)
