import argparse
import json
import imageio.v3 as iio
import numpy as np
from skimage.color import rgb2gray
import cv2
import utils.image_logger as image_logger
import utils.output_dir_generator as output_generator
from tqdm.auto import tqdm
import os
import utils


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


def normalize_histogram_left_and_right(img, new_position_of_histogram_max):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    img = clahe.apply(img)
    out = np.copy(img)
    histogram_max_pos = np.argmax(np.histogram(img.reshape(-1), bins=256)[0])
    out[out <= histogram_max_pos] = img[img <= histogram_max_pos] / histogram_max_pos * new_position_of_histogram_max
    out[out > histogram_max_pos] = (out[out > histogram_max_pos] - histogram_max_pos) / (255 - histogram_max_pos) * (255 - new_position_of_histogram_max) + new_position_of_histogram_max
    return out


def crack_mask_by_thresholds(input_path, output_dir_path):
    """
    This method uses CRACKS_CONFIG global setup. If you want to change behavior edit this config file.
    Thresholds are applied as follows first threshold defined is used as "seeds", second threshold is used as
    "seed potential" so every pixel belonging to the 2nd threshold is marked as crack iff lay next to seed
    or transitively lay next to "seed potential" already marked as crack.

    @param input_path: input image which will be thresholded.
    @param output_dir_path: output dir path where outputs will be stored (debug with prefix [user])
    @return: crack binary mask
    """

    # Load input image
    gray_img = _load_input(input_path)
    image_logger.info(gray_img, output_dir_path, "[user]input-grayscale.png")

    # Normalize histogram
    gray_norm = normalize_histogram_left_and_right(gray_img, CRACKS_CONFIG["median_level"])
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

    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Preprocessing of the crack scans. Target of this routine is to create crack mask.
    """)
    parser.add_argument(
        "input_image_path",
        type=str,
        help="Path to grayscale input image where cracks should be detected.")
    output_generator.add_argparse_argument(parser)
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
    utils.configure_logger(os.path.join(args.output_dir_path, "cracks_mask.log"))

    global CRACKS_CONFIG
    CRACKS_CONFIG = json.load(open(args.config))
    json.dump(CRACKS_CONFIG, open(os.path.join(args.output_dir_path, "cracks_config_used.json"), "wt"))

    crack_mask_by_thresholds(input_path=args.input_image_path, output_dir_path=args.output_dir_path)
