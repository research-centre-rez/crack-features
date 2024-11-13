import logging
import imageio.v3 as iio
import os
import numpy as np

logger = logging.getLogger(__name__)


def dump_image(file_path, img):
    # Normalize image
    if img.dtype != np.uint8:
        logger.warning(f"{file_path.split(os.path.sep)[-2]}: Output image do not have requested format (uint8). Converting to uint8, range stretched.")
        img = (((img - np.min(img))/(np.max(img) - np.min(img))) * 255).astype(np.uint8)
    iio.imwrite(file_path, img)


def debug(img, folder_path, filename):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Dumping image to {filename}")
        dump_image(os.path.join(folder_path, filename), img)


def info(img, folder_path, filename):
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Dumping image to {filename}")
        dump_image(os.path.join(folder_path, filename), img)


def warning(img, folder_path, filename):
    if logger.isEnabledFor(logging.WARNING):
        logger.warning(f"Dumping image to {filename}")
        dump_image(os.path.join(folder_path, filename), img)


def error(img, folder_path, filename):
    if logger.isEnabledFor(logging.ERROR):
        logger.error(f"Dumping image to {filename}")
        dump_image(os.path.join(folder_path, filename), img)
