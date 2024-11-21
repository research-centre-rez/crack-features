import numpy as np
import skimage.color
from skimage.color import rgb2lab
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
import os
import logging
import json
import utils.image_logger as image_logger
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _phases_selected(key, config):
    """
    From PHASE CONFIG filter out records, which should not be detected
    """
    return [
        value
        for value, should_detect in zip(config[key], config["detect"])
        if should_detect
    ]


def apply_threshold(phase_map, config, lightness_clip=[1, 99], adjust_intensity=None, gaussian_blur=0, distance_metric=None, output_dir=None, output_file_prefix=""):
    """
    For raw phase image do cleanup. Cleanup contains:
    - rescale of the intensity level (in RGB) according to adjust_intensity (@see skimage.exposure.rescale_intensity)
    - blur accodring to gaussian_blur set to gaussian sigma (@see skimage.filters.gaussian_blur)
    - lightness clipping by lightness_clip
    After these preprocessing steps image is converted to La*b* color space and the nearest color id from the
    PHASE_CONFIG["colors"] is put to the return value.

    @param phase_map: RGB image colored according to the segments found
    @param config: PHASE CONFIG dict
    @param lightness_clip: Clip the intensity range to lightness. Could contain one or two value tuple. If single value
        is present it is used as the lower threshold.
    @param adjust_intensity: None or tuple defines limits for stretching of the histogram
        (@see in_range parameter of skimage.exposure.rescale_intensity)
    @param gaussian_blur: Gaussian blur applied to input image
    @param distance_metric: Distance metric to use for distance computation
    @param output_dir: Directory to save the output image
    @return 2D array with dimensions <height, width> of input image. Value of each "pixel" is set to id of a pahse in
        the PHASE_CONFIG["colors"]
    """
    json.dump(config, open(os.path.join(output_dir, f"{output_file_prefix}phases_config_used.json"), "wt"))
    image_logger.info(phase_map, output_dir, f"{output_file_prefix}phase_map_rgb.png")

    # Rescale image intensity into specified range
    img = rescale_intensity(phase_map, in_range=adjust_intensity)
    image_logger.info(img, output_dir, f"{output_file_prefix}phase_map_adjusted.png")

    if gaussian_blur != 0:
        img = gaussian(img, sigma=gaussian_blur, channel_axis=2)
        image_logger.info(img, output_dir, f"{output_file_prefix}phase_map_gaussian_blur.png")

    img_lab = rgb2lab(img)
    phases_color_rgb = np.array(_phases_selected("colors", config)).astype(np.uint8)
    phases_color_lab = rgb2lab(phases_color_rgb)
    image_logger.info(img_lab, output_dir, f"{output_file_prefix}phase_img_lab.png")

    # Attach pixels to phases
    distances = color_distance(img_lab, phases_color_lab, mode=distance_metric)
    phase_map_img = np.argmin(distances, axis=2)

    # clip values out of lightness_limit range
    # TODO: Prochazka - why there is set to zero? Shouldn't this be "Matrix" value?
    phase_map_img[img_lab[:,:,0] < lightness_clip[0]] = 0
    if len(lightness_clip) > 1:
        phase_map_img[img_lab[:,:,0] > lightness_clip[1]] = 0

    # TODO: Original algorithm removes cracks from this map (phase set to 0)
    image_logger.info(phases_color_rgb[phase_map_img], output_dir, f"{output_file_prefix}[user]phase_map.png")
    image_logger.dump_image(os.path.join(output_dir, f"{output_file_prefix}phase_map.png"), phase_map_img.astype(np.uint8))

    return phase_map_img


def color_distance(img_lab, reference_colors_lab, mode=None):
    """
    This method goes thru all image pixels and measure the distance of the color coordinates to each of reference colors
    stored in reference_colors. The result is array with same resolution as original image and 3rd dimension references
    the reference_colors (for each of them there is one computed distance scalar).

    Color distance measurement highly depends on img_lab construction. In principle this can be lightness based, hue
    based or any other. We assume, that the purpose of img_lab coloring is to visualize to the user different segments.
    For this reason we work in La*b* coordinates and using various metrics @see variable mode

    @param img_lab contains 3D array with dimensions <height, width, La*b*>
    @param reference_colors_lab contains 2D array with dimensions <color, La*b*>
    @param mode str
        defines mode for distance computation, by default CIE DE2000 is used,
        but La*b*-Euclidean as well as a*b*-Euclidean can be used.
    """
    if mode in ["lab", "La*b*-Euclidean"]:
        distances = np.array([
            np.linalg.norm(img_lab.reshape(-1, 3) - phase_color_lab, axis=1)
            for phase_color_lab in reference_colors_lab
        ]).T.reshape(img_lab.shape[0], img_lab.shape[1], len(reference_colors_lab))
    elif mode in ["ab", "a*b*-Euclidean"]:
        distances = np.array([
            np.linalg.norm(img_lab[:, :, 1:].reshape(-1, 2) - phase_color_lab[1:])
            for phase_color_lab in reference_colors_lab
        ]).reshape(img_lab.shape[0], img_lab.shape[1], len(reference_colors_lab))
    else:
        distances = np.stack([
            skimage.color.deltaE_ciede2000(reference_color_lab,
                                           img_lab.reshape(-1, 3).T).reshape(img_lab.shape[:2])
            for reference_color_lab in tqdm(reference_colors_lab,
                                            total=len(reference_colors_lab),
                                            desc="Computing CIE DE2000 per phase")], axis=2)
    return distances

