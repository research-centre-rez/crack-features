import numpy as np
import imageio.v3 as iio
from skimage.color import rgb2lab, rgb2gray
from skimage.filters import gaussian
from skimage.measure import label
from skimage.exposure import rescale_intensity
import argparse
import logging
import re

logger = logging.getLogger(__name__)

LIGHTNESS_LIMITS = "1,99"

def phase_threshold(image, phases, args):
    def imgaussfilt(img, sigma):
        return gaussian(img, sigma=sigma, channel_axis=2)

    # Emulate MATLAB's `imadjust` using `skimage`'s `rescale_intensity`
    def imadjust(img, limits=None):
        """
        Adjust the image intensity (grayscale).
        For uint8 images the output intensity levels are in range <0, 255>.
        For float images the output intensity levels are in range <0, 1>.
        When limits are specified, the intensity levels are rescaled to the given limits.
        """
        if limits is None:
            return rescale_intensity(img)

        # format "min, max"
        in_range = tuple([float(value)
                          for value in re.findall(r"([0-9\.]+)", limits)])
        return rescale_intensity(img, in_range=in_range)


    verbose = []

    # Extract phases to be detected
    phases_selected = [phase for phase in phases[0] if phase['Detect'][0][0]]
    cracks_image = args.cracks
    verbose.append({"command": "cracks", "img": cracks_image})
    phases_count = len(phases_selected)

    # Set the output lightness limits.
    if isinstance(args.threshold, str):  # there can be one or two values
        lightness_limits = sorted([float(value.strip(" ")) for value in args.threshold.split(",")])
    else:
        lightness_limits = [args.threshold]

    verbose.append({"command": "threshold", "img": image})
    assert 0 < len(lightness_limits) < 3

    # Rescale image intensity into specified range
    img = imadjust(image, args.adjust_intensity)
    verbose.append({"command": f"adjust_intensity={args.adjust_intensity}", "img": img})

    if args.gaussian_blur != 0:
        img = imgaussfilt(img, args.gaussian_blur)
        verbose.append({"command": f"gaussian_blur={args.gaussian_blur}", "img": img})

    cracks_image = cracks_image.astype(float)
    img_lab = rgb2lab(img)
    phases_color_lab = rgb2lab(np.array([phase['Colors'] for phase in phases_selected]))

    # TODO: Use CIE DE2000 instead of eucledian distance here
    img_without_cracks_lab = np.copy(img_lab)
    img_without_cracks_lab[cracks_image.astype(bool), :] = 0
    if args.lab_distance:
        verbose.append({"command": "lab_distance", "img": img_without_cracks_lab})
        distances = np.array([
            np.linalg.norm(img_without_cracks_lab.reshape(-1, 3) - phase_color_lab, axis=1)
            for phase_color_lab in phases_color_lab
        ]).T.reshape(img_lab.shape[0], img_lab.shape[1], phases_count)
    elif args.ab_distance:
        verbose.append({"command": "ab_distance", "img": img_without_cracks_lab})
        distances = np.array([
            np.linalg.norm(img_without_cracks_lab[:,:,1:].reshape(-1, 2) - phase_color_lab[1:])
            for phase_color_lab in phases_color_lab
        ]).reshape(img_lab.shape[0], img_lab.shape[1], phases_count)
    else:
        raise Exception("No distance computation specified.")

    phase_map_img = np.argmin(distances, axis=2)
    phase_map_img[img_without_cracks_lab[:,:,0] < lightness_limits[0]] = 0
    if len(lightness_limits) > 1:
        phase_map_img[img_without_cracks_lab[:,:,0] > lightness_limits[1]] = 0

    layers = []
    for i in range(phases_count):
        phase_mask = (phase_map_img == i)
        layers.append({
            'mask': phase_mask,
            'label': phases_selected[i]['Labels'],
            'rgb': phases_selected[i]['Colors'],
            'pixel_count': np.sum(phase_mask),
            'pixel_ratio': np.sum(phase_mask) / phase_mask.size,
            'segments_count': np.max(label(phase_mask))
        })
    nophase_mask = phase_map_img != 0
    layers.append({
        'mask': nophase_mask,
        'Label': 'n/a',
        'rgb': np.zeros(3),
        'pixel_count': np.sum(nophase_mask),
        'pixel_ratio': np.sum(nophase_mask) / nophase_mask.size,
        'segments_count': np.max(label(nophase_mask))
    })
    return phase_map_img, layers, verbose


# # Example usage placeholder for IMG and phases
# img = np.random.rand(100, 100, 3)  # Randomly generated sample image
# phases = [{'Detect': True, 'Colors': [255, 0, 0], 'Labels': 'Phase 1'},
#           {'Detect': True, 'Colors': [0, 255, 0], 'Labels': 'Phase 2'},
#           {'Detect': False, 'Colors': [0, 0, 255], 'Labels': 'Phase 3'}]
# Execute

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Configure arguments for phase threshold function. 
    AB denotes a* and b* dimensions in La*b* color space. 
    One of --ab_distance or --lab_distance must be set.
    """)

    parser.add_argument(
        '--ab_distance',
        action='store_true',
        required=False,
        help="Set AB distance calculation to False")
    parser.add_argument(
        '--lab_distance',
        action='store_true',
        required=False,
        help="Set La*b* distance calculation to True")
    parser.add_argument(
        '--gaussian_blur',
        type=float,
        required=False,
        help="Apply gaussian filter with given sigma value",
        default=0)
    parser.add_argument(
        '--adjust_intensity',
        type=str,
        required=False,
        help="""Adjust the image intensity from specified range. 
        Format should be "min, max".""")
    parser.add_argument(
        '-t',
        '--threshold',
        type=str,
        required=False,
        help="Set thresholds. Takes 1 or 2 values, separated by comma.",
        default=LIGHTNESS_LIMITS)
    parser.add_argument(
        '--cracks',
        type=str,
        required=False,
        help="Path to crack mask image",
        default=None
    )

    args = parser.parse_args()

    if args.cracks is None:
        # Use empty image when no crack map is linked
        args.cracks = np.zeros_like(img[:, :, 0])
    else:
        args.cracks = rgb2gray(iio.imread(args.cracks))

    map, layers, verbose = phase_threshold(img, phases, args)
