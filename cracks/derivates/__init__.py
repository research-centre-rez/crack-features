from argparse import ArgumentParser
from skimage.morphology import binary_dilation, disk, medial_axis, closing

from utils import configure_logger
import utils.image_logger as image_logger
import utils.output_dir_generator as output_generator
from cracks.derivates.bw_morph import endpoints as _endpoints, branches as _branches

import numpy as np
import logging
import os
import imageio.v3 as iio
import pandas as pd


def smooth_mask(crack_mask, smooth_radius_px=5):
    if smooth_radius_px == 0:
        return crack_mask
    # TODO: Discuss this with Jan Prochazka opening/closing/dilation
    return closing(crack_mask, footprint=disk(smooth_radius_px))


def skeleton(crack_mask_smooth):
    return medial_axis(crack_mask_smooth, return_distance=True)


def skeleton_endpoints(crack_skeleton):
    endpoints_mask = _endpoints(crack_skeleton)
    endpoint_coords = np.column_stack(np.where(endpoints_mask.T))

    return endpoints_mask, endpoint_coords


def skeleton_branches(cracks_skeleton):
    """
    Split skeleton into 1px wide paths.
    @param cracks_skeleton: mask of the crack skeleton (1px wide)
    @return: mask of skeleton branches, mask of skeleton nodes and coordinates of skeleton nodes
    """
    skeleton_nodes = _branches(cracks_skeleton)
    skeleton_nodes_coords = np.column_stack(np.where(skeleton_nodes.T))

    # Create a structuring element (disk with radius 2)
    structuring_element = disk(2)

    # Perform binary dilation on bw_skeleton_nodes
    dilated_nodes = binary_dilation(skeleton_nodes, structuring_element)

    # Subtract the dilated image from bw_skeleton and check where it's greater than 0
    skeleton_branches = np.logical_and(cracks_skeleton, np.logical_not(dilated_nodes))

    return skeleton_branches, skeleton_nodes, skeleton_nodes_coords


if __name__ == '__main__':
    argparse = ArgumentParser(description="""
    Module process cracks mask into derivates, i.e.: skeleton branches mask, skeleton endpoints mask and skeleton 
    nodes mask. Endpoints and nodes are also stored in form of coordinates.
    """)

    argparse.add_argument(
        "crack_mask_img_path",
        type=str,
        help="Path to the input file. It is expected an grayscale image with binary values {0,1} or {0,255}"
    )
    argparse.add_argument(
        "-o",
        "--output-dir-path",
        type=str,
        help="Path to output directory. If directory does not exist, it will be created (but parent directory must exist)."
    )
    argparse.add_argument(
        "-s",
        "--mask-smoothing",
        type=int,
        help=f"Before processing the crack mask can be smoothed by application of dilation (which produces less hairy "
             f"skeletons. The value is reasonable between 0 and width of a median skeleton in pixels).",
        default=5
    )
    args = argparse.parse_args()

    output_generator.prepare_output_path(args.output_dir_path)
    configure_logger(os.path.join(args.output_dir_path, "crack_derivates.log"))

    logging.debug(f"Reading input image {args.crack_mask_img_path}")
    cracks_mask = iio.imread(args.crack_mask_img_path)

    logging.debug(f"Smoothing crack mask {args.mask_smoothing}.")
    cracks_mask_smooth = smooth_mask(args.mask_smoothing)

    logging.debug("Creating skeletons of cracks.")
    crack_skeleton_mask, skeleton_to_boundary_distance = skeleton(cracks_mask_smooth)
    image_logger.dump_image(os.path.join(args.output_dir_path, "cracks-skeletons.png"), crack_skeleton_mask.astype(np.uint8))
    image_logger.dump_image(os.path.join(args.output_dir_path, "cracks-skeletons-boundary-distances.tif"), skeleton_to_boundary_distance.astype(np.uint16))

    skeletons_overview = np.zeros(cracks_mask.shape + (3,))
    skeletons_overview[cracks_mask.astype(bool), :] = np.array([255, 255, 255])
    skeletons_overview[crack_skeleton_mask.astype(bool), :] = np.array([255, 0, 0])
    image_logger.info(skeletons_overview, args.output_dir_path, "[user]cracks-skeletons-overview.png")

    logging.debug("Extracting skeleton endpoints.")
    cracks_endpoints_derivates = skeleton_endpoints(crack_skeleton_mask)
    image_logger.dump_image(os.path.join(args.output_dir_path, "cracks-skeletons-endpoints-mask.png"),
                            cracks_endpoints_derivates[0].astype(np.uint8))
    pd.DataFrame(cracks_endpoints_derivates[1], columns=["x", "y"]).to_csv(
        os.path.join(args.output_dir_path, "cracks-skeletons-endpoints-coordinates.csv"), index=False)

    logging.debug("Cutting skeletons into branches.")
    cracks_branches_derivates = skeleton_branches(crack_skeleton_mask)
    image_logger.dump_image(os.path.join(args.output_dir_path, "cracks-skeletons-branches-mask.png"),
                            cracks_branches_derivates[0].astype(np.uint8))
    image_logger.dump_image(os.path.join(args.output_dir_path, "cracks-skeletons-nodes-mask.png"),
                            cracks_branches_derivates[1].astype(np.uint8))
    pd.DataFrame(cracks_branches_derivates[2], columns=["x", "y"]).to_csv(
        os.path.join(args.output_dir_path, "cracks-skeletons-nodes-coordinates.csv"), index=False)
