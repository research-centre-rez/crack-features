import argparse
import os
import imageio.v3 as iio
import phase_map.cleanup
import phase_map.grains
import utils
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Script prepares phase map for further analysis:
    - the phase map is segmented into classes defined in phases_config.json
    - continuous areas (grains) are defined
    - unclear and small areas are removed and attached to the closest grain or matrix phase.
    
    Select raw phase image: [phase_img_path]
    Select corresponding crack_map: [crack_mask_path]
    
    Define phase colors config file path: [--config]
    Configure phase threshold function: [--gaussian_blur, --adjust_intensity, --lightness_clip, --distance_metric]    
    """)
    parser.add_argument(
        "phase_img_path",
        type=str,
        help="Path to phase image.")
    parser.add_argument(
        "crack_mask_path",
        type=str,
        help="Path to crack mask.")
    utils.output_dir_generator.add_argparse_argument(parser)
    utils.config_loader.add_argparse_argument_phase_config_path(parser)
    parser.add_argument(
        '--distance_metric',
        type=str,
        required=False,
        help=f"Select one of these [lab, ab] if you want use euclidean distance in La*b* color space (a*b* respectively)."
             f"If not specified CIE DE2000 is used (@see https://en.wikipedia.org/wiki/Color_difference#CIEDE2000)",
        default=None
    )
    parser.add_argument(
        '--gaussian_blur',
        type=float,
        required=False,
        help="Apply gaussian blur on phase image before segmentation (with given sigma value)",
        default=0)
    parser.add_argument(
        '--adjust_intensity',
        type=str,
        required=False,
        default="image",
        help="""Adjust the image intensity from specified range. Format should be "min, max".""")
    parser.add_argument(
        '--lightness_clip',
        nargs='+',
        type=int,
        required=False,
        help="Set thresholds. Takes 1 or 2 values, separated by comma.",
        default=[1, 99])
    utils.cli_arguments.add_sample_name(parser)
    args = parser.parse_args()

    output_dir_path, output_file_prefix = utils.output_dir_generator.prepare_output_path(args.output_dir_path, args.sample_name)
    utils.configure_logger(os.path.join(output_dir_path, f"{output_file_prefix}phase_map.log"))
    utils.config_loader.Config.load_config(args.phase_config_path, "phases")
    PHASES_CONFIG = utils.config_loader.Config.get_config("phases")

    phase_img = iio.imread(args.phase_img_path)
    crack_mask = iio.imread(args.crack_mask_path)

    phase_map_img = phase_map.cleanup.apply_threshold(
        phase_map=phase_img,
        config=PHASES_CONFIG,
        lightness_clip=args.lightness_clip,
        adjust_intensity=args.adjust_intensity,
        gaussian_blur=args.gaussian_blur,
        output_dir=output_dir_path,
        output_file_prefix=output_file_prefix
    )

    # segment single grains
    grains_map = phase_map.grains.segment_grains(
        phase_map_img,
        output_dir_path=output_dir_path,
        output_file_prefix=output_file_prefix
    )
    # filter out small grains
    grains_map, _, matrix_ids_start = phase_map.grains.grain_size_filter(
        grains_map,
        output_dir_path=output_dir_path,
        output_file_prefix=output_file_prefix
    )
    # create phase map where small grains are attached to a matrix phase
    phase_map_filtered = np.copy(phase_map_img)
    phase_map_filtered[grains_map >= matrix_ids_start] = np.where(np.array(PHASES_CONFIG["labels"]) == "Matrix")[0][0]
    for l, label in enumerate(PHASES_CONFIG["labels"]):
        utils.image_logger.dump_image(
            os.path.join(output_dir_path, f"{output_file_prefix}[user]phase_map-{label}.png"),
            (phase_map_filtered == l).astype(np.uint8) * 255)
    utils.image_logger.dump_image(os.path.join(output_dir_path, f"{output_file_prefix}phase_map_filtered.png"), phase_map_filtered.astype(np.uint8))
