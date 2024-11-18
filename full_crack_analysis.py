import argparse
import os
import imageio.v3 as iio
import cracks.features
import cracks.derivates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from phase_map.cleanup import PHASES_CONFIG
from utils import output_dir_generator, configure_logger

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="""
    Toolbox for analysis of cracks in the microscopic sample.
    """)

    argparse.add_argument(
        "cracks_mask_path",
        type=str,
        help="Path to the cracks mask file."
    )
    # This should split features according to performance requirements
    argparse.add_argument(
        "-f",
        "--features",
        action="append",
        type=str,
        required=True,
        help="Specify features you want measure: [crack_features, phase_features]"
    )
    argparse.add_argument(
        "--phase-map-path",
        type=str,
        help=f"Path to matching phase map file. Required if feature phase_features is enabled. "
             f"Filtered phase map can be created by phase_map module.",
    )
    argparse.add_argument(
        "--sample-name",
        type=str,
        help="The name of the sample will be used as prefix for all produced outputs."
    )
    output_dir_generator.add_argparse_argument(argparse)
    args = argparse.parse_args()

    output_dir_generator.prepare_output_path(args.output_dir_path)
    configure_logger(os.path.join(args.output_dir_path, "full_crack_analysis.log"))

    cracks_mask = iio.imread(args.cracks_mask_path)
    smooth_mask = cracks.derivates.smooth_mask(cracks_mask)

    if "crack_features" in args.features:
        features = cracks.features.compute(smooth_mask)
        pd.DataFrame(features).to_csv(os.path.join(args.output_dir_path, "cracks_features.csv"), index=False)

    if "phase_features" in args.features:
        assert args.phase_map_path is not None, "Please specify --grains-map-path (@see -h)"
        phase_map = iio.imread(args.phase_map_path).astype(int)
        phase_map_through, phase_map_edge = cracks.features.phase_analysis(smooth_mask, phase_map)
        phase_through_ids, phase_through_count = np.unique(phase_map_through, return_counts=True)
        phase_edge_pairs, phase_edge_count = np.unique(
            np.sort(phase_map_edge.reshape(-1,2), axis=1),
            axis=0,
            return_counts=True
        )

        #### Charts out:
        # edge phase stacked
        labels = np.array(PHASES_CONFIG["labels"])
        plt.figure(figsize=(15,15))
        plt.bar(labels[phase_through_ids[1:]], phase_through_count[1:], width=0.7)
        plt.title(f"Crack pixels through phase {args.sample_name}")
        plt.xlabel("Phase name")
        plt.ylabel("Pixel count")
        plt.savefig(os.path.join(args.output_dir_path, f"{args.sample_name}-phase-through.png"))
        plt.show()

        # through phase bar
        plt.figure(figsize=(15, 15))
        bottom = np.zeros_like(labels)
        # TODO: reformat phase_edge_pairs into bars
        for phase_edge_pair, count in zip(phase_edge_pairs, phase_edge_count):
            plt.bar(np.array(labels)[phase_through_ids[1:]], phase_through_count[1:], width=0.7, bottom=bottom)
            bottom += count
        plt.title(f"Crack pixels through phase {args.sample_name}")
        plt.xlabel("Phase name")
        plt.ylabel("Pixel count")
        plt.savefig(os.path.join(args.output_dir_path, f"{args.sample_name}-phase-through.png"))
        plt.show()

        #### CSV outs:
        # simple_counts.csv
        #    - cracked_grains - grain_id, crack_px_count
        #    - cracked_phases - phase_id, crack_px_count
        #    - cracks_on_borders - border phase A, border phase B, crack_px_count
        # crack_id -> branch_id -> phases_through_id
        # crack_id -> branch_id -> phases_edge_id
        # phases_edge_id = [phase_A, grain_A, phase_B, grain_B, count]
        # phases_through_id = [phase, grain, count]

        #### Matrices (tif? 16bit)
        # left_neighbor[x]
        # left_neighbor[y]
        # left_neighbor_phase_id
        # left_neighbor_grain_id
        # right_neighbor[x]
        # right_neighbor[y]
        # right_neighbor_phase_id
        # right_neighbor_grain_id
