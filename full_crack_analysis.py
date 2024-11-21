import argparse
import os
import imageio.v3 as iio
import cracks.features
import cracks.derivates
import pandas as pd
import numpy as np
import phase_map
import matplotlib.pyplot as plt
import utils
from skimage.measure import label
import logging


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
    utils.cli_arguments.add_sample_name(argparse)
    utils.config_loader.add_argparse_argument_phase_config_path(argparse)
    utils.output_dir_generator.add_argparse_argument(argparse)
    args = argparse.parse_args()

    output_dir_path, output_file_prefix = utils.output_dir_generator.prepare_output_path(args.output_dir_path, args.sample_name)
    utils.configure_logger(os.path.join(output_dir_path, "computing-features.log"))
    logger = logging.getLogger(__name__)

    PHASES_CONFIG = utils.config_loader.Config.load_config(args.phase_config_path, "phases")
    utils.configure_logger(os.path.join(output_dir_path, f"{output_file_prefix}full_crack_analysis.log"))

    cracks_mask = iio.imread(args.cracks_mask_path)
    smooth_mask = cracks.derivates.smooth_mask(cracks_mask)

    if "crack_features" in args.features:
        features = cracks.features.compute(smooth_mask)
        pd.DataFrame(features).to_csv(os.path.join(output_dir_path, f"{output_file_prefix}cracks_features.csv"), index=False)

    if "phase_features" in args.features:
        assert args.phase_map_path is not None, "Please specify --grains-map-path (@see -h)"
        phase_map_img = iio.imread(args.phase_map_path).astype(int)
        phase_map_through, phase_map_edge, neigh_coords_per_px, branches_map = cracks.features.phase_analysis(smooth_mask, phase_map_img)
        phase_through_ids, phase_through_count = np.unique(phase_map_through, return_counts=True)
        phase_edge_pairs, phase_edge_count = np.unique(
            np.sort(phase_map_edge.reshape(-1,2), axis=1),
            axis=0,
            return_counts=True
        )

        #### Charts out:
        # edge phase stacked
        labels = np.array(PHASES_CONFIG["labels"])
        plt.figure(figsize=(7,7))
        plt.bar(labels[phase_through_ids[1:]], phase_through_count[1:], width=0.7)
        plt.title(f"Crack pixels through phase {args.sample_name}")
        plt.xlabel("Phase name")
        plt.ylabel("Pixel count")
        plt.savefig(os.path.join(output_dir_path, f"{output_file_prefix}phase-through.png"))
        #plt.show()
        plt.close()

        # Through phase bar
        plt.figure(figsize=(7, 7))
        bars_bottom = np.zeros_like(labels, dtype=int)

        affected_phases = np.unique(phase_edge_pairs)
        all_phases_used = np.arange(np.min(affected_phases), np.max(affected_phases) + 1)
        edge_counts = np.zeros((all_phases_used.size, all_phases_used.size))
        for phase_edge_pair, count in zip(phase_edge_pairs, phase_edge_count):
            phase_A_idx = np.where(all_phases_used == phase_edge_pair[0])[0][0]
            phase_B_idx = np.where(all_phases_used == phase_edge_pair[1])[0][0]
            edge_counts[phase_A_idx, phase_B_idx] += count
            edge_counts[phase_B_idx, phase_A_idx] += count

        bars_bottom = np.zeros((all_phases_used.size - 1,))
        for phase_order, phase_id in enumerate(all_phases_used):
            # first phase is invalid, skip it
            if phase_id < 0:
                continue
            plt.bar(np.array(labels), edge_counts[1:, phase_order], width=0.7, bottom=bars_bottom, label=labels[phase_id])
            bars_bottom += edge_counts[1:, phase_order]
        plt.title(f"Crack pixels through phase {args.sample_name}")
        plt.xlabel("Phase name")
        plt.ylabel("Pixel count")
        plt.legend()
        plt.savefig(os.path.join(output_dir_path, f"{output_file_prefix}phase-edge.png"))
        # plt.show()
        plt.close()

        #### CSV outs:
        # simple_counts.csv
        # - cracked_phases - phase_id, crack_px_count (aggregated in one table with edges)
        #     pd.DataFrame(
        #        np.array([phase_through_ids[1:], labels[phase_through_ids[1:]], phase_through_count[1:]]).T,
        #        columns=["phase id", "label", "through (px)"]).to_csv(
        #           os.path.join(output_dir_path, f"{output_file_prefix}phase-through.csv"), index=False)
        # - cracks_on_borders - border phase A, border phase B, crack_px_count (aggregated in one table with through)
        #     pd.DataFrame(
        #        np.concatenate([all_phases_used[1:].reshape(1, -1), labels[all_phases_used[1:]].reshape(1, -1), edge_counts[1:,1:].astype(int)], axis=0).T,
        #        columns=["phase id", "label"] + labels[all_phases_used[1:]].tolist()).to_csv(
        #           os.path.join(output_dir_path, f"{output_file_prefix}phase-edges.csv"), index=False)
        # - merge those two above into one table
        phases_cracks_count = edge_counts[1:, 1:]
        for phase_id, count in zip(phase_through_ids[1:], phase_through_count[1:]):
            phases_cracks_count[phase_id, phase_id] = count
        pd.DataFrame(
            np.concatenate([all_phases_used[1:].reshape(1, -1), labels[all_phases_used[1:]].reshape(1, -1), phases_cracks_count.astype(int)], axis=0).T,
            columns=["phase id", "label"] + labels[all_phases_used[1:]].tolist()).to_csv(
            os.path.join(output_dir_path, f"{output_file_prefix}phases_cracks.csv"), index=False)
        #    - cracked_grains - grain_id, crack_px_count

        # valid neigh coord are those who has both (left and right phase) with valid id
        valid_neigh_coords = np.where(
            np.logical_and(np.logical_and(neigh_coords_per_px[:, :, 0, 0] >= 0, neigh_coords_per_px[:, :, 0, 1] >= 0),
                           np.logical_and(neigh_coords_per_px[:, :, 1, 0] >= 0, neigh_coords_per_px[:, :, 1, 1] >= 0)))
        pd.DataFrame(np.stack([
            valid_neigh_coords[0], # skeleton x
            valid_neigh_coords[1], # skeleton y
            # TODO: Maybe remove crack ids not in valid neigh coords. Otherwise there are holes in indices
            label(smooth_mask, background=0)[valid_neigh_coords],  # crack id
            branches_map[valid_neigh_coords],  # branch id
            neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 0, 0],  # left x
            neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 0, 1],  # left y
            neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 1, 0],  # right x
            neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 1, 1],  # right y
            phase_map_img[ # left phase
                neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 0, 0],
                neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 0, 1],
            ],
            phase_map_img[  # right phase
                neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 0, 0],
                neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 0, 1],
            ]
        ], axis=1), columns=[
            "crack skeleton coord X",
            "crack skeleton coord Y",
            "crack ID",
            "branch ID",
            "left neighbor X",
            "left neighbor Y",
            "right neighbor X",
            "right neighbor Y",
            "left neighbor phase ID",
            "right neighbor phase ID"
        ]).to_csv(os.path.join(output_dir_path, f"{output_file_prefix}skeleton-neighbors.csv"), index=False)

    if "grain_features" in args.features:
        logger.error("Grain features not yet implemented.")
        pass
        # phases_edge_id = [phase_A, grain_A, phase_B, grain_B, count]
        # phases_through_id = [phase, grain, count]
