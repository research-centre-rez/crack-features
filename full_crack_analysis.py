import argparse
import os
import imageio.v3 as iio
import cracks.features
import cracks.derivates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from skimage.measure import label
import logging


def through_chart(phase_through_ids, phase_through_count, labels, sample_name, output_file_prefix):
    """
    Draws a bar chart where cracks (px) are shown for each phase
    @param phase_through_ids: List of indices representing phases to plot.
    @param phase_through_count: List of counts corresponding to each phase.
    @param labels: List of phase names corresponding to the provided indices.
    @param sample_name: Name of the sample to be included in the plot title.
    @param output_file_prefix: Prefix for the output file name where the plot will be saved.
    @return: None
    """
    plt.figure(figsize=(7, 7))
    plt.bar(labels[phase_through_ids[1:]], phase_through_count[1:], width=0.7)
    plt.title(f"Crack pixels through phase {sample_name}")
    plt.xlabel("Phase name")
    plt.ylabel("Pixel count")
    plt.savefig(os.path.join(output_dir_path, f"{output_file_prefix}phase-through.png"))
    plt.close()


def edge_stacked_chart(phase_edge_pairs, phase_edge_count, labels, sample_name, output_file_prefix):
    """
    Draws a bar chart where cracks (in px) are shown for each pair of phases. Cracks are between these two pahses.

    @param phase_edge_pairs: A list of tuples where each tuple contains a pair of phase indices representing an edge.
    @param phase_edge_count: A list of counts corresponding to each phase edge pair, representing the number of occurrences of each edge.
    @param labels: A list of labels for each phase to be used in the chart.
    @param sample_name: A string representing the name of the sample, used in the chart title.
    @param output_file_prefix: A string prefix for the output filename of the chart image.

    @return: A tuple containing:
      - A numpy array of all phases in the configuration.
      - A numpy array representing the edge counts (in pixels) between two different phases.
    """
    affected_phases = np.unique(phase_edge_pairs)
    all_phases_used = np.arange(np.min(affected_phases), np.max(affected_phases) + 1)
    edge_counts = np.zeros((all_phases_used.size, all_phases_used.size))
    for phase_edge_pair, count in zip(phase_edge_pairs, phase_edge_count):
        phase_A_idx = np.where(all_phases_used == phase_edge_pair[0])[0][0]
        phase_B_idx = np.where(all_phases_used == phase_edge_pair[1])[0][0]
        edge_counts[phase_A_idx, phase_B_idx] += count
        edge_counts[phase_B_idx, phase_A_idx] += count

    bars_bottom = np.zeros((all_phases_used.size - 1,))
    plt.figure(figsize=(7, 7))
    for phase_order, phase_id in enumerate(all_phases_used):
        # first phase is invalid, skip it
        if phase_id < 0:
            continue
        plt.bar(np.array(labels), edge_counts[1:, phase_order], width=0.7, bottom=bars_bottom, label=labels[phase_id])
        bars_bottom += edge_counts[1:, phase_order]
    plt.title(f"Crack neighbors - phase pairs {sample_name}", fontsize="xx-large")
    plt.xlabel("Phase name", fontsize="xx-large")
    plt.ylabel("Pixel count", fontsize="xx-large")
    plt.legend(fontsize="xx-large")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_path, f"{output_file_prefix}phase-edge.png"), transparent=True)
    plt.close()

    return all_phases_used, edge_counts


def phases_cracks_to_csv(labels, edge_counts, phase_through_ids, phase_through_count, all_phases_used, output_dir_path, output_file_prefix):
    """
    Method dumps simple matrix where columns and rows correspond to phases and each cell in this table contains number
    of pixels belonging to cracks between this pair of phases.

    @param labels: An array of labels representing different phases.
    @param edge_counts: A 2D array where each element represents the count of edges/cracks between phases.
    @param phase_through_ids: An array of phase IDs that passed through some processing.
    @param phase_through_count: An array of counts corresponding to how many times each phase ID passed through processing.
    @param all_phases_used: An array indicating all phases used in the process.
    @param output_dir_path: The directory path where the resulting CSV file will be saved.
    @param output_file_prefix: The prefix for the output file name.
    @return: None
    """
    phases_cracks_count = edge_counts[1:, 1:]
    for phase_id, count in zip(phase_through_ids[1:], phase_through_count[1:]):
        phases_cracks_count[phase_id, phase_id] = count
    pd.DataFrame(
        np.concatenate([all_phases_used[1:].reshape(1, -1), labels[all_phases_used[1:]].reshape(1, -1),
                        phases_cracks_count.astype(int)], axis=0).T,
        columns=["phase id", "label"] + labels[all_phases_used[1:]].tolist()).to_csv(
        os.path.join(output_dir_path, f"{output_file_prefix}phases_cracks.csv"), index=False)


def skeleton_neighbors_to_csv(gradient_map, neigh_coords_per_px, smooth_mask, branches_map, phase_map_img,
                              output_dir_path, output_file_prefix):
    """
    Overview of the evaluation. Each cracks-skeleton-pixel in this table is here with corresponding:
     - crack id
     - crack branch id
     - neigboring phases
     - neigbours coordinates

    @param neigh_coords_per_px: A numpy array containing the coordinates of neighbors for each pixel in the skeleton image.
    @param smooth_mask: A binary mask image that identifies the cracks in the input image.
    @param branches_map: A numpy array mapping branch IDs for pixels in the skeleton image.
    @param phase_map_img: A numpy array representing the phase map image with phase IDs.
    @param output_dir_path: The directory path where the output CSV file will be saved.
    @param output_file_prefix: The prefix for the output file name.
    @return: None. This function saves a CSV file containing the skeleton neighbors and their attributes.
    """
    # valid neigh coord are those who has both (left and right phase) with valid id
    valid_neigh_coords = np.where(
        np.logical_and(np.logical_and(neigh_coords_per_px[:, :, 0, 0] >= 0, neigh_coords_per_px[:, :, 0, 1] >= 0),
                       np.logical_and(neigh_coords_per_px[:, :, 1, 0] >= 0, neigh_coords_per_px[:, :, 1, 1] >= 0)))
    pd.DataFrame(np.stack([
        valid_neigh_coords[1],  # skeleton x
        valid_neigh_coords[0],  # skeleton y
        # TODO: Maybe remove crack ids not in valid neigh coords. Otherwise there are holes in indices
        label(smooth_mask, background=0)[valid_neigh_coords],  # crack id
        branches_map[valid_neigh_coords],  # branch id
        gradient_map[valid_neigh_coords[0], valid_neigh_coords[1], 0],
        gradient_map[valid_neigh_coords[0], valid_neigh_coords[1], 1],
        neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 0, 0],  # left x
        neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 0, 1],  # left y
        neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 1, 0],  # right x
        neigh_coords_per_px[valid_neigh_coords[0], valid_neigh_coords[1], 1, 1],  # right y
        phase_map_img[  # left phase
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
        "gradient X",
        "gradient Y",
        "left neighbor X",
        "left neighbor Y",
        "right neighbor X",
        "right neighbor Y",
        "left neighbor phase ID",
        "right neighbor phase ID"
    ]).to_csv(os.path.join(output_dir_path, f"{output_file_prefix}skeleton-neighbors.csv"), index=False)


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
        pd.DataFrame(features).to_csv(
            os.path.join(output_dir_path, f"{output_file_prefix}cracks_features.csv"),
            index=False
        )

    if "phase_features" in args.features:
        assert args.phase_map_path is not None, "Please specify --grains-map-path (@see -h)"
        phase_map_img = iio.imread(args.phase_map_path).astype(int)
        phase_map_through, phase_map_edge, neigh_coords_per_px, branches_map, gradient_map = cracks.features.phase_analysis(
            smooth_mask, phase_map_img
        )
        phase_through_ids, phase_through_count = np.unique(phase_map_through, return_counts=True)
        phase_edge_pairs, phase_edge_count = np.unique(
            np.sort(phase_map_edge.reshape(-1,2), axis=1),
            axis=0,
            return_counts=True
        )

        #### Charts out:
        labels = np.array(PHASES_CONFIG["labels"])

        through_chart(phase_through_ids, phase_through_count, labels, args.sample_name, output_file_prefix)
        all_phases_used, edge_counts = edge_stacked_chart(phase_edge_pairs, phase_edge_count, labels, args.sample_name, output_file_prefix)

        #### CSV outs:
        phases_cracks_to_csv(
            labels,
            edge_counts,
            phase_through_ids,
            phase_through_count,
            all_phases_used,
            output_dir_path,
            output_file_prefix
        )

        skeleton_neighbors_to_csv(
            gradient_map,
            neigh_coords_per_px,
            smooth_mask,
            branches_map,
            phase_map_img,
            output_dir_path,
            output_file_prefix
        )
        # TODO: cracked_grains - grain_id, crack_px_count


    if "grain_features" in args.features:
        logger.error("Grain features not yet implemented.")
        pass
        # phases_edge_id = [phase_A, grain_A, phase_B, grain_B, count]
        # phases_through_id = [phase, grain, count]
