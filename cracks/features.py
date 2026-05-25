from skimage.measure import label
from skimage.segmentation import mark_boundaries
import numpy as np
from tqdm.auto import tqdm
from cracks.branches import sort_branch_pixels, path_direction
import cracks.derivates
import pandas as pd
import matplotlib.pyplot as plt
from cracks.projection import adaptive_closing, heal_chipped_mask
from cracks.segment import pipeline_geometric
import os
import gc
import logging

logger = logging.getLogger(__name__)


def attach_neigh_phases_to_skeleton_branch(branch_mask, phase_map):
    """
    For each point of a branch goes in the direction perpendicular to the skeleton in-point derivative and looks for
    phase in the phase map. Output contains -1 for invalids and phase_id for the pixels where phase were found.

    @param branch_struct contains points belonging to the branch
    @param phase_map - phase image has for each pixel a phase ID attached (but cracks has zero)
    @return list of phase-pairs for each branch point
            list of point-pairs i.e. neighbors belonging to branch point
    """
    branch_ordered_pixels = sort_branch_pixels(branch_mask)
    gradient = path_direction(branch_ordered_pixels, phase_map)
    if gradient is None:
        return np.array([]), np.array([]), np.array([])
    # Normalize gradient
    gradient = np.stack([gradient[:, 0] / np.linalg.norm(gradient, axis=1),
                         gradient[:, 1] / np.linalg.norm(gradient, axis=1)]).T
    norm = np.matmul(gradient, [[0, -1], [1, 0]])
    # Phases found contains 1 when left/right phase pixel have been found
    phases_found = -2 * np.ones((len(branch_ordered_pixels), 2))
    neighbors = - np.ones((len(branch_ordered_pixels), 2, 2))  # 2nd dimension is [left, right], 3rd dim [x, y]

    steps = 1
    while not np.all(phases_found != -2):
        neighbors_done = phases_found != -2
        left_neighbor = np.round(branch_ordered_pixels + steps * norm).astype(int)
        right_neighbor = np.round(branch_ordered_pixels - steps * norm).astype(int)

        steps += 1
        for nid, neighbor in enumerate([left_neighbor, right_neighbor]):
            # Do not solve points out of image
            out_of_range = np.logical_or(
                np.logical_or(
                    neighbor[:, 0] < 0,
                    neighbor[:, 0] >= phase_map.shape[0]
                ), np.logical_or(
                    neighbor[:, 1] < 0,
                    neighbor[:, 1] >= phase_map.shape[1]))

            to_be_set_invalid = np.logical_and(out_of_range, ~neighbors_done[:, nid])
            phases_found[to_be_set_invalid, nid] = -1  # No valid phase found
            neighbors[to_be_set_invalid, nid, :] = -1
            to_be_set = np.logical_and(~out_of_range, ~neighbors_done[:, nid])
            phases_found[to_be_set, nid] = phase_map[neighbor[to_be_set, 0], neighbor[to_be_set, 1]]
            neighbors[to_be_set, nid, :] = neighbor[to_be_set, :]

    return phases_found.astype(int), neighbors.astype(int), gradient


def phase_analysis(crack_mask_smooth, phase_map):
    """
    Goes through crack skeleton pixels and for each pixel finds left and right phase.
    @param crack_mask_smooth: smoothed crack map
    @param phase_map: map of the grains
    @return: arrays with first two dimensions corresponding to pixel coordinates, then
        - through contains phase_id (left and right phase is the same)
        - edge contains two phase ids (left phase id and right phase id)
        - neighbors_map then contains 2D array with coordinates of left neighbor and right neighbor
    """
    skeleton, _ = cracks.derivates.skeleton(crack_mask_smooth)
    branches_mask, _, _ = cracks.derivates.skeleton_branches(skeleton)
    branches_map = label(branches_mask, background=0)
    branches = np.unique(branches_map)
    neighbors_map = -2 * np.ones(branches_map.shape + (2, 2))
    gradient_map = np.zeros(branches_map.shape + (2, ))

    phase_map[crack_mask_smooth.astype(bool)] = -2  # erase phase information where cracks were detected
    through = -np.ones_like(phase_map, dtype=int)
    edge = -np.ones(phase_map.shape + (2,), dtype=int)  # left and right differ
    # first id corresponds to background and will be skipped
    for branch_id in tqdm(branches[1:], total=len(branches) - 1, desc="Phase left/right (branches)"):
        branch_mask = branches_map == branch_id
        phases_per_px, neigh_coords_per_px, gradient = attach_neigh_phases_to_skeleton_branch(branch_mask, phase_map)
        if phases_per_px.size > 0:
            coords = np.where(branch_mask)
            xx, yy = coords
            neighbors_map[np.where(branch_mask)] = neigh_coords_per_px
            gradient_map[np.where(branch_mask)] = gradient
            for phases, x, y in zip(phases_per_px, xx, yy):
                if phases[0] == phases[1]:
                    through[x, y] = phases[0]
                else:
                    edge[x, y, :] = phases

    return through, edge, neighbors_map.astype(int), branches_map, gradient_map


def compute_cracks(crack_mask_smooth):
    """
    @param crack_mask_smooth: A binary mask indicating the location of cracks in an image. Value of 1 indicates crack region, and 0 indicates background.
    @return: A list of dictionaries, each containing features of individual cracks:
        - label: The label of the crack.
        - crackSize_px: The size of the crack in pixels.
        - skeleton_px: The number of pixels in the crack's skeleton.
        - maxWidth_px: The maximum width of the crack.
        - avgWidth_px: The average width of the crack.
        - boundaryLength_px: The length of the boundary of the crack.
        - farthestPoints_px: The distance between the farthest points in the crack.
    """
    if np.sum(crack_mask_smooth) == 0:
        return []

    cracks_skeletons, skeleton_to_boundary_distance = cracks.derivates.skeleton(crack_mask_smooth)
    mask_labeled = label(crack_mask_smooth, background=0)
    crack_labels, cracks_areas_px = np.unique(mask_labeled, return_counts=True)

    if len(crack_labels) <= 1:
        return []

    out = []
    for l, crack_area_px in tqdm(zip(crack_labels[1:], cracks_areas_px[1:]),
                                 total=len(crack_labels)-1, # Corrected total
                                 desc="Computing crack features"):
        crack_smooth = mask_labeled == l
        skeleton = np.logical_and(cracks_skeletons, crack_smooth)
        distance = skeleton_to_boundary_distance * skeleton
        
        if not np.any(skeleton):
            avg_width = 0.0
            max_width = 0.0
        else:
            max_width = np.max(distance)
            avg_width = np.mean(distance[distance != 0])

        y, x = np.where(crack_smooth)
        length = (np.sqrt((np.max(x) - np.min(x)) ** 2 + (np.max(y) - np.min(y)) ** 2))

        left_top = (np.min(y), np.min(x))
        bottom_right = (np.max(y) + 1, np.max(x) + 1)
        
        crack_patch = crack_smooth[left_top[0]:bottom_right[0], left_top[1]: bottom_right[1]]
        label_patch = mask_labeled[left_top[0]:bottom_right[0], left_top[1]: bottom_right[1]]

        marked = mark_boundaries(
            np.pad(crack_patch, ((1, 1), (1, 1))),
            np.pad(label_patch, ((1, 1), (1, 1))),
            outline_color=(0.5, 0, 0),
            mode="outer"
        )
        boundary = np.where(marked[:, :, 0] == 0.5)

        out.append({
            "label": l,
            "crackSize_px": crack_area_px,
            "skeleton_px": np.sum(skeleton),
            "maxWidth_px": max_width,
            "avgWidth_px": avg_width,
            "boundaryLength_px": len(boundary[0]),
            "farthestPoints_px": length
        })

    return out

def compute_bubbles(bubble_mask_smooth):
    """
    @param bubble_mask_smooth: A binary mask indicating the location of bubbles in an image. Value of 1 indicates bubble region, and 0 indicates background.
    @return: A list of dictionaries, each containing features of individual bubbles:
        - label: The label of the bubble.
        - bubbleArea_px: The size of the bubble in pixels.
        - boundaryLength_px: The length of the boundary of the bubble.
        - farthestPoints_px: The distance between the farthest points in the bubble's bounding box.
    """
    if np.sum(bubble_mask_smooth) == 0:
        return []

    mask_labeled = label(bubble_mask_smooth, background=0)
    bubble_labels, bubbles_areas_px = np.unique(mask_labeled, return_counts=True)

    if len(bubble_labels) <= 1:
        return []

    out = []
    for l, bubble_area_px in tqdm(zip(bubble_labels[1:], bubbles_areas_px[1:]),
                                 total=len(bubble_labels)-1,
                                 desc="Computing bubble features"):
        bubble_smooth = mask_labeled == l

        y, x = np.where(bubble_smooth)
        length = (np.sqrt((np.max(x) - np.min(x)) ** 2 + (np.max(y) - np.min(y)) ** 2))

        left_top = (np.min(y), np.min(x))
        bottom_right = (np.max(y) + 1, np.max(x) + 1)
        
        bubble_patch = bubble_smooth[left_top[0]:bottom_right[0], left_top[1]: bottom_right[1]]
        label_patch = mask_labeled[left_top[0]:bottom_right[0], left_top[1]: bottom_right[1]]

        marked = mark_boundaries(
            np.pad(bubble_patch, ((1, 1), (1, 1))),
            np.pad(label_patch, ((1, 1), (1, 1))),
            outline_color=(0.5, 0, 0),
            mode="outer"
        )
        boundary = np.where(marked[:, :, 0] == 0.5)

        out.append({
            "label": l,
            "bubbleArea_px": bubble_area_px,
            "boundaryLength_px": len(boundary[0]),
            "farthestPoints_px": length
        })

    return out


def global_summary(crack_table: pd.DataFrame, bubble_table: pd.DataFrame):
    res_table = pd.DataFrame({
        "skeletonSum_px": [0.0],
        "crackBoundaryLengthSum_px": [0.0],
        "bubbleAreaSum_px": [0.0],
        "bubbleBoundaryLengthSum_px": [0.0],
        "bubbleCount": [0],
    })

    if not crack_table.empty:
        res_table["skeletonSum_px"] = np.sum(crack_table["skeleton_px"])
        res_table["crackBoundaryLengthSum_px"] = np.sum(crack_table["boundaryLength_px"])

    if not bubble_table.empty:
        res_table["bubbleAreaSum_px"] = np.sum(bubble_table["bubbleArea_px"])
        res_table["bubbleBoundaryLengthSum_px"] = np.sum(bubble_table["boundaryLength_px"])
        res_table["bubbleCount"] = len(bubble_table)

    return res_table

def process_npy_to_features(npy_path, save_path):
    if not isinstance(npy_path, str) or not os.path.exists(npy_path):
        return pd.DataFrame()
    
    name = npy_path.split("/")[4]

    is_exp = "after" in npy_path
    stack = np.load(npy_path)

    mask, _, _ = adaptive_closing(stack)
    outer, inner = heal_chipped_mask(mask, inner_scale=0.85)

    cracks, bubbles, diff_proj = pipeline_geometric(stack, outer, inner)

    # 1. First Image: Pure Mask (cracks and bubbles over black background)
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    
    ax1.imshow(np.zeros_like(diff_proj), cmap="gray", vmin=0, vmax=1) 
    ax1.imshow(np.where(cracks > 0, 1, np.nan), cmap="Reds", vmin=0, vmax=1, alpha=0.8)
    ax1.imshow(np.where(bubbles > 0, 1, np.nan), cmap="Blues", vmin=0, vmax=1, alpha=0.8)
    
    # Remove all borders, labels, and spacing
    ax1.axis('off')
    fig1.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig1)

    # 2. Second Image: Pure Projection (diff_proj)
    base_path, ext = os.path.splitext(save_path)
    projection_save_path = f"{base_path}_projection{ext}"

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.imshow(diff_proj, cmap="gray")
    
    # Remove all borders, labels, and spacing
    ax2.axis('off')
    fig2.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    plt.savefig(projection_save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig2)

    # Features processing
    table_cracks = compute_cracks(cracks)
    table_bubbles = compute_bubbles(bubbles)
    df_cracks = global_summary(pd.DataFrame(table_cracks), pd.DataFrame(table_bubbles))

    df_cracks['sample_id'] = name

    logger.info(f"Image: {npy_path} Stage: {'AFTER' if is_exp else 'before'} \n{df_cracks}")
    
    del stack, mask, outer, inner, cracks, bubbles, diff_proj
    gc.collect()

    return df_cracks

def process_single_pair(idx, root, file_before, file_after):
    pair_dir = os.path.join(root, "{:02d}".format(idx))
    os.makedirs(pair_dir, exist_ok=True)
    
    name_before = os.path.splitext(os.path.basename(file_before))[0] + ".png"
    name_after = os.path.splitext(os.path.basename(file_after))[0] + ".png"
    
    save_before = os.path.join(pair_dir, name_before)
    save_after = os.path.join(pair_dir, name_after)
    
    df_before  = process_npy_to_features(file_before, save_before)
    df_after = process_npy_to_features(file_after, save_after)
    
    df_before['stage'] = 'before'
    df_before['pair_id'] = idx
        
    df_after['stage'] = 'after'
    df_after['pair_id'] = idx
    
    logger.info("pair loaded")
        
    return df_before, df_after

def process_pairs(pairs_df: pd.DataFrame, load_root, savedir, save_name, temperature_map):
    results_dict = {}
    col_before = 'Sample before exposure'
    col_after = 'Sample after exposure'   

    logger.info("Starting pair comparison iteration.")
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Comparing pairs"):
        file_before = os.path.join(load_root, str(row.get(col_before, '')))
        file_after = os.path.join(load_root, str(row.get(col_after, '')))

        logger.debug(f"Processing Pair {idx} - Before: {file_before} | After: {file_after}")

        if not os.path.exists(file_before) or not os.path.exists(file_after):
            logger.warning(f"Incomplete pair at index {idx}: Both files must exist. Skipping.")
            continue

        df_before, df_after = process_single_pair(idx, savedir, file_before, file_after)
        results_dict[idx] = {
            'before': df_before,
            'after': df_after,
        }

    logger.info("Pair iteration complete. Aggregating results.")

    all_before = [v['before'] for v in results_dict.values() if not v['before'].empty]
    all_after = [v['after'] for v in results_dict.values() if not v['after'].empty]

    if not all_before or not all_after:
        logger.error("No valid pairs processed. Aborting aggregation.")
        return

    df_before_combined = pd.concat(all_before)
    df_after_combined = pd.concat(all_after)

    agg_before = df_before_combined.groupby('pair_id').agg({
        **{col: 'sum' for col in df_before_combined.columns if col not in ['pair_id', 'sample_id']},
        'sample_id': 'first'
    })

    agg_after = df_after_combined.groupby('pair_id').agg({
        **{col: 'sum' for col in df_after_combined.columns if col not in ['pair_id', 'sample_id']},
        'sample_id': 'first'
    })

    numeric_cols = agg_after.select_dtypes(include=[np.number]).columns
    diff_df = agg_after[numeric_cols] - agg_before[numeric_cols]

    agg_before_prefixed = agg_before[numeric_cols].add_prefix('before_')
    agg_after_prefixed = agg_after[numeric_cols].add_prefix('after_')
    diff_df_prefixed = diff_df.add_prefix('diff_')

    side_by_side = pd.concat([agg_before_prefixed, agg_after_prefixed, diff_df_prefixed], axis=1)

    side_by_side['sample_id'] = agg_before['sample_id']
    side_by_side['temperature'] = side_by_side['sample_id'].map(temperature_map)

    cols = ['sample_id', 'temperature'] + [c for c in side_by_side.columns if c not in ['sample_id', 'temperature']]
    side_by_side = side_by_side[cols]

    save_path = os.path.join(savedir, save_name)
    side_by_side.to_csv(save_path)
    logger.info(f"Results successfully saved to {save_path}")