from skimage.measure import label
from skimage.segmentation import mark_boundaries
import numpy as np
from tqdm.auto import tqdm
from cracks.branches import sort_branch_pixels, path_direction
import cracks.derivates


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


def compute(crack_mask_smooth):
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
    cracks_skeletons, skeleton_to_boundary_distance = cracks.derivates.skeleton(crack_mask_smooth)
    mask_labeled = label(crack_mask_smooth, background=0)
    crack_labels, cracks_areas_px = np.unique(mask_labeled, return_counts=True)

    out = []
    # Skip label == 0 which is background
    for l, crack_area_px in tqdm(zip(crack_labels[1:], cracks_areas_px[1:]),
                                 total=len(crack_labels),
                                 desc="Computing crack features"):
        crack_smooth = mask_labeled == l
        skeleton = np.logical_and(cracks_skeletons, crack_smooth)
        distance = skeleton_to_boundary_distance * skeleton

        y, x = np.where(crack_smooth)
        length = (np.sqrt((np.max(x) - np.min(x)) ** 2 + (np.max(y) - np.min(y)) ** 2))

        # Why there is not used just dilation and subtraction? Faster?
        left_top = (np.min(y), np.min(x))
        bottom_right = (np.max(y) + 1, np.max(x) + 1)
        crack_bounding_box = mask_labeled[left_top[0]:bottom_right[0], left_top[1]: bottom_right[1]]
        marked = mark_boundaries(
            np.pad(crack_smooth[left_top[0]: bottom_right[0], left_top[1]: bottom_right[1]],
                   ((1, 1), (1, 1))),
            np.pad(crack_bounding_box, ((1, 1), (1, 1))),
            outline_color=(0.5, 0, 0),
            mode="outer"
        )
        boundary = np.where(marked[:, :, 0] == 0.5)

        out.append({
            "label": l,
            "crackSize_px": crack_area_px,
            "skeleton_px": np.sum(skeleton),
            "maxWidth_px": np.max(distance),
            "avgWidth_px": np.mean(distance[distance != 0]),
            "boundaryLength_px": len(boundary[0]),
            "farthestPoints_px": length
        })
    return out
