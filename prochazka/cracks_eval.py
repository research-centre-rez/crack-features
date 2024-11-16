from copy import deepcopy

import numpy as np
from skimage.morphology import disk, skeletonize, medial_axis
from skimage.measure import label
from scipy.ndimage import binary_dilation

# bw_morph is python code simulating MATLAB bwmorph lib
from bw_morph import endpoints, branches
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)


def crack_metadata(crack, dilation_radius=5):
    segment_labels = crack["segment_labels"].toarray()
    outline = np.logical_xor(
        binary_dilation(segment_labels, structure=disk(dilation_radius)),
        segment_labels)

    outline_skeleton = medial_axis(np.logical_or(outline, segment_labels))
    skeleton_outline_pixel_count = np.sum(outline_skeleton)
    end_points = np.column_stack(np.where(endpoints(outline_skeleton).T))
    nodes = branches(outline_skeleton)

    # Transposition is due to image coordinates indexation
    nodes_coords = np.column_stack(np.where(nodes.T))

    return {
        "segment_labels": segment_labels,
        "outline": outline,
        "outline_skeleton": outline_skeleton,
        "skeleton_outline_pixel_count": skeleton_outline_pixel_count,
        "end_points_coords": end_points,
        "nodes": {
            "mask": nodes,
            "coords": nodes_coords
        }
    }


def point_metadata(x, y, skeleton_branches, image_shape, is_end):
    height, width = image_shape
    return {
        'coords': [x, y],
        # branches contains array of branch numbers near [-4,+4] the endpoint
        'branches': np.unique(skeleton_branches[
                              max(0, y - 4):min(height, y + 5),
                              max(0, x - 4):min(width, x + 5)
                              ])[1:],
        'is_end': is_end,
        'is_terminal': False
    }


def get_skeleton_branches(skeleton, skeleton_nodes):
    # Create a structuring element (disk with radius 2)
    structuring_element = disk(2)

    # Perform binary dilation on bw_skeleton_nodes
    dilated_nodes = binary_dilation(skeleton_nodes, structuring_element)

    # Subtract the dilated image from bw_skeleton and check where it's greater than 0
    binary_diff = np.logical_and(skeleton, np.logical_not(dilated_nodes))

    # Label the regions in the binary_diff image
    labeled_skeleton_branches = label(binary_diff)

    return labeled_skeleton_branches, np.max(labeled_skeleton_branches)


def cracks_evaluation(grains_map, cracks, grains_metadata, dilation_radius):
    """
    Processing of the prepared segmentation maps

    @param grains_map - segmentation map from the miscroscope according to pixel-material phase
    @param cracks - list of cracks (@see ... TODO)
    """
    cracks_out = deepcopy(cracks)

    for crack in tqdm(cracks_out, total=len(cracks_out), desc="Cracks evaluation"):
        metadata = crack_metadata(crack, dilation_radius)
        skeleton_branches, branches_count = get_skeleton_branches(
            metadata['outline_skeleton'],
            metadata['nodes']['mask']
        )
        # Remove crack pixels from the grains_map
        grains_map_wo_crack = np.copy(grains_map)
        grains_map_wo_crack[metadata["outline_skeleton"]] = 0

        points = [point_metadata(x, y, skeleton_branches, grains_map.shape, is_end=True)
                  for x, y in metadata["end_points_coords"]]
        points.extend([point_metadata(x, y, skeleton_branches, grains_map.shape, is_end=False)
                       for x, y in metadata["nodes"]["coords"]])

        skeletons_endpoints_distance = np.array([
            np.linalg.norm(metadata["end_points_coords"] - endpoint, axis=1)
            for endpoint in metadata["end_points_coords"]
        ])

        furthest_endpoints_indices = np.unravel_index(
            skeletons_endpoints_distance.argmax(),
            skeletons_endpoints_distance.shape
        )
        for endpoint_id in furthest_endpoints_indices:
            points[endpoint_id]['is_terminal'] = True

        branch_struct = [{
            'mask': skeleton_branches == branch_id,
            'Nodes': [point["coords"] for point in points if branch_id in point['branches']],
            'xy': np.column_stack(np.nonzero(skeleton_branches == branch_id)),
            'OnEdge': [],
            'Through': []
        } for branch_id in range(1, branches_count + 1)]

        for branch_id in range(branches_count):
            # branch_struct[branch_id] = sortPoints(branch_struct[branch_id], points)
            neighbors_phases, neighbors_coords = attach_neigh_phases_to_skeleton_branch(
                branch_struct[branch_id],
                grains_map_wo_crack
            )
            neighbors_phases = neighbors_phases[(neighbors_phases.min(axis=1) != 0), :]  # Remove zero-phase entries
            branch_struct[branch_id]['Through'] = neighbors_phases[(neighbors_phases[:, 0] == neighbors_phases[:, 1])]
            branch_struct[branch_id]['OnEdge'] = neighbors_phases[(neighbors_phases[:, 0] != neighbors_phases[:, 1])]

        Through = [bs['Through'] for bs in branch_struct if bs['Through'].size != 0]
        if len(Through) > 0:
            Through = np.concatenate(Through)
            Unq = np.unique(Through)
            crack['through'] = {
                "grains_id": Unq,
                "phases_id": [grains_metadata[u]['phase_id'] for u in Unq],
                "grains_counts": [np.sum(Through == u) for u in Unq]
            }
        else:
            crack['through'] = {
                "grains_id": [],
                "phases_id": [],
                "grains_counts": []
            }

        OnEdge = [bs['OnEdge'] for bs in branch_struct if bs['OnEdge'].size != 0]
        if len(OnEdge) > 0:
            OnEdge = np.concatenate(OnEdge)
            Unq = np.unique(OnEdge, axis=0)
            crack['OnEdge'] = {
                "grains_id_pairs": Unq,
                "phases_id_pairs": np.array([grains_metadata[u]['phase_id'] for u in Unq.reshape(-1)]).reshape(
                    Unq.shape),
                "counts": [np.sum(np.prod(Unq[edge_point_id, :2] == OnEdge, axis=1)) for edge_point_id in
                           range(Unq.shape[0])]
            }
        else:
            crack['OnEdge'] = {
                "grains_id_pairs": [],
                "phases_id_pairs": [],
                "counts": []
            }

    return cracks_out, points, branch_struct


def _find_path_start(binary_mask):
    """
    Find a pixel with only one neighbor to use as a start pixel.
    """
    path_pixels = np.argwhere(binary_mask == 1)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    for pixel in path_pixels:
        neighbors = 0
        for d in directions:
            neighbor = (pixel[0] + d[0], pixel[1] + d[1])
            if (0 <= neighbor[0] < binary_mask.shape[0] and
                    0 <= neighbor[1] < binary_mask.shape[1] and
                    binary_mask[neighbor] == 1):
                neighbors += 1
        if neighbors == 1:  # Pixel with only one neighbor is an endpoint
            return pixel
    return path_pixels[0]  # Fallback if no endpoint is found (i.e. branch is a cycle)


def sort_branch_pixels(binary_mask):
    """
    Goes thru point set in the binary mask and order non-zero pixel into a path.
    """

    # Get coordinates of path pixels
    path_pixels = np.argwhere(binary_mask == 1)
    # Starting pixel
    start_pixel = _find_path_start(binary_mask)
    # Directions for moving to neighboring pixels (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    # Initialize the list to store ordered pixels and set of visited pixels
    ordered_pixels = []
    visited = set()

    # Depth-First Search (DFS) to follow the path
    def dfs(pixel):
        stack = [pixel]
        while stack:
            px = stack.pop()
            if tuple(px) in visited:
                continue
            visited.add(tuple(px))
            ordered_pixels.append(px)

            # Look in each direction for connected path pixels
            for d in directions:
                neighbor = (px[0] + d[0], px[1] + d[1])

                # Check if the neighbor is within bounds, unvisited, and part of the path
                if (0 <= neighbor[0] < binary_mask.shape[0] and
                        0 <= neighbor[1] < binary_mask.shape[1] and
                        binary_mask[neighbor] == 1 and
                        tuple(neighbor) not in visited):
                    stack.append(neighbor)
        return ordered_pixels

    # Run DFS starting from the initial pixel
    dfs(start_pixel)
    # Convert ordered pixels to a list of tuples
    return [tuple(px) for px in ordered_pixels]


def pixels_gradient(branch_ordered_pixels, mask):
    """
    Computes pixel gradient according to available inputs. See the code.
    """
    if len(branch_ordered_pixels) < 2:  # in this case it is not possible to compute gradient
        crack_nearby = mask[
            np.max([0, branch_ordered_pixels[0][0] - 1]):np.min([branch_ordered_pixels[0][0] + 2, mask.shape[0]]),
            np.max([0, branch_ordered_pixels[0][1] - 1]):np.min([branch_ordered_pixels[0][1] + 2, mask.shape[1]]),
        ]
        nearby_crack_pixels_count = np.where(crack_nearby == 0)[0].size
        if nearby_crack_pixels_count == 1:  # there is no other pixel belonging to a crack
            logger.error("Invalid crack branch (single pixel). No gradient.")
            return None
        elif nearby_crack_pixels_count == 2:  # there is 1 other pixel belonging to a crack
            gradient = np.gradient(np.stack(np.where(crack_nearby == 0), axis=1), axis=0)[:1]
        elif nearby_crack_pixels_count == 3:  # ideal case, there is pixel before and after and gradient can be computed
            gradient = np.gradient(np.stack(np.where(crack_nearby == 0), axis=1), axis=0)[1:2]
        else:  # there is more than two pixels, it is not clear what to use for gradient computation
            return None
    else:
        gradient = np.gradient(branch_ordered_pixels, axis=0)
    return gradient


def attach_neigh_phases_to_skeleton_branch(branch_struct, mask):
    """
    For each point of a branch goes in the direction perpendicular to the skeleton in-point derivative and looks for
    phase in the phase map

    @param branch_struct contains points belonging to the branch
    @param mask - phase image has for each pixel a phase ID attached
    @return list of phase-pairs for each branch point
            list of point-pairs i.e. neighbors belonging to branch point
    """
    branch_ordered_pixels = sort_branch_pixels(branch_struct['mask'])
    gradient = pixels_gradient(branch_ordered_pixels, mask)
    if gradient is None:
        return np.array([]), np.array([])
    # Normalize gradient
    gradient = np.stack([gradient[:, 0] / np.linalg.norm(gradient, axis=1),
                         gradient[:, 1] / np.linalg.norm(gradient, axis=1)]).T
    norm = np.matmul(gradient, [[0, -1], [1, 0]])
    # Phases found contains 1 when left/right phase pixel have been found
    phases_found = np.zeros((len(branch_ordered_pixels), 2))
    neighbors = np.zeros((len(branch_ordered_pixels), 2, 2))  # 2nd dimension is [left, right], 3rd dim [x, y]

    left_neighbor = np.copy(branch_ordered_pixels)
    right_neighbor = np.copy(branch_ordered_pixels)

    while not np.all(phases_found != 0):
        neighbors_done = phases_found != 0
        left_neighbor = np.round(left_neighbor + norm).astype(int)
        right_neighbor = np.round(right_neighbor - norm).astype(int)

        for nid, neighbor in enumerate([left_neighbor, right_neighbor]):
            # Do not solve points out of image
            out_of_range = np.logical_or(
                np.logical_or(
                    neighbor[:, 0] < 0,
                    neighbor[:, 0] >= mask.shape[0]
                ), np.logical_or(
                    neighbor[:, 1] < 0,
                    neighbor[:, 1] >= mask.shape[1]))

            phases_found[np.logical_and(out_of_range, ~neighbors_done[:, nid]), nid] = -1  # No valid phase found
            to_be_set = np.logical_and(~out_of_range, ~neighbors_done[:, nid])
            phases_found[to_be_set, nid] = mask[neighbor[to_be_set, 0], neighbor[to_be_set, 1]]
            neighbors[to_be_set, nid, :] = neighbor[to_be_set, :]

    return phases_found.astype(int), neighbors.astype(int)
