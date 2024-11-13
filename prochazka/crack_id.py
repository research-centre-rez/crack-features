import numpy as np
from scipy.sparse import csr_matrix
from skimage.measure import label
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def build_crack_metadata(skeleton, cracks_mask):
    """
    Creates list of cracks metadata
    @param skeleton - binary image with the skeleton of the cracks
    @param crack_mask - mask of the cracks

    @return metadata and labeled segmented image
    """

    # Enumerate the cracks
    labeled_img = label(cracks_mask, background=0)

    # Align Skeleton enumeration with cracks enumeration
    labeled_skeleton = labeled_img * skeleton.astype(float)

    # Count individual cracks
    cracks = []
    cracks_ids = np.unique(labeled_img)[1:]
    for crack_id in tqdm(cracks_ids, total=len(cracks_ids), desc="Crack labeling"):
        crack_mask = labeled_img == crack_id
        crack_size_px = np.sum(crack_mask)
        skeleton_mask = labeled_skeleton == crack_id
        skeleton_size_px_jb = np.sum(skeleton_mask)
        cracks.append({
            'area_pixel_count': crack_size_px,
            'skeleton_pixel_count_jb': skeleton_size_px_jb,
            'width_px': crack_size_px / skeleton_size_px_jb if skeleton_size_px_jb != 0 else np.nan,  # ??
            'segment_labels': csr_matrix(crack_mask),
            'skeleton': csr_matrix(skeleton_mask)
        })

    return cracks, labeled_img


def highlight_segment(group):
    # build group boundary
    group_boundary = binary_dilation(group) ^ group
    group_boundary_labeled = label(group_boundary)  # ? could be split into more than one area
    group_boundary_parts_count = np.max(group_boundary_labeled)

    if group_boundary_parts_count <= 1:
        return None

    # Take first label
    piece = group_boundary_labeled == 1
    # Remove it from the image
    rest = np.logical_xor(piece, group)

    # Use this if you want to draw the closest point between rest and piece
    # closest_p_point, closest_r_point, distance = find_closest(areaA_binary, areaB_binary)

    # background = 0, piece = 1, rest = 2
    return piece + 2 * rest


def find_closest(areaA_binary, areaB_binary):
    """
    For the two areas (binary images) finds two closest points and count thier distance
    """

    # take all points in the areaA_binary and extract coordinates
    a_coords = np.column_stack(np.where(areaA_binary > 0))
    # take all point in the areaB_binary and extract coordinates
    b_coords = np.column_stack(np.where(areaB_binary > 0))
    # compute distances between each pair
    distances = cdist(a_coords, b_coords)

    closest_a_dist_id, closest_b_dist_id = np.unravel_index(
        distances.argmin(), distances.shape
    )
    closest_a_point = a_coords[closest_a_dist_id]
    closest_b_point = b_coords[closest_b_dist_id]

    return closest_a_point, closest_b_point, distances.min()
