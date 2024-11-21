import logging

from skimage.morphology import disk, opening, dilation, label
import numpy as np
import skimage.measure
import os
from tqdm.auto import tqdm
import utils.image_logger as image_logger


GRAIN_STRUCT_ELEMENT = disk(10)
SUBGRAIN_STRUCT_ELEMENT = disk(5)
logger = logging.getLogger(__name__)


def segment_grains(phase_map, output_dir_path=None, output_file_prefix=""):
    """
        Method goes thru the grain map an apply morphology to founded grains.
        Morphology includes opening for reduction of "threads" and attachment of leftovers to most frequent grain nearby.
        @param phase_map contains raw data from the microscope where each pixel is attached to some phase
        @returns refined phase_map after applying morphology. Phase ids could not match.
        """
    logger.info(f'Computation of the grain set started.')

    # Label individual grains in phase_id-th phase
    labeled_phase_map = label(phase_map, background=0)
    # Do opening of these grains (i.e. erosion and dilation) - this removes long threads (grains do not have threads)
    grains_open = opening(labeled_phase_map, GRAIN_STRUCT_ELEMENT)
    # relabel opened grains
    sub_grains = label(grains_open, background=0)
    # dilate relabeled opened grains, but remove pixels belonging to a crack.
    # This way we mark relevant areas with the closest grain_id
    the_grains = dilation(sub_grains, SUBGRAIN_STRUCT_ELEMENT) * (labeled_phase_map > 0).astype(int)
    # There are some residua from the phase map, which were not attached to any grain
    # These leftovers should be managed as follows:
    # 1. for their outline find intersection with a grain
    # 2a. if no such grain exists then the leftover belongs to a crack
    # 2b. if such grain exists compute the most frequent one (by means of pixels) and attach leftover to this grain
    leftovers = label(np.logical_xor(labeled_phase_map > 0, the_grains.astype(bool)))
    leftovers_dilated = dilation(leftovers, disk(2))
    leftovers_outline = np.logical_xor(leftovers, leftovers_dilated)
    leftovers_outline_grain_overlap = the_grains * leftovers_outline
    # remove all leftovers (now are cracks)
    the_grains[leftovers != 0] = 0
    # for leftovers overlapping with its outline some grains attach those leftovers to grains
    attachable_leftovers = np.unique((leftovers_outline_grain_overlap != 0) * leftovers_dilated)[1:]
    for leftover_id in tqdm(attachable_leftovers, total=attachable_leftovers.size, desc="Grain refine (filling leftovers)"):
        leftover_mask = leftovers_dilated == leftover_id
        grain_ids, counts = np.unique(leftover_mask * leftovers_outline_grain_overlap, return_counts=True)
        assert grain_ids.size != 1, f"{grain_ids} attached to this leftover outline do not match outline intersection with grains"
        attached_grain_id = grain_ids[grain_ids != 0][np.argsort(counts[grain_ids != 0])[0]]
        the_grains[leftover_mask] = attached_grain_id

    logger.info(f'Computation of the grain set done. Total {np.max(the_grains)} grains.')
    image_logger.info(the_grains, output_dir_path, f"{output_file_prefix}[user]grain_map.tiff")
    if output_dir_path is not None:
        image_logger.dump_image(os.path.join(output_dir_path, f"{output_file_prefix}grain_map.tiff"), the_grains.astype(np.uint16))

    return the_grains


def grain_size_filter(grain_map, grain_size_px_limit=9000, output_dir_path=None, output_file_prefix=""):
    logger.info(f'Grain filter started.')

    grain_idx, grain_sizes_px = np.unique(grain_map.ravel(), return_counts=True)

    grain_ids_bigger_than_limit = [grain_id
                                   for grain_id, grain_size_px in zip(grain_idx, grain_sizes_px)
                                   if grain_size_px >= grain_size_px_limit]

    # Delete grains smaller than limit and relabel the rest.
    grain_map_size_limit = np.zeros_like(grain_map)
    grain_map_size_limit[np.isin(grain_map, grain_ids_bigger_than_limit)] = 1
    grain_map_size_limit = label(grain_map_size_limit, background=0)
    non_matrix_grains_count = np.max(grain_map_size_limit)
    image_logger.info((grain_map_size_limit != 0).astype(float), output_dir_path, f"{output_file_prefix}[user]non_matrix_grains.png")

    # matrix is phase mixed from two (or more) phases and the rest of filtered out grains
    matrix_grains, matrix_grains_count = skimage.measure.label(
        grain_map_size_limit == 0,
        background=0,
        return_num=True
    )
    image_logger.info((matrix_grains != 0).astype(float), output_dir_path, f"{output_file_prefix}[user]matrix_grains.png")

    matrix_grains[matrix_grains != 0] += non_matrix_grains_count
    grain_map_size_limit[grain_map_size_limit == 0] = matrix_grains[matrix_grains > 0]

    logger.info(f'Grain filter done. Total {np.max(non_matrix_grains_count)} grains and {matrix_grains_count} matrix grains.')
    image_logger.info(grain_map_size_limit, output_dir_path, f"{output_file_prefix}[user]grain_map_filtered.tiff")
    # Dump filtered grains image into output
    if output_dir_path is not None:
        image_logger.dump_image(os.path.join(output_dir_path, f"{output_file_prefix}grain_map_filtered.tiff"), grain_map_size_limit.astype(np.uint16))

    return grain_map_size_limit, grain_size_px_limit, non_matrix_grains_count
