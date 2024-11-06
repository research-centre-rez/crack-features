import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, opening, dilation, label
from tqdm.auto import tqdm
from datetime import datetime
import logging

GRAIN_STRUCT_ELEMENT = disk(10)
SUBGRAIN_STRUCT_ELEMENT = disk(5)

logger = logging.getLogger(__name__)


def grainID2(phase_map):
    start_time = datetime.now()

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
    for leftover_id in tqdm(attachable_leftovers, total=attachable_leftovers.size):
        leftover_mask = leftovers_dilated == leftover_id
        grain_ids, counts = np.unique(leftover_mask * leftovers_outline_grain_overlap, return_counts=True)
        assert grain_ids.size != 1, f"{grain_ids} attached to this leftover outline do not match outline intersection with grains"
        attached_grain_id = grain_ids[grain_ids != 0][np.argsort(counts[grain_ids != 0])[0]]
        the_grains[leftover_mask] = attached_grain_id

    stop_time = datetime.now()
    logger.info(f'Computation of the grain set done. Elapsed Time: {stop_time - start_time}')
    return the_grains
