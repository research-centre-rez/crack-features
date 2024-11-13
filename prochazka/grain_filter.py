from copy import deepcopy

import numpy as np
from datetime import datetime
from grain_param import build_grain_metadata
import skimage.measure
import logging

logger = logging.getLogger(__name__)


def grain_size_filter(grains_metadata, grain_map, layers, grain_size_px_limit=9000):
    start_time = datetime.now()

    grain_sizes_px = np.array([(grain_id, grain['size_px']) for grain_id, grain in grains_metadata.items()])
    grain_ids_bigger_than_limit = [grain_id for grain_id, grain_size_px in grain_sizes_px if grain_size_px >= grain_size_px_limit]


    grain_map_size_limit = np.zeros_like(grain_map)
    grains_metadata_size_limit = []
    layers_updated = deepcopy(layers)

    new_grain_id = 0
    for grain_id in range(1, len(grain_sizes_px)):
        if grain_id in grain_ids_bigger_than_limit:
            # create new IDS for grains
            grain_metadata = deepcopy(grains_metadata[grain_id])
            grain_metadata['grain_id'] = new_grain_id
            # store metadata into new structure
            grains_metadata_size_limit.append(grain_metadata)
            # mark grain into new map
            grain_map_size_limit[grain_map == grain_id] = new_grain_id
            new_grain_id += 1

    # matrix is phase mixed from two (or more) phases and the rest of filtered out grains
    # TODO: matrix should be split into grains first

    matrix_grains, matrix_grains_count = skimage.measure.label(
        grain_map_size_limit == 0,
        background=0,
        return_num=True
    )

    non_matrix_grains_count = len(grains_metadata_size_limit)
    for matrix_id in np.arange(matrix_grains_count):
        matrix_metadata = build_grain_metadata(matrix_id, matrix_grains, phase_id=7, label="Matrix")
        matrix_metadata['grain_id'] = non_matrix_grains_count + matrix_id
        grain_map_size_limit[matrix_grains == matrix_id] = non_matrix_grains_count + matrix_id
        grains_metadata_size_limit.append(
            matrix_metadata
        )

    for layer in layers_updated:
        layer['children'] = np.unique(grain_map_size_limit[layer['mask'] > 0])

    logger.info(f'Grain filter done in: {datetime.now() - start_time}')
    return grain_map_size_limit, grains_metadata_size_limit, layers_updated, grain_size_px_limit
