import numpy as np
from scipy.ndimage import binary_erosion
from datetime import datetime
from tqdm.auto import tqdm


def build_grain_metadata(grain_id, grain_map, phase_id, label):
    """
    Method create grain metadata i.e.: id, phase_id, label, grain size (px), grain outline size (px)
    and ratio of outline size and grain size
    @param grain_id: int grain id
    @param grain_map: 2D int array grain map
    @param phase_id: int phase id
    @param label: str label
    @return dict
    """
    grain_mask = grain_map == grain_id
    grain_size_pixels = np.sum(grain_mask)
    grain_outline_size_pixels = np.sum(np.logical_xor(grain_mask, binary_erosion(grain_mask)))
    return {
        'grain_id': grain_id,
        'phase_id': phase_id,
        'label': label,
        'size_px': grain_size_pixels,
        'outline_size_px': grain_outline_size_pixels,
        'outline_size_to_size_ratio': grain_outline_size_pixels / grain_size_pixels
    }


def build_grains_metadata(grain_map, phase_map, layers):
    """
    Method split grains into corresponding phases and for each grain build its metadata.
    @param grain_map: 2D int array grain map
    @param phase_map: 2D int array phase map
    @param layers: list of phase layers
    @return dict with grain metadata and list with layers metadata
    """
    start_time = datetime.now()
    unique_phases = np.unique(phase_map)

    # Initialize grains_metadata dictionary
    grains_metadata = {}

    # Proč se toto počítá/generuje pro každou fázi zvlášť?
    # a) grain může patřit k více fázím a potom se objeví ve výpisu několikrát přepíše??
    for phase_id in tqdm(unique_phases, desc='Building grains metadata (phases)'):
        # maska částí zrn obsažených v konkrétní fázi
        # NOTE: možná místo původní fázové mapy by se tu měla použít refinovaná fázová mapa i.e. grain_map
        grains_in_phase = (phase_map == phase_id) * grain_map
        # seznam zrn ve fázi
        grains_in_phase_ids = np.unique(grains_in_phase)
        grains_in_phase_ids = grains_in_phase_ids[grains_in_phase_ids != 0]
        # label pro danou fázi
        label = layers[phase_id]['label']
        # children pro danou fázi
        layers[phase_id]['children'] = grains_in_phase_ids

        for grain_id in grains_in_phase_ids:
            grains_metadata[grain_id] = build_grain_metadata(grain_id, grain_map, phase_id, label)

    end_time = datetime.now()
    print(f'Grain parameters set in: {end_time - start_time}')

    return grains_metadata, layers

