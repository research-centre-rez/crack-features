import os
import numpy as np
from skimage.io import imread
from skimage.measure import label
import logging
import phase_thr as pt
from grain_id import refine_grains
from grain_param import build_grains_metadata
from grain_filter import grain_size_filter as grain_filter
from crack_id import build_crack_metadata as crack_id
from cracks_eval import cracks_evaluation
from cracks_count import cracksCount as cracks_count
from types import SimpleNamespace

# Testing folder
ROOT = "/Users/gimli/Library/CloudStorage/OneDrive-UJV/JCAMP_NRA_LOM_SEM-EDX/SEM-EDX"
RESULTS = "/Users/gimli/cvr/data/microscopy/SEM-EDX"
# Crop element labels in the bottom part of the image
CROP_RANGE = np.array([[0, 2000], [0, 2000]])


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(RESULTS, 'filesystem-scan-log.txt'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def path_to_edx_input(parent_folder_path):
    edx_map_list = []
    for root, dirs, files in os.walk(parent_folder_path):
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                edx_map_list.append(os.path.join(root, file))
    edx_map_list = sorted(edx_map_list)
    return edx_map_list


def scan_filesystem_for_samples(main_folder):
    sample_paths = sorted([
        os.path.join(main_folder, folder)
        for folder in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, folder))
    ])

    samples = []
    for sample_path in sample_paths:
        if not os.path.isdir(os.path.join(sample_path, 'JBmasks')):
            logger.warning(f"Sample {os.path.basename(sample_path)} do not have JBmasks")
            continue

        crack_mask_path = sorted([
            file for file in os.listdir(os.path.join(sample_path, 'JBmasks'))
            if file.endswith('.png')
        ])

        skeleton_mask_path = sorted([
            file for file in os.listdir(os.path.join(sample_path, 'JBskeletons'))
            if file.endswith('.png')
        ])

        edx_map_list = []
        edx_map_list.extend(path_to_edx_input(os.path.join(sample_path, 'EDX layered images')))
        edx_map_list.extend(path_to_edx_input(os.path.join(sample_path, 'EDX layered maps')))
        samples.append((sample_path, sorted(set(edx_map_list)), crack_mask_path, skeleton_mask_path))
    return samples


def load_inputs(parent_folder, edx_list, crack_list, skeleton_list):
    map_path = os.path.join(parent_folder, 'EDX layered images', edx_list[0])
    phase_map = imread(map_path)
    map_size = phase_map.shape
    CROP_RANGE[:, 1] = np.minimum(CROP_RANGE[:, 1], map_size[:2])
    phase_map = phase_map[CROP_RANGE[0, 0]:CROP_RANGE[0, 1], CROP_RANGE[1, 0]:CROP_RANGE[1, 1], :3]

    mask_path = os.path.join(parent_folder, 'JBmasks', crack_list[0])
    crack_mask = (imread(mask_path) > 0).astype(float)
    crack_mask = crack_mask[CROP_RANGE[0, 0]:CROP_RANGE[0, 1], CROP_RANGE[1, 0]:CROP_RANGE[1, 1]]

    skel_path = os.path.join(parent_folder, 'JBskeletons', skeleton_list[0])
    skeleton_mask = (imread(skel_path) > 0).astype(float)
    skeleton_mask = skeleton_mask[CROP_RANGE[0, 0]:CROP_RANGE[0, 1], CROP_RANGE[1, 0]:CROP_RANGE[1, 1]]

    sample_file_name = edx_list[0].split(os.path.sep)[-1].split('.')[0]
    output_path = os.path.join(RESULTS, sample_file_name)
    os.makedirs(output_path, exist_ok=True)

    return phase_map, crack_mask, skeleton_mask, output_path


def process_sample(sample_dir, edx_map_path, crack_mask_path, skeleton_mask_path):
    phase_map_rgb, crack_mask, skeleton_mask, output_dir = load_inputs(
        parent_folder=sample_dir,
        edx_list=edx_map_path,
        crack_list=crack_mask_path,
        skeleton_list=skeleton_mask_path
    )
    logger = logging.getLogger(os.path.basename(output_dir))
    logger.addHandler(logging.FileHandler(os.path.join(output_dir, 'log.txt')))


    if np.max(label(crack_mask)) == 0 or np.max(label(skeleton_mask)) == 0:
        logger.info("no cracks found")

    # TODO: CLI arguments
    args = SimpleNamespace(**{
        "cracks": crack_mask,
        "gaussian_blur": 3,
        "threshold": 5,
        "lab_distance": True,
        "adjust_intensity": None
    })

    thresholded_phase_map, layers = pt.phase_threshold(phase_map_rgb, args, output_dir)

    grain_map_base = refine_grains(thresholded_phase_map)
    logger.warning("Grains metadata")
    grains_metadata, layers = build_grains_metadata(grain_map_base, thresholded_phase_map, layers)
    grain_map, grains_metadata, layers, size_limit = grain_filter(grains_metadata, grain_map_base, layers)

    # remove cracks from phase map
    cracks, cracks_map = crack_id(skeleton_mask, crack_mask)
    # Why is missing return values?
    cracks, _, _ = cracks_evaluation(grain_map, cracks, grains_metadata, 3)
    en, el, in_, il = cracks_count(cracks, pt.PHASES_CONFIG)

    save_path = os.path.join(output_dir, "stats.npz")
    np.savez(save_path,
             ch_phases=pt.PHASES_CONFIG,
             layers=layers,
             grain_struct=grains_metadata,
             cracks=cracks,
             grain_map=grain_map,
             map2=thresholded_phase_map,
             cracks_map=cracks_map
    )


if __name__ == "__main__":

    samples = scan_filesystem_for_samples(ROOT)

    # TODO: Run this in parallel
    for sample_id, (sample_dir, edx_list, crack_list, skeleton_list) in enumerate(samples):
        process_sample(sample_dir, edx_list, crack_list, skeleton_list)

    # TODO: this is probably also an input parameter
    selected_sample = '3'

    logger.info(f'BatchRun, selected_sample = {selected_sample}')

    js = '2'
