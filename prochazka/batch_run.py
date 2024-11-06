import os
import datetime
import numpy as np
import scipy.io
from skimage.io import imread
from skimage.measure import label
import matplotlib.pyplot as plt
import logging
from phase_thr import phase_threshold as phase_thr
from grain_id import grainID2 as grain_id_2
from grain_param import grainParam as grain_param
from grain_filter import grainFilter as grain_filter
from crack_id import crackID as crack_id
from cracks_eval import cracks_evaluation as cracks_eval
from cracks_count import cracksCount as cracks_count
from types import SimpleNamespace

# Testing folder
ROOT = "/Users/gimli/Library/CloudStorage/OneDrive-UJV/JCAMP_NRA_LOM_SEM-EDX/SEM-EDX"
RESULTS = "/Users/gimli/cvr/data/microscopy/SEM-EDX"
# Crop element labels in the bottom part of the image
CROP_RANGE = np.array([[1, 2000], [1, 2000]])


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(RESULTS, 'log.txt'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger()


if __name__ == "__main__":
    # TODO: chPhases
    e10_phases_x6 = scipy.io.loadmat("E10-phases_x6.mat")
    ch_phases = e10_phases_x6['ChPhases']

    main_folder = ROOT

    sample_dirs = sorted([folder
                          for folder in os.listdir(main_folder)
                          if os.path.isdir(os.path.join(main_folder, folder))
                          ])

    fig, ax = plt.subplots()
    axl = ax
    axr = ax

    t0 = datetime.datetime.now()
    i0 = 0
    selected_sample = '3'

    logger.info(f'BatchRun, selected_sample = {selected_sample}')

    for sample_dir in sample_dirs:
        # sample_index = sample_dirs.index(sample_dir)
        ti = datetime.datetime.now()
        i0 += 1
        edx_map_list = []
        for root, dirs, files in os.walk(os.path.join(main_folder, sample_dir, 'EDX layered images')):
            for file in files:
                if file.endswith('.tif') or file.endswith('.tiff'):
                    edx_map_list.append(os.path.join(root, file))
        for root, dirs, files in os.walk(os.path.join(main_folder, sample_dir, 'EDX layered maps')):
            for file in files:
                if file.endswith('.tif') or file.endswith('.tiff'):
                    edx_map_list.append(os.path.join(root, file))
        edx_map_list = sorted(edx_map_list)

        if not os.path.isdir(os.path.join(main_folder, sample_dir, 'JBmasks')):
            continue

        mask_list = sorted(
            [file for file in os.listdir(os.path.join(main_folder, sample_dir, 'JBmasks')) if file.endswith('.png')])
        skeleton_list = sorted([file for file in os.listdir(os.path.join(main_folder, sample_dir, 'JBskeletons')) if
                                file.endswith('.png')])

        js = '2'

        logger.info(f'\n{ti}: sample_index = {sample_dirs.index(sample_dir)} ({i0}/{len(sample_dirs)}) js = {js}')

        for idx, jj in enumerate(js):
            tj = datetime.datetime.now()
            j0 = idx + 1
            jj = int(jj) - 1

            logger.info(f'\n{tj} - {sample_dir}/{skeleton_list[jj]}: process started, jj={jj} ({j0}/{len(js)})')

            file_name = edx_map_list[jj].split('.')[0]
            stats_dir = os.path.join(RESULTS, sample_dir, 'Stats')
            os.makedirs(stats_dir, exist_ok=True)
            save_path = os.path.join(stats_dir, f'{file_name}.npz')

            try:
                map_path = os.path.join(main_folder, sample_dir, 'EDX layered images', edx_map_list[jj])
                sample_map = imread(map_path)
                map_size = sample_map.shape
                CROP_RANGE[:, 1] = np.minimum(CROP_RANGE[:, 1], map_size[:2])
                sample_map = sample_map[CROP_RANGE[0, 0]:CROP_RANGE[0, 1], CROP_RANGE[1, 0]:CROP_RANGE[1, 1], :3]

                mask_path = os.path.join(main_folder, sample_dir, 'JBmasks', mask_list[jj])
                mask = (imread(mask_path) > 0).astype(float)
                mask = mask[CROP_RANGE[0, 0]:CROP_RANGE[0, 1], CROP_RANGE[1, 0]:CROP_RANGE[1, 1]]

                skel_path = os.path.join(main_folder, sample_dir, 'JBskeletons', skeleton_list[jj])
                skel = (imread(skel_path) > 0).astype(float)
                skel = skel[CROP_RANGE[0, 0]:CROP_RANGE[0, 1], CROP_RANGE[1, 0]:CROP_RANGE[1, 1]]

                if np.max(label(mask)) == 0 or np.max(label(skel)) == 0:
                    logger.info(f'\n{datetime.datetime.now()} - {sample_dir}/{skeleton_list[jj]}: no cracks')

                args = SimpleNamespace(**{
                    "cracks": mask,
                    "gaussian_blur": 3,
                    "threshold": 5,
                    "lab_distance": True,
                    "adjust_intensity": None
                })
                thresholed_phase_map, layers, verbose = phase_thr(sample_map, ch_phases, args)

                grain_map0 = grain_id_2(thresholed_phase_map)
                grain_struct, layers = grain_param(grain_map0, thresholed_phase_map, layers)
                grain_struct, grain_map, _, size_limit = grain_filter(grain_struct, grain_map0, layers)

                thresholed_phase_map[mask == 1] = 0
                cracks, cracks_map = crack_id(skel, mask)
                cracks = cracks_eval(grain_map, cracks, grain_struct, 3, 2)
                en, el, in_, il = cracks_count(cracks, ch_phases)

                np.savez(save_path, ch_phases=ch_phases, layers=layers, grain_struct=grain_struct, cracks=cracks,
                         grain_map=grain_map, map2=thresholed_phase_map, cracks_map=cracks_map)

                t = datetime.datetime.now()
                logger.info(f'\n{t} - {sample_dir}/{skeleton_list[jj]}: process ended. Duration: {t - tj}')

            except Exception as e:
                t = datetime.datetime.now()
                logger.info(f'\n{t} - {sample_dir}/{skeleton_list[jj]}: Error {e}')

        t = datetime.datetime.now()
        logger.info(f'{t} - {sample_dir}: Set ended. Duration: {t - ti}')

    t = datetime.datetime.now()
    logger.info(f'{t}: Batch ended. Duration: {t - t0}')
