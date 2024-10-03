import os
import numpy as np
import datetime
from skimage import io
from skimage.color import rgb2lab
from skimage.filters import gaussian
import matplotlib.pyplot as plt


# Add imports for any additional required packages (e.g., scipy, etc.)

def main_caller(ch_phases):
    # Set up directories
    main_folder = os.path.join('D:', 'OneDrive', 'UJV', 'Halodova Patricie - JCAMP_NRA_LOM_SEM-EDX', 'SEM-EDX')
    samples = [d for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))][2:]

    # Crop range
    crop_range = np.array([[1, 2000], [1, 2000]])

    # Initialize variables
    cracks = np.nan
    cracks_map = np.nan
    layers = np.nan
    map2 = np.nan
    map_ = np.nan
    grain_struct = np.nan
    grain_map = np.nan
    skel = np.nan
    mask = np.nan

    # Plot setup
    fig, (axl, axr) = plt.subplots(1, 2)
    fig.tight_layout(pad=0)
    fig.savefig('tight_layout_fig.png')  # To avoid displaying the plot in script run

    # Initialize colormap and labels
    cmap0 = np.vstack(([0, 0, 0], [ch.colors for ch in ch_phases if ch.detect > 0]))
    labels0 = ["Unassigned"] + [ch.label for ch in ch_phases if ch.detect > 0]

    t0 = datetime.datetime.now()
    i0 = 0
    is_test = '3'

    # Logging
    with open(os.path.join(main_folder, 'log.txt'), 'a') as fid:
        fid.write(f'\n\n{t0}: initializeJB, Is = {is_test}')

    # Run through all samples found
    for ii in [int(i) for i in is_test]:
        ti = datetime.datetime.now()
        i0 += 1

        # Find all EDX layered images for phase analyses
        edx_map_list = sorted(os.listdir(os.path.join(main_folder, samples[ii], 'EDX layered images')))
        edx_map_list = [f for f in edx_map_list if f.endswith('.tif')]

        # Find all cracks images (processed by Jan Blazek)
        mask_list = sorted(os.listdir(os.path.join(main_folder, samples[ii], 'JBmasks')))
        mask_list = [f for f in mask_list if f.endswith('.png')]

        # Find all skeleton images (processed by Jan Blazek)
        skeleton_list = sorted(os.listdir(os.path.join(main_folder, samples[ii], 'JBskeletons')))
        skeleton_list = [f for f in skeleton_list if f.endswith('.png')]

        js_test = '2'

        with open(os.path.join(main_folder, 'log.txt'), 'a') as fid:
            fid.write(f'\n{ti}: ii = {ii} ({i0}/{len([int(i) for i in is_test])}) Js = {js_test}')

        # Run through all images
        j0 = 0
        for jj in [int(j) for j in js_test]:
            tj = datetime.datetime.now()
            j0 += 1

            with open(os.path.join(main_folder, 'log.txt'), 'a') as fid:
                fid.write(
                    f'\n{tj} - {os.path.join(samples[ii], skeleton_list[jj])}: process started, jj={jj} ({j0}/{len([int(j) for j in js_test])})')

            cracks = np.nan
            cracks_map = np.nan
            layers = np.nan
            map2 = np.nan
            map_ = np.nan
            grain_struct = np.nan
            grain_map = np.nan
            skel = np.nan
            mask = np.nan

            try:
                # Read data
                filename, _ = os.path.splitext(edx_map_list[jj])
                map_ = io.imread(os.path.join(main_folder, samples[ii], 'EDX layered images', edx_map_list[jj]))[...,
                       :3]
                map_ = map_[crop_range[0, 0]:crop_range[0, 1], crop_range[1, 0]:crop_range[1, 1]]

                # Read and process crack image
                mask = (io.imread(os.path.join(main_folder, samples[ii], 'JBmasks', mask_list[jj])) > 0).astype(
                    np.double)
                mask = mask[crop_range[0, 0]:crop_range[0, 1], crop_range[1, 0]:crop_range[1, 1]]

                # Read and process skeleton image
                skel = (io.imread(os.path.join(main_folder, samples[ii], 'JBskeletons', skeleton_list[jj])) > 0).astype(
                    np.double)
                skel = skel[crop_range[0, 0]:crop_range[0, 1], crop_range[1, 0]:crop_range[1, 1]]

                if np.max(label(mask)) == 0 or np.max(label(skel)) == 0:
                    with open(os.path.join(main_folder, 'log.txt'), 'a') as fid:
                        fid.write(
                            f'\n{datetime.datetime.now()} - {os.path.join(samples[ii], skeleton_list[jj])}: no cracks')
                else:
                    # Process data
                    map21, layers = phase_thr(map_, ch_phases, mask, 'gauss', 3, 'thr', 5, 'lab')
                    layers, map2 = phase_id(layers, '-spikes', 2, '-ni')

                    grain_map0 = grain_id2(map2, layers)
                    grain_struct, layers = grain_param(grain_map0, map2, layers)
                    grain_struct, grain_map, _, _ = grain_filter(grain_struct, grain_map0, layers)

                    cracks, cracks_map = crack_id(skel, mask)

                    if not os.path.exists(os.path.join(main_folder, samples[ii], 'Stats')):
                        os.makedirs(os.path.join(main_folder, samples[ii], 'Stats'))

                    save_path = os.path.join(main_folder, samples[ii], 'Stats', filename + '.npz')
                    np.savez(save_path, ch_phases=ch_phases, layers=layers, grain_struct=grain_struct, cracks=cracks,
                             grain_map=grain_map, map2=map2, cracks_map=cracks_map)

                    plt.imshow(map2, alpha=1 - mask)
                    plt.colorbar()
                    plt.title('Phases processed')
                    plt.savefig(os.path.join(main_folder, samples[ii], 'EDX_assign', filename + '.png'))

                    t = datetime.datetime.now()
                    with open(os.path.join(main_folder, 'log.txt'), 'a') as fid:
                        fid.write(
                            f'\n{t} - {os.path.join(samples[ii], skeleton_list[jj])}: process ended. Duration: {t - tj}')
            except Exception as e:
                t = datetime.datetime.now()
                with open(os.path.join(main_folder, 'log.txt'), 'a') as fid:
                    fid.write(f'\n{t} - {os.path.join(samples[ii], skeleton_list[jj])}: Error.')

                save_path = os.path.join(main_folder, samples[ii], 'Stats', filename + '_error.npz')
                np.savez(save_path, error=str(e), ch_phases=ch_phases, mask=mask, skel=skel, layers=layers,
                         grain_struct=grain_struct, cracks=cracks, grain_map=grain_map, map2=map2,
                         cracks_map=cracks_map)

        t = datetime.datetime.now()
        with open(os.path.join(main_folder, 'log.txt'), 'a') as fid:
            fid.write(f'\n{t} - {samples[ii]}: Set ended. Duration: {t - ti}')

    t = datetime.datetime.now()
    with open(os.path.join(main_folder, 'log.txt'), 'a') as fid:
        fid.write(f'\n{t}: Batch ended. Duration: {t - t0}')


if __name__ == '__main__':
    # Define your `ChPhases` class/structure and initialize as needed
    ch_phases = []
    main_caller(ch_phases)
