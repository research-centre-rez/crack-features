import numpy as np
from skimage.color import rgb2lab
from skimage.filters import gaussian
from skimage.measure import label
from skimage.exposure import rescale_intensity, equalize_adapthist
from scipy.ndimage import binary_fill_holes
from skimage import exposure


def phaseTHR(IMG, Phases, *args):
    def imgaussfilt(img, sigma):
        return gaussian(img, sigma=sigma, multichannel=True)

    # Emulate MATLAB's `imadjust` using `skimage`'s `rescale_intensity`
    def imadjust(img, limits=None):
        if limits is None:
            return rescale_intensity(img)
        return rescale_intensity(img, in_range=tuple(limits))

    # Data preallocation
    Nv = max(1, len(args))
    Verbose = [{'Img': None, 'Command': ''} for _ in range(Nv)]
    Used = [False] * Nv

    LABdistance = False
    L_out = [1, 99]

    # Extract phases to be detected
    Phases = [phase for phase in Phases if phase['Detect']]

    Cracks = np.zeros_like(IMG[:, :, 0])
    Np = len(Phases)
    Layers = [{'Map': None, 'Label': 'n/a', 'RGB': np.zeros(3), 'Partition': None, 'Numel': None, 'Count': None} for _
              in range(Np + 1)]

    ii = 0
    while ii < len(args):
        arg = args[ii].lower()
        if arg in ['-ab', '-abdistance', '-record']:
            Used[ii] = True
            Verbose[ii]['Command'] = args[ii]
            Verbose[ii]['Img'] = IMG
            LABdistance = False
            ii += 1
        elif arg in ['-lab', '-labdistance']:
            Used[ii] = True
            Verbose[ii]['Command'] = args[ii]
            Verbose[ii]['Img'] = IMG
            LABdistance = True
            ii += 1
        elif arg == '-gauss':
            Used[ii] = True
            Verbose[ii]['Command'] = args[ii:ii + 2]
            IMG = imgaussfilt(IMG, args[ii + 1])
            Verbose[ii]['Img'] = IMG
            ii += 2
        elif arg in ['-imadjust', '-imadj']:
            Used[ii] = True
            if ii + 1 < Nv and isinstance(args[ii + 1], (tuple, list, np.ndarray)):
                IMG = imadjust(IMG, args[ii + 1])
                Verbose[ii]['Command'] = args[ii:ii + 2]
                ii += 2
            else:
                IMG = imadjust(IMG)
                Verbose[ii]['Command'] = args[ii]
                ii += 1
            Verbose[ii]['Img'] = IMG
        elif arg in ['-thr', '-threshold']:
            if ii + 1 < Nv and isinstance(args[ii + 1], (tuple, list, np.ndarray)):
                if len(args[ii + 1]) == 1:
                    L_out[0] = args[ii + 1][0]
                elif len(args[ii + 1]) == 2:
                    L_out = args[ii + 1]
                else:
                    warning('incorrect threshold values')
                    L_out = args[ii + 1][:2]
                Verbose[ii]['Command'] = args[ii:ii + 2]
                Verbose[ii]['Img'] = IMG
                ii += 2
            else:
                warning('no threshold defined, default used instead.')
                ii += 1
        elif arg == '-cracks':
            Cracks = args[ii + 1]
            Verbose[ii]['Command'] = args[ii:ii + 2]
            Verbose[ii]['Img'] = Cracks
            ii += 2
        else:
            ii += 1

    Cracks = Cracks.astype(float)
    LAB = rgb2lab(IMG)
    Markers = rgb2lab(np.array([phase['Colors'] for phase in Phases]))
    IL = LAB[:, :, 0] * ~Cracks
    IA = LAB[:, :, 2] * ~Cracks
    IB = LAB[:, :, 1] * ~Cracks
    ML = Markers[:, 0]
    MA = Markers[:, 1]
    MB = Markers[:, 2]

    Distances = np.empty((IL.shape[0], IL.shape[1], Np))
    for ii in range(Np):
        if LABdistance:
            Distances[:, :, ii] = np.sqrt((IA - MA[ii]) ** 2 + (IB - MB[ii]) ** 2 + (IL - ML[ii]) ** 2)
        else:
            Distances[:, :, ii] = np.sqrt((IA - MA[ii]) ** 2 + (IB - MB[ii]) ** 2)

    Map = np.argmin(Distances, axis=2) + 1  # MATLAB is 1-indexed, Python is 0-indexed
    Map[IL < L_out[0]] = 0
    Map[IL > L_out[2]] = 0

    for ii in range(Np):
        MapII = Map == ii + 1
        Layers[ii]['Map'] = MapII
        Layers[ii]['Label'] = Phases[ii]['Labels']
        Layers[ii]['RGB'] = Phases[ii]['Colors']
        Layers[ii]['Numel'] = np.sum(MapII)
        Layers[ii]['Partition'] = Layers[ii]['Numel'] / MapII.size
        Layers[ii]['Count'] = np.max(label(MapII))

    MapII = ~Map
    Layers[-1]['Map'] = MapII
    Layers[-1]['Numel'] = np.sum(MapII)
    Layers[-1]['Partition'] = Layers[-1]['Numel'] / MapII.size
    Layers[-1]['Count'] = np.max(label(MapII))

    return Map, Layers, Verbose


# Example usage placeholder for IMG and Phases
IMG = np.random.rand(100, 100, 3)  # Randomly generated sample image
Phases = [{'Detect': True, 'Colors': [255, 0, 0], 'Labels': 'Phase 1'},
          {'Detect': True, 'Colors': [0, 255, 0], 'Labels': 'Phase 2'},
          {'Detect': False, 'Colors': [0, 0, 255], 'Labels': 'Phase 3'}]

# Execute
Map, Layers, Verbose = phaseTHR(IMG, Phases, '-gauss', 1, '-imadjust', '-thr', [5, 95], '-cracks',
                                np.zeros_like(IMG[:, :, 0]))
