import numpy as np
from skimage.morphology import disk, opening
from scipy.ndimage import label
from datetime import datetime


def phaseID(Layers, *args):
    def layers2allMaps(AllLayers):
        NL = len(AllLayers)
        Maps = np.zeros_like(AllLayers[0]['mask'])
        for LMi in range(NL - 1):
            Maps[AllLayers[LMi]['mask'] > 0] = LMi + 1  # MATLAB is 1-indexed, Python is 0-indexed
        return Maps

    def allMaps2Layers(AllLayers, Map):
        Indices = np.unique(Map)
        Indices = Indices[Indices > 0]  # Exclude index 0
        for idx in Indices:
            AllLayers[idx - 1]['mask'] = Map == idx
        AllLayers[-1]['mask'] = Map == 0
        return AllLayers

    def extrapolateNI(ENLayers, ENMap):
        NAcount = np.sum(ENMap < 1)
        while NAcount > 0:
            NAcount1 = np.sum(ENMap < 1)
            coords = np.argwhere(ENMap < 1)
            ENMap = np.pad(ENMap, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            coords += 1
            for eNi in coords:
                K, L = eNi
                neighborhood = ENMap[K - 1:K + 2, L - 1:L + 2].ravel()
                unique_vals, counts = np.unique(neighborhood, return_counts=True)
                if np.any(unique_vals > 0):
                    unique_vals = unique_vals[unique_vals > 0]
                    counts = counts[unique_vals > 0]
                    max_count_val = unique_vals[np.argmax(counts)]
                    ENMap[K, L] = max_count_val
            ENMap = ENMap[1:-1, 1:-1]
            NAcount = NAcount1 - np.sum(ENMap < 1)
        return allMaps2Layers(ENLayers, ENMap), ENMap

    def cleanSpikes(CSLayers, Filter):
        if isinstance(Filter, int):
            Probe = disk(Filter)
        elif isinstance(Filter, (tuple, list)):
            Probe = disk(*Filter)
        else:
            raise ValueError("Unsupported filter type")

        CSN = len(CSLayers)
        for cSi in range(CSN - 1):
            Map = CSLayers[cSi]['mask']
            Map2 = opening(Map, Probe)
            CSLayers[cSi]['mask'] = Map2
            CSLayers[-1]['mask'][np.logical_xor(Map, Map2)] = 1
        return CSLayers

    T0 = datetime.now()
    Nv = max(1, len(args))
    Verbose = [{'Layers': None, 'mask': None, 'Command': ''} for _ in range(Nv)]
    Used = [False] * Nv
    FillCracks = True

    ii = 0
    while ii < len(args):
        if args[ii].lower() in ['-cracks', '-crack', '-cr']:
            FillCracks = False
            Cracks = args[ii + 1]
            ii += 2
        elif args[ii].lower() in ['-spikes', '-cleanspikes']:
            Layers = cleanSpikes(Layers, args[ii + 1])
            TheMap = layers2allMaps(Layers)
            Used[ii] = True
            Verbose[ii].update({'Layers': Layers, 'mask': TheMap, 'Command': args[ii:ii + 2]})
            ii += 2
        elif args[ii].lower() in ['-extrapolateni', '-ni', '-extrapolate']:
            Layers, TheMap = extrapolateNI(Layers, TheMap)
            Used[ii] = True
            Verbose[ii].update({'Layers': Layers, 'mask': TheMap, 'Command': args[ii]})
            ii += 1
        else:
            ii += 1

    Verbose = [v for v, u in zip(Verbose, Used) if u]

    if FillCracks:
        Layers = allMaps2Layers(Layers, TheMap)
    else:
        TheMap[Cracks > 0] = 0
        Verbose.append({'Layers': Layers, 'mask': TheMap, 'Command': ['-cracks', Cracks]})

    T1 = datetime.now()
    print(f'phaseID: {T1 - T0}')
    return Layers, TheMap, Verbose


# Example usage placeholder for Layers
Layers = [{'Label': 'Layer1', 'mask': np.random.randint(0, 2, (100, 100))} for _ in range(3)]
Layers.append({'Label': 'Not assigned', 'mask': np.zeros((100, 100))})

# Execute
Layers, TheMap, Verbose = phaseID(Layers, '-spikes', 3, '-extrapolate')
