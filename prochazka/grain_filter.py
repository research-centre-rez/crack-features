import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from datetime import datetime


def grainFilter(GrainStruct, GrainMap, Layers, SizeLimit=9000):
    plt.close('all')
    T0 = datetime.now()

    Sizes = np.array([grain['Numel'] for grain in GrainStruct])
    Indices = np.argsort(np.where(Sizes >= SizeLimit)[0])
    Bins = np.logspace(0, 6, 25)

    Bars1 = weight_counts(Sizes, Bins, normalization='probability')

    kk = 0
    for ii in range(len(Sizes)):
        if ii in Indices:
            kk += 1
            GrainStruct[ii]['ID'] = kk
            GrainMap[GrainMap == ii + 1] = kk  # MATLAB is 1-indexed, Python is 0-indexed
        else:
            GrainMap[GrainMap == ii + 1] = 0

    GrainStruct = [GrainStruct[i] for i in Indices]

    GrainStruct.append({
        'ID': len(GrainStruct) + 1,
        'PhaseID': 7,
        'Label': 'Matrix',
        'Numel': np.sum(GrainMap < 1),
        'Edge': np.sum(np.logical_xor(GrainMap < 1, binary_erosion(GrainMap < 1))),
        'E2A': np.sum(np.logical_xor(GrainMap < 1, binary_erosion(GrainMap < 1))) / np.sum(GrainMap < 1)
    })

    GrainMap[GrainMap == 0] = len(GrainStruct)

    for layer in Layers:
        layer['Children'] = np.unique(GrainMap[layer['mask'] > 0])

    Bars2 = weight_counts(np.array([grain['Numel'] for grain in GrainStruct]), Bins, normalization='probability')
    X = Bins[:-1] + np.diff(Bins)

    for bar1, bar2 in zip(Bars1, Bars2):
        plt.plot(X, [bar1, bar2], marker='.')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([SizeLimit, SizeLimit], list(plt.ylim()))

    T1 = datetime.now()
    print(f'grainFilter: {T1 - T0}')


def weight_counts(data, bins, normalization=None):
    counts, _ = np.histogram(data, bins=bins)
    if normalization == 'probability':
        counts = counts / len(data)
    return counts


# Example usage, placeholder for constructing GrainStruct and Layers
class Grain:
    def __init__(self, numel=0):
        self.Numel = numel


class Layer:
    def __init__(self, map_data):
        self.Map = map_data
        self.Children = []


if __name__ == "__main__":
    # You need to replace the following with your actual GrainStruct and Layers
    GrainStruct = [{'Numel': np.random.randint(1000, 10000)} for _ in range(10)]
    GrainMap = np.random.randint(0, len(GrainStruct), (100, 100))
    Layers = [{'mask': np.random.randint(0, 2, (100, 100))} for _ in range(5)]

    # Execute
    grainFilter(GrainStruct, GrainMap, Layers, SizeLimit=9000)
