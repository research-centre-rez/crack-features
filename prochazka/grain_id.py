import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, opening, dilation, label
from skimage.color import label2rgb
from datetime import datetime


def grainID2(Map, Layers):
    T0 = datetime.now()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(Map, cmap='gray')

    Probe1 = disk(10)
    Probe2 = disk(5)
    GrainSet = np.zeros_like(Map, dtype=int)

    LastIndex = 0

    unique_vals = np.unique(Map)

    # Cycle through phases
    for ii in unique_vals:
        # Label individual grains in ii-th phase
        Labels = label(Map == ii)

        cmap = plt.cm.jet
        cmap.set_bad(color='black')

        # Cycle through the grains
        for jj in range(1, Labels.max() + 1):
            # Get the jj-th grain
            Grain = Labels == jj

            # Shrink the grain to erase bottlenecks
            Grain2 = opening(Grain, Probe1)

            # Label residual subgrains
            SubGrains = label(Grain2)
            TheGrains = np.zeros_like(Map, dtype=int)

            if SubGrains.max() == 1:
                TheGrains = label(Grain)
                ax1.imshow(TheGrains, cmap=cmap)
                plt.draw()
            else:
                # Cycle through detected subgrains
                for kk in range(1, SubGrains.max() + 1):
                    SubGrain = SubGrains == kk
                    SubGrain = dilation(SubGrain, Probe2)
                    ax1.imshow(SubGrain & Grain, cmap=cmap)
                    TheGrains[SubGrain & Grain] = kk
                    plt.draw()

                Leftovers = label(TheGrains == 0)

                # Cycle through missing areas
                for kk in range(1, Leftovers.max() + 1):
                    Chunk = Leftovers == kk
                    Outline = np.logical_and(Chunk, ~dilation(Chunk, disk(2)))
                    Hits = TheGrains * Outline
                    Indices = np.unique(Hits)
                    Indices = Indices[Indices > 0]

                    hist = np.bincount(Hits.ravel())
                    sorted_inds = np.argsort(hist[Indices])[::-1]

                    if sorted_inds.size > 0:
                        TheGrains[Chunk] = Indices[sorted_inds[0]]
                    else:
                        TheGrains[Chunk] = 1

            TheGrains[TheGrains <= 0] = np.nan
            TheGrains += LastIndex
            TheGrains[np.isnan(TheGrains)] = 0
            GrainSet += TheGrains
            LastIndex = np.nanmax(GrainSet)
            ax2.imshow(GrainSet, cmap=cmap)
            plt.draw()

    T1 = datetime.now()
    print(f'Elapsed Time: {T1 - T0}')
    plt.show()
    return GrainSet


# Example usage placeholder for Map and Layers
# You need to replace the following with your actual data
Map = np.random.randint(0, 5, (100, 100))  # Randomly generated sample data
Layers = [{'Map': np.random.randint(0, 2, (100, 100))} for _ in range(5)]

# Execute
GrainSet = grainID2(Map, Layers)
