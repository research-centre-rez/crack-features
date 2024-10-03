import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import binary_erosion
from datetime import datetime
import concurrent.futures


def calculate_grain_parameters(jj, GrainMap, ii, Label):
    Grain = GrainMap == jj
    Area = np.sum(Grain)
    Edge = np.sum(np.logical_xor(Grain, binary_erosion(Grain)))
    return {
        'ID': jj,
        'PhaseID': ii,
        'Label': Label,
        'Numel': Area,
        'Edge': Edge,
        'E2A': Edge / Area
    }


def grainParam(GrainMap, PhMap, Layers):
    T0 = datetime.now()
    NG = np.max(GrainMap)
    unique_phases = np.unique(PhMap)

    # Initialize GrainStruct dictionary
    GrainStruct = {ii: {'ID': None, 'PhaseID': None, 'Label': "", 'Numel': None, 'Edge': None, 'E2A': None} for ii in
                   range(1, NG + 1)}

    for ii in unique_phases:
        TempMap = (PhMap == ii) * GrainMap
        TempIDs = np.unique(TempMap)
        TempIDs = TempIDs[TempIDs != 0]
        Label = Layers[ii]['Label']
        Layers[ii]['Children'] = TempIDs

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda jj: calculate_grain_parameters(jj, GrainMap, ii, Label), TempIDs))

        for result in results:
            GrainStruct[result['ID']].update(result)

        T1 = datetime.now()
        print(f'grainParam {ii}/{len(unique_phases)}: {T1 - T0}')

    T1 = datetime.now()
    print(f'grainParam: {T1 - T0}')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda ii: update_grain_id(GrainStruct, ii), GrainStruct.keys())

    T1 = datetime.now()
    print(f'grainParam: {T1 - T0}')
    return GrainStruct, Layers


def update_grain_id(GrainStruct, ii):
    GrainStruct[ii]['ID'] = ii


# Example usage placeholder for GrainMap, PhMap, and Layers
# You need to replace the following with your actual data
GrainMap = np.random.randint(1, 10, (100, 100))  # Randomly generated sample data
PhMap = np.random.randint(1, 3, (100, 100))  # Randomly generated sample phase map
Layers = [{'Label': f'Layer {i}'} for i in range(3)]

# Execute
GrainStruct, Layers = grainParam(GrainMap, PhMap, Layers)
