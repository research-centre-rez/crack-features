import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, skeletonize
from skimage.measure import label
from scipy.ndimage import binary_dilation, binary_hit_or_miss


def cracksEval(PhaseIMG, Cracks, Phases, ThRadius, InRadius):
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(19.20, 9.73))
    axs[0].imshow(PhaseIMG, cmap='gray')
    axs[1].imshow(PhaseIMG, cmap='gray')

    NC = len(Cracks)
    MI, NI = PhaseIMG.shape

    # Add not-assigned phase (0 in PhaseIMG)
    Phases.append({
        'Mask': PhaseIMG == 0,
        'Label': 'n/a',
        'LABcolor': [0, 0, 0],
        'GRBcolor': [0, 0, 0],
        'AreaPx': np.sum(PhaseIMG == 0)
    })
    LayerID = PhaseIMG.copy()
    LayerID[PhaseIMG == 0] = len(Phases)

    CracksList = list(range(NC))

    for ii in CracksList:
        print(f'{ii + 1} / {NC}')
        Meat = Cracks[ii]['Meat'].toarray()  # Ensure you have sparse matrix support
        Outline = binary_dilation(Meat, structure=disk(ThRadius)) ^ Meat
        Skel = skeletonize(Outline | Meat)
        Cracks[ii]['SkelPxJP'] = np.sum(Skel)

        l, k = np.where(binary_hit_or_miss(Skel))

        if len(l) == 0:
            L, K = np.nonzero(Skel)
            L, K = L[0], K[0]
            Skel[L, K] = 0
            l, k = np.where(binary_hit_or_miss(Skel))
            l = np.append(l, L)
            k = np.append(k, K)
            Sk = np.zeros_like(Skel)
            Sk[l[0], k[0]] = 1
            Sk[l[1], k[1]] = 1
            axs[1].imshow(Outline + Meat + Skel + Sk, cmap='gray')

        EndPoints = np.column_stack((k, l))

        Nods = binary_hit_or_miss(Skel)
        Branches = label(Skel & (~binary_dilation(Nods, structure=disk(2))))
        l, k = np.where(Nods)
        NodPoints = np.column_stack((k, l))

        EndStruct = [{'xy': [k_, l_],
                      'Branches': np.unique(Branches[max(0, l_ - 4):min(MI, l_ + 5), max(0, k_ - 4):min(NI, k_ + 5)])[
                                  1:], 'IsEnd': True, 'IsTerminal': False} for k_, l_ in EndPoints]

        Distances = np.zeros((len(EndPoints), len(EndPoints)))
        for jj, (k_, l_) in enumerate(EndPoints):
            Points = np.array([es['xy'] for es in EndStruct])
            XY = Points - [k_, l_]
            Distances[:, jj] = np.sqrt(np.sum(XY ** 2, axis=1))

        Hits = np.unravel_index(Distances.argmax(), Distances.shape)
        for hit in Hits:
            EndStruct[hit]['IsTerminal'] = True

        NodStruct = [{'xy': [k_, l_],
                      'Branches': np.unique(Branches[max(0, l_ - 4):min(MI, l_ + 5), max(0, k_ - 4):min(NI, k_ + 5)])[
                                  1:], 'IsEnd': False, 'IsTerminal': False} for k_, l_ in zip(k, l)]

        NodStruct.extend(EndStruct)

        NBranch = Branches.max()
        IncidentPoints = np.array([ns['Branches'] for ns in NodStruct]).T
        BranchStruct = [{'Map': Branches == jj + 1, 'Nodes': np.where(IncidentPoints == jj + 1)[0],
                         'xy': np.column_stack(np.nonzero(Branches == jj + 1)), 'OnEdge': [], 'Through': []} for jj in
                        range(NBranch)]

        for jj in range(NBranch):
            BranchStruct[jj] = sortPoints(BranchStruct[jj], NodStruct)
            Phs, Is = sniffSides(BranchStruct[jj], PhaseIMG)
            Phs = Phs[(Phs.min(axis=1) != 0), :]  # Remove zero-phase entries
            BranchStruct[jj]['Through'] = Phs[(Phs[:, 0] == Phs[:, 1])]
            BranchStruct[jj]['OnEdge'] = Phs[(Phs[:, 0] != Phs[:, 1])]

        Through = np.vstack([bs['Through'] for bs in BranchStruct])
        OnEdge = np.vstack([bs['OnEdge'] for bs in BranchStruct])

        Unq = np.unique(Through)
        Unq = np.column_stack((Unq, [Phases[u]['PhaseID'] for u in Unq], [np.sum(Through == u) for u in Unq]))
        Cracks[ii]['Through'] = Unq

        OnEdge = OnEdge[np.argsort(OnEdge, axis=1)]
        Unq = np.unique(OnEdge, axis=0)
        Unq = np.column_stack((Unq, [Phases[Unq[j, 0]]['PhaseID'] for j in range(Unq.shape[0])],
                               [Phases[Unq[j, 1]]['PhaseID'] for j in range(Unq.shape[0])],
                               [np.sum(np.prod(Unq[j, :2] == OnEdge, axis=1)) for j in range(Unq.shape[0])]))
        Cracks[ii]['OnEdge'] = Unq

    Cracks = [Cracks[i] for i in CracksList]
    return Cracks, NodStruct, BranchStruct


def sniffSides(BStruct, Map):
    sN = BStruct['xy'].shape[0]
    I = np.full((sN, 4), np.nan)
    Ph = np.full((sN, 2), np.nan)

    for si in range(sN):
        i1 = max(si - 1, 1)
        i2 = min(si + 1, sN)
        I1 = BStruct['xy'][i2 - 1, ::-1]
        I2 = BStruct['xy'][i1 - 1, ::-1]
        T = I2 - I1
        N = np.matmul(T, [[0, -1], [1, 0]]) / np.linalg.norm(T)
        J0 = BStruct['xy'][si, ::-1]

        IsPhase = False
        kk = 1

        while not IsPhase:
            J1 = np.minimum(np.maximum(J0 - kk * N, [1, 1]), [Map.shape[0], Map.shape[1]])
            J1 = np.round(J1).astype(int) - 1
            if np.all(J0 - kk * N == J1):
                IsPhase = Map[J1[0], J1[1]] != 0
            else:
                IsPhase = True
            kk += 1

        IsPhase = False
        kk = 1

        while not IsPhase:
            J2 = np.minimum(np.maximum(J0 + kk * N, [1, 1]), [Map.shape[0], Map.shape[1]])
            J2 = np.round(J2).astype(int) - 1
            if np.all(J0 + kk * N == J2):
                IsPhase = Map[J2[0], J2[1]] != 0
            else:
                IsPhase = True
            kk += 1

        I[si, :] = [J1[0], J1[1], J2[0], J2[1]]
        Ph[si, :] = [Map[J1[0], J1[1]], Map[J2[0], J2[1]]]

    return Ph, I


def sortPoints(BranchStruct, NodStruct):
    BranchStruct['xy'] = np.array(sorted(BranchStruct['xy'], key=lambda p: -NodStruct[p[0] > p[1]]))
    return BranchStruct


# Example usage, placeholder for constructing Cracks and Phases
class Crack:
    def __init__(self, meat):
        self.Meat = meat
        self.SkelPxJP = None
        self.OnEdge = None
        self.Through = None


class Phase:
    def __init__(self, phaseID, label):
        self.PhaseID = phaseID
        self.Label = label


# You need to replace the following with your actual Cracks and Phases
Cracks = [Crack(np.random.random((10, 10)) > 0.5) for _ in range(10)]
Phases = [Phase(i, f"Phase {i}") for i in range(5)]
PhaseIMG = np.random.randint(0, len(Phases), (100, 100))

# Execute
Cracks, NodStruct, BranchStruct = cracksEval(PhaseIMG, Cracks, Phases, ThRadius=2, InRadius=2)
