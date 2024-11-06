import numpy as np
from skimage.morphology import disk, skeletonize
from skimage.measure import label
from scipy.ndimage import binary_dilation
# bw_morph is python code simulating MATLAB bwmorph lib
from bw_morph import endpoints, branches


def crack_metadata(crack, dilation_radius=5):
    segment_labels = crack["segment_labels"]
    outline = np.logical_xor(
        binary_dilation(segment_labels, structure=disk(dilation_radius)),
        segment_labels)

    outline_skeleton = skeletonize(np.logical_or(outline, segment_labels))
    skeleton_outline_pixel_count = np.sum(outline_skeleton)
    end_points = np.column_stack(np.where(endpoints(outline_skeleton)))
    nodes = branches(outline_skeleton)

    nodes_coords = np.column_stack(np.where(nodes))

    return {
        "segment_labels": segment_labels,
        "outline": outline,
        "outline_skeleton": outline_skeleton,
        "skeleton_outline_pixel_count": skeleton_outline_pixel_count,
        "end_points_coords": end_points,
        "nodes": {
            "mask": nodes,
            "coords": nodes_coords
        }
    }


def point_metadata(x, y, skeleton_branches, image_shape):
    height, width = image_shape
    return {
        'xy': [x, y],
        # branches contains array of branch numbers near [-4,+4] the endpoint
        'Branches': np.unique(skeleton_branches[
                              max(0, y - 4):min(height, y + 5),
                              max(0, x - 4):min(width, x + 5)
                              ])[1:],
        'IsEnd': True,
        'IsTerminal': False
    }


def get_skeleton_branches(skeleton, skeleton_nodes):
    # Create a structuring element (disk with radius 2)
    structuring_element = disk(2)

    # Perform binary dilation on bw_skeleton_nodes
    dilated_nodes = binary_dilation(skeleton_nodes, structuring_element)

    # Subtract the dilated image from bw_skeleton and check where it's greater than 0
    binary_diff = (skeleton - dilated_nodes) > 0

    # Label the regions in the binary_diff image
    return label(binary_diff)


def cracks_evaluation(phase_img, cracks, phases, dilation_radius):
    """
    Processing of the prepared segmentation maps

    @param phase_img - segmentation map from the miscroscope according to pixel-material phase
    @param cracks - list of cracks (@see ... TODO)
    """

    # Add not-assigned phase (0 in phase_img)
    phases.append({
        'Mask': phase_img == 0,
        'Label': 'n/a',
        'LABcolor': [0, 0, 0],
        'GRBcolor': [0, 0, 0],
        'area_pixel_count': np.sum(phase_img == 0)
    })
    layer_ids = phase_img.copy()
    layer_ids[phase_img == 0] = len(phases)

    for crack in cracks:
        metadata = crack_metadata(crack, dilation_radius)
        skeleton_branches, branches_count = get_skeleton_branches(
            metadata['outline_skeleton'],
            metadata['nodes']['mask']
        )
        points = [point_metadata(x, y, skeleton_branches, phase_img.shape)
                  for x, y in metadata["end_points"]]
        points.extend([point_metadata(x, y, skeleton_branches, phase_img.shape)
                       for x, y in metadata["nodes"]["coords"]])

        skeletons_endpoints_distance = [
            np.linalg.norm(metadata["end_points"] - endpoint, axis=1)
            for endpoint in metadata["end_points"]
        ]

        Hits = np.unravel_index(
            skeletons_endpoints_distance.argmax(),
            skeletons_endpoints_distance.shape
        )
        for hit in Hits:
            endpoints[hit]['IsTerminal'] = True

        incident_points = np.array([point['Branches'] for point in points]).T
        branch_struct = [{
            'mask': branches == branch_id,
            'Nodes': np.where(incident_points == branch_id)[0],
            'xy': np.column_stack(np.nonzero(branches == branch_id)),
            'OnEdge': [],
            'Through': []
        } for branch_id in range(branches_count)]

        for branch_id in range(branches_count):
            branch_struct[branch_id] = sortPoints(branch_struct[branch_id], points)
            phases, Is = sniffSides(branch_struct[branch_id], phase_img)
            phases = phases[(phases.min(axis=1) != 0), :]  # Remove zero-phase entries
            branch_struct[branch_id]['Through'] = phases[(phases[:, 0] == phases[:, 1])]
            branch_struct[branch_id]['OnEdge'] = phases[(phases[:, 0] != phases[:, 1])]

        Through = np.vstack([bs['Through'] for bs in branch_struct])
        OnEdge = np.vstack([bs['OnEdge'] for bs in branch_struct])

        Unq = np.unique(Through)
        Unq = np.column_stack((Unq, [phases[u]['PhaseID'] for u in Unq], [np.sum(Through == u) for u in Unq]))
        crack['Through'] = Unq

        OnEdge = OnEdge[np.argsort(OnEdge, axis=1)]
        Unq = np.unique(OnEdge, axis=0)
        Unq = np.column_stack((Unq, [phases[Unq[j, 0]]['PhaseID'] for j in range(Unq.shape[0])],
                               [phases[Unq[j, 1]]['PhaseID'] for j in range(Unq.shape[0])],
                               [np.sum(np.prod(Unq[j, :2] == OnEdge, axis=1)) for j in range(Unq.shape[0])]))
        crack['OnEdge'] = Unq

    return cracks, points, branch_struct


def sniffSides(branch_struct, mask):
    branch_points_count = branch_struct['xy'].shape[0]
    I = np.full((branch_points_count, 4), np.nan)
    Ph = np.full((branch_points_count, 2), np.nan)

    for branch_point_id in range(branch_points_count):
        low_bid = max(branch_point_id - 1, 1)
        high_bid = min(branch_point_id + 1, branch_points_count)
        high_bid_coords = branch_struct['xy'][high_bid - 1, ::-1]
        low_bid_coords = branch_struct['xy'][low_bid - 1, ::-1]
        derivative = low_bid_coords - high_bid_coords
        branch_norm = np.matmul(derivative, [[0, -1], [1, 0]]) / np.linalg.norm(derivative)

        branch_point_xy = branch_struct['xy'][branch_point_id, ::-1]

        is_phase = False
        kk = 1

        # find, in the direction of the norm a phase pixel?
        while not is_phase:
            J1 = np.minimum(
                np.maximum(branch_point_xy - kk * branch_norm, [1, 1]),
                            [mask.shape[0], mask.shape[1]])
            J1 = np.round(J1).astype(int) - 1
            if np.all(branch_point_xy - kk * branch_norm == J1):
                is_phase = mask[J1[0], J1[1]] != 0
            else:
                is_phase = True
            kk += 1

        is_phase = False
        kk = 1

        while not is_phase:
            J2 = np.minimum(np.maximum(branch_point_xy + kk * branch_norm, [1, 1]), [mask.shape[0], mask.shape[1]])
            J2 = np.round(J2).astype(int) - 1
            if np.all(branch_point_xy + kk * branch_norm == J2):
                is_phase = mask[J2[0], J2[1]] != 0
            else:
                is_phase = True
            kk += 1

        I[branch_point_id, :] = [J1[0], J1[1], J2[0], J2[1]]
        Ph[branch_point_id, :] = [mask[J1[0], J1[1]], mask[J2[0], J2[1]]]

    return Ph, I


def sortPoints(BranchStruct, NodStruct):
    BranchStruct['xy'] = np.array(sorted(BranchStruct['xy'], key=lambda p: -NodStruct[p[0] > p[1]]))
    return BranchStruct


# Example usage, placeholder for constructing cracks and phases
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

if __name__ == '__main__':
    # You need to replace the following with your actual cracks and phases
    Cracks = [Crack(np.random.random((10, 10)) > 0.5) for _ in range(10)]
    Phases = [Phase(i, f"Phase {i}") for i in range(5)]
    PhaseIMG = np.random.randint(0, len(Phases), (100, 100))

    # Execute
    Cracks, NodStruct, BranchStruct = cracks_evaluation(PhaseIMG, Cracks, Phases, dilation_radius=2, InRadius=2)
