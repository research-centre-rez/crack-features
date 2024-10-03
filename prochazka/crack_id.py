import numpy as np
from scipy.sparse import csr_matrix
from skimage.measure import label
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist


def crackID(skeleton, meat):
    # Enumerate the cracks
    meat = label(meat)

    # Align Skeleton enumeration with cracks enumeration
    skeleton = meat * skeleton.astype(float)

    # Count individual cracks
    n = np.max(meat)

    # Declare the cracks structure
    cracks = [dict(AreaPx=np.nan,
                   SkelPxJB=np.nan,
                   SkelPxJP=np.nan,
                   WidthPx=np.nan,
                   Meat=csr_matrix(meat.shape),
                   Skel=csr_matrix(meat.shape)) for _ in range(n)]

    for ii in range(1, n + 1):
        cracks[ii - 1]['AreaPx'] = np.sum(meat == ii)
        cracks[ii - 1]['SkelPxJB'] = np.sum(skeleton == ii)
        cracks[ii - 1]['WidthPx'] = cracks[ii - 1]['AreaPx'] / cracks[ii - 1]['SkelPxJB']
        cracks[ii - 1]['Meat'] = csr_matrix(meat == ii)
        cracks[ii - 1]['Skel'] = csr_matrix(skeleton == ii)

    return cracks, meat


def connect_group(group):
    group = binary_dilation(group) ^ group
    group = label(group)
    cgn = np.max(group)

    while cgn > 1:
        piece = group == 1
        rest = np.logical_xor(piece, group)
        p, r = find_closest(piece, rest)
        group = piece + 2 * rest
        cgn = 1
        # You could use matplotlib to plot the line if needed
        # plt.plot([p, r], '*', color='m')

    return group


def find_closest(piece, rest):
    p_coords = np.column_stack(np.where(piece > 0))
    r_coords = np.column_stack(np.where(rest > 0))
    distances = cdist(p_coords, r_coords)

    p0, r0 = np.unravel_index(distances.argmin(), distances.shape)
    p = p_coords[p0]
    r = r_coords[r0]

    return p, r
