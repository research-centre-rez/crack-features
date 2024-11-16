import numpy as np


def weight_counts(data, bin_edges, normalization='count'):
    """
    Calculate weighted counts and optionally normalize them.

    Parameters:
    data (array-like): Input data to bin.
    bin_edges (array-like): Edges of the bins.
    normalization (str): Normalization method: 'count' or 'probability'.

    Returns:
    total_lengths (numpy.ndarray): Total lengths (or sums of values) in each bin.
    bin_edges (numpy.ndarray): Edges of the histogram bins.
    bin_centers (numpy.ndarray): Centers of the histogram bins.
    counts (numpy.ndarray): Counts of data points in each bin.
    """
    # Compute histogram counts
    counts, bin_edges = np.histogram(data, bins=bin_edges)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initialize total lengths (or weighted sums)
    total_lengths = np.zeros_like(counts, dtype=float)

    # Calculate total lengths/values in each bin using vectorized operations
    for i in range(len(counts)):
        in_bin = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
        total_lengths[i] = np.sum(data[in_bin])

    # Normalization
    if normalization == 'probability':
        total_count = np.sum(counts)
        if total_count > 0:
            counts = counts / total_count  # Normalize counts to probabilities
            total_lengths = total_lengths / total_count  # Normalize lengths to average weights
    elif normalization == 'count':
        # No normalization needed, already counts
        pass
    else:
        raise ValueError(f"Unsupported normalization method: {normalization}")

    return total_lengths, bin_edges, bin_centers, counts



def cracksCount(cracks, phases):
    phase_pairs = [(int(single_pair[0]), int(single_pair[1]))
                   for single_pair in np.concatenate([pair for pair in [c['OnEdge']["phases_id_pairs"]
                                                                        for c in cracks] if len(pair) != 0])]

    # Concatenates all OnEdge fields from Cracks into a single matrix.
    # Each row likely contains information about a crack edge.
    all_on_edge_fields = np.vstack([c['OnEdge'] for c in cracks])
    # Number of unique phase labels on edges.
    # TODO: Maybe unique will handle this better way
    # First dimension denotes cracks, second contains columns: [grain_A, grain_B, phase_A, phase_B, counts]
    phases_count = np.max(all_on_edge_fields[:, 2:4])
    # Count cracks on edges
    counts = np.full((phases_count, phases_count), np.nan)
    # ?
    lengths = np.copy(counts)

    for phase_id in range(phases_count):
        # Edges which has phase_A (left) equal to ii
        temp = all_on_edge_fields[all_on_edge_fields[:, 2] == phase_id]
        # Bin positions for phase accumulator
        e = np.arange(phase_id, phases_count + 2) - 0.5
        l, e, n = weight_counts(temp[:, 3], e)
        b = e[:-1] + np.diff(e) / 2

        counts[b.astype(int) - 1, phase_id - 1] = n
        lengths[b.astype(int) - 1, phase_id - 1] = l

    through = np.vstack([c['Through'] for c in cracks])
    l, e, _, n = weight_counts(through[:, 1], np.arange(through[:, 1].max() + 2) - 0.5)
    b = e[:-1] + np.diff(e) / 2

    for phase_id in range(phases_count):
        # next_tile = tc.flat[ii] if ii < 9 else plt.figure()
        counts[phase_id, :] = counts[:, phase_id]
        lengths[phase_id, :] = lengths[:, phase_id]

    return counts, lengths, n, l
