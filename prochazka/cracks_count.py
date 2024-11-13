import numpy as np
import matplotlib.pyplot as plt


def weightcounts(data, edges):
    counts, edges = np.histogram(data, bins=edges)
    lengths = counts * np.diff(edges)
    return lengths, edges, counts


def cracksCount(cracks, phases):
    phase_pairs = [(int(single_pair[0]), int(single_pair[1]))
                   for single_pair in np.concatenate([pair for pair in [c['OnEdge']["phases_id_pairs"]
                                                                        for c in cracks] if len(pair) != 0])]
    edges = np.vstack([c['OnEdge'] for c in cracks])
    np_max = np.max(edges[:, 2:4])  # první dimenze jsou jednotlivé cracky, druhá dimenze [grain_A, grain_B, phase_A, phase_B, count]
    counts = np.full((np_max, np_max), np.nan)
    lengths = np.copy(counts)

    for ii in range(1, np_max + 1):
        temp = edges[edges[:, 2] == ii]
        e = np.arange(ii, np_max + 2) - 0.5
        l, e, n = weightcounts(temp[:, 3], e)
        b = e[:-1] + np.diff(e) / 2

        counts[b.astype(int) - 1, ii - 1] = n
        lengths[b.astype(int) - 1, ii - 1] = l

    through = np.vstack([c['Through'] for c in cracks])
    l, e, _, n = weightcounts(through[:, 1], np.arange(through[:, 1].max() + 2) - 0.5)
    b = e[:-1] + np.diff(e) / 2

    # plt.close('all')
    #
    # fig1, tc = plt.subplots(3, 3, figsize=(9, 9))
    # fig1.tight_layout(pad=0.4)
    #
    # fig2, tl = plt.subplots(3, 3, figsize=(9, 9))
    # fig2.tight_layout(pad=0.4)

    for ii in range(np_max):
        # next_tile = tc.flat[ii] if ii < 9 else plt.figure()
        counts[ii, :] = counts[:, ii]
        lengths[ii, :] = lengths[:, ii]

        # next_tile.bar(np.arange(1, len(counts) + 1), counts[:, ii])
        # next_tile.set_title(f'Crack on edge of {phases[ii]["Labels"]} with:')
        # next_tile.set_xticklabels([p['Labels'] for p in phases], rotation=45)
        # next_tile.set_ylim([0, 180])

    # tc.flat[-1].bar(np.arange(1, len(counts) + 1), n)
    # tc.flat[-1].set_title('Crack in the phase')
    # tc.flat[-1].set_xticklabels([p['Labels'] for p in phases], rotation=45)
    # tc.flat[-1].set_ylim([0, 720])
    #
    # tl.flat[-1].bar(np.arange(1, len(counts) + 1), counts.sum(axis=1))
    # tl.flat[-1].set_xticklabels([p['Labels'] for p in phases], rotation=45)
    # tl.flat[-1].legend([p['Labels'] for p in phases])

    return counts, lengths, n, l


# Example usage
class Crack:
    def __init__(self, on_edge, through):
        self.OnEdge = on_edge
        self.Through = through


class Phase:
    def __init__(self, labels):
        self.Labels = labels


if __name__ == "__main__":
    # Replace placeholder data with actual data
    cracks = [Crack(np.random.randint(0, 5, (10, 4)), np.random.randint(0, 5, (10, 2))) for _ in range(10)]
    phases = [Phase(f'Phase {i}') for i in range(5)]

    counts, lengths, n, l = cracksCount(cracks, phases)
