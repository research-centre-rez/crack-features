import numpy as np
import logging

logger = logging.getLogger(__name__)


def _find_path_start(binary_mask):
    """
    Find a pixel with only one neighbor to use as a start pixel.
    """
    path_pixels = np.argwhere(binary_mask == 1)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    for pixel in path_pixels:
        neighbors = 0
        for d in directions:
            neighbor = (pixel[0] + d[0], pixel[1] + d[1])
            if (0 <= neighbor[0] < binary_mask.shape[0] and
                    0 <= neighbor[1] < binary_mask.shape[1] and
                    binary_mask[neighbor] == 1):
                neighbors += 1
        if neighbors == 1:  # Pixel with only one neighbor is an endpoint
            return pixel
    return path_pixels[0]  # Fallback if no endpoint is found (i.e. branch is a cycle)


def sort_branch_pixels(binary_mask):
    """
    Goes thru point set in the binary mask and order non-zero pixel into a path.
    """

    # Get coordinates of path pixels
    path_pixels = np.argwhere(binary_mask == 1)
    # Starting pixel
    start_pixel = _find_path_start(binary_mask)
    # Directions for moving to neighboring pixels (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    # Initialize the list to store ordered pixels and set of visited pixels
    ordered_pixels = []
    visited = set()

    # Depth-First Search (DFS) to follow the path
    def dfs(pixel):
        stack = [pixel]
        while stack:
            px = stack.pop()
            if tuple(px) in visited:
                continue
            visited.add(tuple(px))
            ordered_pixels.append(px)

            # Look in each direction for connected path pixels
            for d in directions:
                neighbor = (px[0] + d[0], px[1] + d[1])

                # Check if the neighbor is within bounds, unvisited, and part of the path
                if (0 <= neighbor[0] < binary_mask.shape[0] and
                        0 <= neighbor[1] < binary_mask.shape[1] and
                        binary_mask[neighbor] == 1 and
                        tuple(neighbor) not in visited):
                    stack.append(neighbor)
        return ordered_pixels

    # Run DFS starting from the initial pixel
    dfs(start_pixel)
    # Convert ordered pixels to a list of tuples
    return [tuple(px) for px in ordered_pixels]


def path_direction(branch_ordered_pixels, mask):
    """
    Computes pixel gradient according to available inputs. See the code.
    """
    if len(branch_ordered_pixels) < 2:  # in this case it is not possible to compute gradient
        crack_nearby = mask[
            np.max([0, branch_ordered_pixels[0][0] - 1]):np.min([branch_ordered_pixels[0][0] + 2, mask.shape[0]]),
            np.max([0, branch_ordered_pixels[0][1] - 1]):np.min([branch_ordered_pixels[0][1] + 2, mask.shape[1]]),
        ]
        nearby_crack_pixels_count = np.where(crack_nearby == 0)[0].size
        if nearby_crack_pixels_count == 1:  # there is no other pixel belonging to a crack
            logger.error("Invalid crack branch (single pixel). No gradient.")
            return None
        elif nearby_crack_pixels_count == 2:  # there is 1 other pixel belonging to a crack
            gradient = np.gradient(np.stack(np.where(crack_nearby == 0), axis=1), axis=0)[:1]
        elif nearby_crack_pixels_count == 3:  # ideal case, there is pixel before and after and gradient can be computed
            gradient = np.gradient(np.stack(np.where(crack_nearby == 0), axis=1), axis=0)[1:2]
        else:  # there is more than two pixels, it is not clear what to use for gradient computation
            return None
    else:
        gradient = np.gradient(branch_ordered_pixels, axis=0)
    return gradient
