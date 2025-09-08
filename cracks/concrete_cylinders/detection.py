from argparse import ArgumentParser
import imageio.v3 as iio
import os
import pandas as pd
import numpy as np
import cv2
from skimage.measure import label
from scipy.optimize import minimize
from tqdm.auto import tqdm
from skimage.morphology import disk, closing
from skimage.filters.rank import median
from skimage.morphology import binary_erosion
from cracks import features

INPUT_TYPE = "smallv1"
# Empirical value
CIRCULARITY_THRESHOLD = 1.7
# TODO: explain these values, they are artificial due to normalization and should be "debugged somehow"
NORMALIZATION_THRESHOLD_LOWER = 15
NORMALIZATION_THRESHOLD_UPPER = -15
BOUNDARY_EROSION_DISK_SIZE = 5
RAW_MASK_UPPER_THRESHOLD = 10

def adaptive_closing(stack, threshold=RAW_MASK_UPPER_THRESHOLD):
    # Adaptive closing
    # - increases size of the circular kernel until only one segment remains
    morph_size = 1
    intensity_threshold = threshold
    raw_mask = np.min(stack, axis=0) < intensity_threshold
    closed = np.copy(raw_mask)
    while np.unique(label(1 - closed)).size > 2:
        morph_size = morph_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
        # Perform closure
        closed = cv2.morphologyEx((np.min(stack, axis=0) >= intensity_threshold).astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return closed, morph_size, raw_mask

def fit_circle(mask, min_radius=500):
    def circle_error(params):
        center_x, center_y, radius = params
        y, x = np.where(mask == 1)
        pos = np.sum((x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2)
        y, x = np.where(mask == 0)
        neg = np.sum((x - center_x) ** 2 + (y - center_y) ** 2 <= (radius - 1) ** 2)
        return pos + neg

    opt = minimize(circle_error,
                   x0=[mask.shape[1] / 2, mask.shape[0] / 2, min_radius * 1.125],
                   bounds=((min_radius, mask.shape[1] - min_radius),
                           (min_radius, mask.shape[0] - min_radius),
                           (min_radius, min_radius * 1.25)),
                   method="COBYLA")
    return opt

def circular_mask(stack_file, min_threshold=RAW_MASK_UPPER_THRESHOLD):
    try:
        stack = np.load(stack_file)
        mask, kernel_size, raw_mask = adaptive_closing(stack, min_threshold)
        circle = fit_circle(mask)
        circle_mask = np.zeros_like(mask)

        for x in np.arange(circle_mask.shape[1]):
            for y in np.arange(circle_mask.shape[0]):
                if (x - circle.x[0]) ** 2 + (y - circle.x[1]) ** 2 <= circle.x[2] ** 2:
                    circle_mask[y, x] = 1
        return [raw_mask, mask, circle_mask]
    except Exception as e:
        print(f"{os.path.basename(stack_file)}: processing failed with {e}")
        return [None, None, None]


def normalize_lightness(stack_file, layer):
    stack = np.load(stack_file)
    max = np.max(stack, axis=0)
    min = np.min(stack, axis=0)
    # Tady je otázka čím by se to mělo rozmazávat ... velikost disku je poměrně zásadní pro finální výsledek
    # Velikost zřejmě souvisí s velikostí objektů, které se mají ve výsledku detekovat, tj. bude nutné ji nastavit podle typu vzorků
    # Nabízí se otázka jak tento parametr určit
    max_med = median(max, footprint=disk(51), mask=layer[2])
    min_med = median(min, footprint=disk(51), mask=layer[2])

    return max.astype(float) - max_med.astype(float) - (min.astype(float) - min_med.astype(float))


def classify_defects(cracks_and_holes, layer, base_name=None):
    segments = []
    # TODO: Fix this. It is slow an imprecise.
    # np.where to chce řešit nějak jinak, moc dlouho to trvá procházet pixel po pixelu, chce to maticovou operaci
    for l in tqdm(np.arange(1, np.max(cracks_and_holes)), total=np.max(cracks_and_holes) - 1,
                  desc=f"cracks and holes features {base_name}"):
        x, y = np.where(cracks_and_holes == l)
        segments.append([l, np.max(x), np.min(x), np.max(y), np.min(y), len(x)])
    # tento přístup řeší pouze výšku a šířku nejvzdálenějších pixelů
    circularity = [(segment_id, (((maxx - minx + 1) + (maxy - miny + 1)) / 4) ** 2 * np.pi / pixel_count)
                   for segment_id, maxx, minx, maxy, miny, pixel_count in segments]

    # This part is OKish
    border = binary_erosion(layer[2], disk(BOUNDARY_EROSION_DISK_SIZE))
    broken_edge_segments = set(np.unique(cracks_and_holes * (layer[2] - border)).tolist())
    circular = set([seg_id for seg_id, ratio in circularity if ratio < CIRCULARITY_THRESHOLD]) - broken_edge_segments
    non_circular = set([seg_id for seg_id, ratio in circularity if ratio >= CIRCULARITY_THRESHOLD]) - broken_edge_segments

    cracks_A = np.isin(cracks_and_holes, list(non_circular))
    pores = np.isin(cracks_and_holes, list(circular))
    boundary = np.isin(cracks_and_holes, list(broken_edge_segments)[1:])
    inner = closing(layer[2] - np.isin(cracks_and_holes, list(broken_edge_segments)[1:]), disk(8))
    cracks_B = np.logical_and(boundary, inner)
    cracks = np.logical_or(cracks_A, cracks_B)
    # TODO: boundary is 1px thin line closest to the center of the cylinder
    boundary = np.logical_and(boundary, np.logical_not(inner))

    return cracks, pores, boundary

if __name__ == '__main__':
    argparse = ArgumentParser(
        description="""Creation of crack feature vectors for cylindrical concrete samples scanned 360 degrees.""")
    argparse.add_argument("-i", "--inputs_folder", type=str, help="Path to folder with input images.", required=True)
    argparse.add_argument(
        "-t",
        "--type",
        type=str,
        help="Type of input images: {jumbo, smallv1, small_bad}",
        default=INPUT_TYPE,
        choices=list(["smallv1", "jumbo", "small_bad"]))
    argparse.add_argument("-o", "--output_folder", type=str, help="Path to folder where output files will be stored.")
    argparse.add_argument("-v", "--verbose", help="Loglevel.", action="store_true")
    args = argparse.parse_args()

    if args.output_folder is None:
        args.output_folder = args.inputs_folder

    metadata = pd.read_csv(os.path.join(args.inputs_folder, "metadata.csv"), header=None, names=["file path", "type"])
    # go thru directory and select only samples "smallv1"
    stack_files = metadata[metadata["type"] == args.type]["file path"].values.tolist()
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in stack_files]

    if args.verbose:
        print(f"Found {len(stack_files)} files:")
        for f in base_names:
            print(f"\t{f}")

    rows = np.ceil(len(stack_files) / 2).astype(int)
    # Computation of circular mask
    layers = [circular_mask(stack_file) for i, stack_file in tqdm(enumerate(stack_files), total=len(stack_files), desc="Cylinder segmentation")]
    # Lightness normalization
    normalized_stacks = [normalize_lightness(stack_file, layer)
                         for stack_file, layer in tqdm(zip(stack_files, layers), desc="Lightness normalization")
                         if layer[0] is not None]
    # Cracks and holes
    cracks_and_holes = [
        label(np.logical_or(normalized_stack * layer[2] > 15, normalized_stack * layer[2] < -15))
        for normalized_stack, layer in tqdm(zip(normalized_stacks, layers), desc="Rough defect segmentation")
        if layer[0] is not None
    ]
    # crack and holes features
    cracks_pores_boundaries = [classify_defects(ch, layer, base_name)
                               for ch, layer, base_name in tqdm(zip(cracks_and_holes, layers, base_names),
                                                                desc="Defect classification")
                               if layer[0] is not None]

    base_names_filtered = [bn for bn, layer in zip(base_names, layers) if layer[0] is not None]
    for masks, base_name in tqdm(zip(cracks_pores_boundaries, base_names_filtered), desc="Writing features into files"):
        for mask, mask_name in zip(masks, ["cracks", "pores", "boundary"]):
            iio.imwrite(os.path.join(args.output_folder, f"{base_name}-{mask_name}-mask.png"), mask.astype(np.uint8) * 255)
            if np.sum(mask) > 0:
                pd.DataFrame(features.compute(mask)).sort_values(by="skeleton_px", ascending=False).to_csv(os.path.join(args.output_folder, f"{base_name}-{mask_name}.csv"))
            else:
                print(f"{base_name} has no {mask_name}.")
