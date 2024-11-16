import imageio.v3 as iio
import numpy as np
import os
import pickle
from skimage.measure import label
from skimage.segmentation import flood_fill, mark_boundaries
from skimage.morphology import medial_axis
from tqdm.auto import tqdm
import pandas as pd
import cv2

ROOT_DIR = ""
OUT_DIR = ""
MID = 95
LEVELS = (15, 65, 140, 230)

def _replace(origin, rootdir, outdir, suffix_new):
    suffix_old = os.path.splitext(origin)[1]
    return origin.replace(rootdir, outdir).replace(suffix_old, suffix_new)

def _segmented(origin, rootdir=ROOT_DIR, outdir=OUT_DIR):
    return _replace(origin, rootdir, outdir, "segmented.png")

def _cache(origin, rootdir=ROOT_DIR, outdir=OUT_DIR):
    return _replace(origin, rootdir, outdir, "-cache.pkl")

def _maskfile(origin, rootdir=ROOT_DIR, outdir=OUT_DIR):
    return _replace(origin, rootdir, outdir, "-mask.png")

def _skeletonfile(origin, rootdir=ROOT_DIR, outdir=OUT_DIR):
    return _replace(origin, rootdir, outdir, "-skeleton.png")

def _csvfile(origin, rootdir=ROOT_DIR, outdir=OUT_DIR):
    return _replace(origin, rootdir, outdir, "-data.csv")


def _load_cache(file, rootdir, outdir):
    if os.path.isfile(_cache(file, rootdir, outdir)):
        return pickle.load(open(_cache(file, rootdir, outdir), "rb"))
    else:
        return None


def _crack_bounding_box(labels):
    ones = np.where(labels)
    left_top = (np.min(ones[0]), np.min(ones[1]))
    # right_bottom = (np.max(ones[0])+1, np.max(ones[1])+1)
    bounding_box = labels[left_top[0]:np.max(ones[0]) + 1, left_top[1]: np.max(ones[1]) + 1]
    return bounding_box, left_top


def normalize_hist(img, mid_target=MID):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    im = clahe.apply(img[:, :, 0])
    img = np.stack([im, im, im], axis=2)
    out = np.copy(img)
    mid = np.argmax(np.histogram(img[:, :, 0].reshape(-1), bins=256)[0])
    out[out <= mid] = img[img <= mid] / mid * mid_target
    out[out > mid] = (out[out > mid] - mid) / (255 - mid) * (255 - mid_target) + mid_target
    return out


def crop_image(img, crops):
    """
    @param img: should be filename (full path) or numpy array with minimum 3 dims
    @param crops: tuple of three crops (min, max) per dimension
    returns: cropped image
    """
    if isinstance(img, str):
        return iio.imread(img)[crops[0][0]:crops[0][1], crops[1][0]: crops[1][1], crops[2][0]:crops[2][1]]
    else:
        return img[crops[0][0]:crops[0][1], crops[1][0]: crops[1][1], crops[2][0]:crops[2][1]]

def threshold_image(img, levels=LEVELS):
    t1, t2, t3, t4 = levels
    colored = np.copy(img)
    colored[img[:, :, 0] <= 15] = np.array([0, 0, 0])
    colored[np.logical_and(img[:, :, 0] > t1, img[:, :, 0] <= t2)] = np.array([255, 50, 50])
    colored[np.logical_and(img[:, :, 0] > t2, img[:, :, 0] <= t3)] = np.array([200, 200, 200])
    colored[np.logical_and(img[:, :, 0] > t3, img[:, :, 0] <= t4)] = np.array([0, 255, 50])
    colored[img[:, :, 0] >= t4] = np.array([255, 255, 255])
    return colored


def process_file(file, rootdir=ROOT_DIR, outdir=OUT_DIR, levels=LEVELS, recompute=False, crop=None):
    print(os.path.dirname(file).replace(rootdir, outdir))
    os.makedirs(os.path.dirname(file).replace(rootdir, outdir), exist_ok=True)
    cache = _load_cache(file, rootdir, outdir)

    # TODO: load only if necessary
    tif = iio.imread(file).astype(int)
    if len(tif.shape) == 2:
        tif = np.stack([tif,tif,tif], axis=2)
        if np.max(tif) > 255:
            tif = (tif // 256).astype(np.uint8)
    else:
        if tif.shape[2] == 4:
            tif = tif[:,:,:3]
    if (crop is not None and "crop" not in cache) or ("crop" in cache and cache["crop"] != crop):
        cache["crop"] = crop
        recompute = True
        tif = tif[:crop[0], :crop[1], :]

    # THRESHOLDS
    if recompute or cache is None or "cimg" not in cache or cache["levels"] != levels:
        tif = normalize_hist(tif, MID)
        colored = threshold_image(tif, levels)
        cache = {
            "name": file,
            "tif": tif.astype(np.uint8),
            "cimg": colored.astype(np.uint8),
            "levels": levels
        }
        pickle.dump(cache, open(_cache(file, rootdir, outdir), "wb"))
        recompute = True
    else:
        colored = cache["cimg"]

    if recompute or not os.path.isfile(_segmented(file, rootdir, outdir)):
        iio.imwrite(_segmented(file, rootdir, outdir), colored.astype(np.uint8))

    # MASK
    if recompute or "mask" not in cache:
        seeds = np.where(cache["tif"][:, :, 0] <= levels[0])
        presegment = np.zeros((cache["tif"].shape[0], cache["tif"].shape[1]), np.uint8)
        presegment[cache["tif"][:, :, 0] <= levels[1]] = 1

        for seed in tqdm(zip(seeds[0], seeds[1]), total=len(seeds[0]), desc="mask seeds", leave=False):
            if presegment[seed[0], seed[1]] == 1:
                flood_fill(presegment, seed, 2, in_place=True, connectivity=1.9)
        segmented = np.copy(presegment)
        segmented[segmented == 1] = 0
        segmented[segmented == 2] = 1
        cache["mask"] = segmented
        pickle.dump(cache, open(_cache(file, rootdir, outdir), "wb"))
        recompute = True
    else:
        segmented = cache["mask"]
    # MASK image
    if recompute or not os.path.isfile(_maskfile(file, rootdir, outdir)):
        iio.imwrite(_maskfile(file, rootdir, outdir), (segmented * 255).astype(np.uint8))

    # SKELETON
    if recompute or "skeleton" not in cache or "distance" not in cache:
        skeleton, distance = medial_axis(cache["mask"], return_distance=True)
        cache["skeleton"] = skeleton
        cache["distance"] = distance
        pickle.dump(cache, open(_cache(file, rootdir, outdir), "wb"))
        recompute = True
    # SKELETON image
    if recompute or not os.path.isfile(_skeletonfile(file, rootdir, outdir)):
        iio.imwrite(_skeletonfile(file, rootdir, outdir), (cache["skeleton"] * 255).astype(np.uint8))

    # LABELS
    if recompute or "labels" not in cache:
        cache["labels"] = label(cache["mask"])
        pickle.dump(cache, open(_cache(file, rootdir, outdir), "wb"))
        recompute = True
    if recompute or "label_stats" not in cache:
        values, counts = np.unique(cache["labels"], return_counts=True)
        cache["label_stats"] = {}
        for value, count in tqdm(zip(values, counts), leave=False, total=len(values), desc="label stats"):
            if value == 0:
                continue
            y, x = np.where(cache["labels"] == value)
            length = np.sqrt((np.max(x) - np.min(x)) ** 2 + (np.max(y) - np.min(y)) ** 2)
            cache["label_stats"][value] = (count, length)
        pickle.dump(cache, open(_cache(file, rootdir, outdir), "wb"))
        recompute = True
    # There is no LABELS image

    # CRACK stats
    if recompute or "bounds" not in cache:
        bounds = []
        for l in tqdm(range(1, np.max(cache["labels"])), leave=False, total=np.max(cache["labels"])-1, desc="crack stats"):
            crack, left_top = _crack_bounding_box(cache["labels"] == l)
            skeleton, distance = medial_axis(crack, return_distance=True)

            marked = mark_boundaries(
                np.pad(cache["mask"][left_top[0]: left_top[0] + crack.shape[0],
                       left_top[1]: left_top[1] + crack.shape[1]],
                       ((1, 1), (1, 1))),
                np.pad(crack, ((1, 1), (1, 1))),
                outline_color=(0.5, 0, 0), mode="outer")
            boundary = np.where(marked[:, :, 0] == 0.5)
            counter = 0
            for px in boundary:
                if (0 <= px[0] + left_top[0] - 1 < cache["cimg"].shape[0] and
                        0 <= px[1] + left_top[1] - 1 < cache["cimg"].shape[1] and
                    cache["cimg"][px[0] + left_top[0] - 1, px[1] + left_top[1] - 1, 1] == 255):
                    counter += 1
            bounds.append({
                "label": l,
                "size": np.sum(crack),
                "length": np.sum(skeleton),
                "maxWidth": np.max(distance),
                "avgWidth": np.mean(distance[distance != 0]),
                "boundaryLength": len(boundary[0]),
                "precipitates": counter,
                "farthestPoints": cache["label_stats"][l][1]
            })

        cache["bounds"] = bounds
        pickle.dump(cache, open(_cache(file, rootdir, outdir), "wb"))
        recompute = True
    # CRACK stats CSV file
    if recompute or not os.path.isfile(_csvfile(file, rootdir, outdir)):
        pd.DataFrame(cache["bounds"]).to_csv(open(_csvfile(file, rootdir, outdir), "wt"))

    return cache
