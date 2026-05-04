---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: crack-features (3.11.13)
    language: python
    name: python3
---

```python
import os
import gc
import cv2
import pywt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from scipy.optimize import minimize, OptimizeResult
from skimage.morphology import label, disk, remove_small_objects, dilation, skeletonize, opening, closing, medial_axis
from skimage.filters import apply_hysteresis_threshold, meijering, frangi, threshold_otsu
from phasepack import phasecong
from skimage.measure import regionprops
from tqdm.auto import tqdm
from skimage.filters.rank import median
import sys

sys.path.append("../..")

from cracks import features 

ROOT_DIR = "../../experiment3_registered"
PAIRS_CSV = os.path.join(ROOT_DIR, "before_after_pairs.csv")
```

```python
def adaptive_closing(stack, threshold=10):
    morph_size = 1
    intensity_threshold = threshold
    
    raw_mask = (np.min(stack, axis=0) >= intensity_threshold).astype(np.uint8)
    closed = np.copy(raw_mask)
    
    while np.unique(label(closed)).size > 2:
        morph_size += 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        closed = cv2.morphologyEx((np.min(stack, axis=0) >= 10).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
    return closed, morph_size, raw_mask

def heal_chipped_mask(binary_mask, inner_scale=0.8):
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    hull = cv2.convexHull(largest_contour)
    
    healed_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.drawContours(healed_mask, [hull], -1, 1, thickness=cv2.FILLED)

    M = cv2.moments(healed_mask.astype(np.uint8))
    
    # healed mask centroid
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    
    area = M["m00"]
    equivalent_radius = np.sqrt(area / np.pi)
    
    small_radius = int(equivalent_radius * inner_scale)
    
    inner_mask = np.zeros_like(healed_mask, dtype=np.uint8)
    cv2.circle(inner_mask, (center_x, center_y), small_radius, 1, thickness=cv2.FILLED)
    
    return healed_mask, inner_mask 

def peek(img, title="Pipeline Step", cmap=None, size=6):
    plt.figure(figsize=(size, size))
    
    if cmap is None:
        if img.dtype == bool or np.unique(img).size <= 2:
            cmap = "viridis"
        else:
            cmap = "gray"
            
    plt.imshow(img, cmap=cmap)
    plt.title(f"DEBUG: {title} {img.shape}")
    plt.axis("off")
    plt.show()

# class CircleFitResult:
#     def __init__(self, x, y, r):
#         self.x = [x, y, r]

# def fit_circle(mask, min_radius=450, margin_reduction=0.0, prior_circle=None):
#     def circle_error(params):
#         center_x, center_y, radius = params
#         y, x = np.where(mask == 1)
#         pos = np.sum((x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2)
#         y, x = np.where(mask == 0)
#         neg = np.sum((x - center_x) ** 2 + (y - center_y) ** 2 <= (radius - 1) ** 2)
#         return pos + neg

#     if prior_circle is None:
#         initial_guess = [mask.shape[1] / 2, mask.shape[0] / 2, min_radius * 1.125]
#         bounds = ((min_radius, mask.shape[1] - min_radius),
#                   (min_radius, mask.shape[0] - min_radius),
#                   (min_radius, min_radius * 1.25))
#     else:
#         px, py, pr = prior_circle.x
#         shift = 10
        
#         initial_guess = [px, py, pr]
#         bounds = ((px - shift, px + shift),
#                   (py - shift, py + shift),
#                   (pr - shift, pr + shift))

#     opt = minimize(circle_error,
#                    x0=initial_guess,
#                    bounds=bounds,
#                    method="COBYLA")
                   
#     return CircleFitResult(opt.x[0], opt.x[1], opt.x[2] * (1 - margin_reduction))


# def hough_circle(mask, min_radius=450, margin_reduction=0.0, prior_circle=None):
#     if prior_circle is not None:
#         return fit_circle(mask, min_radius, margin_reduction, prior_circle)

#     if mask.max() <= 1:
#         mask_uint8 = (mask.astype(np.uint8) * 255)
#     else:
#         mask_uint8 = mask.astype(np.uint8)

#     circles = cv2.HoughCircles(
#         mask_uint8, cv2.HOUGH_GRADIENT, dp=1,
#         minDist=max(mask.shape), param1=50, param2=20,
#         minRadius=min_radius, maxRadius=min(mask.shape) // 2
#     )

#     if circles is not None:
#         best_circle = circles[0, 0]
#         return CircleFitResult(best_circle[0], best_circle[1], best_circle[2] * (1 - margin_reduction))
#     else:
#         return fit_circle(mask, min_radius, margin_reduction)

def prune_skeleton(skel, min_branch_length=200):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    pruned = skel.copy().astype(bool)
    changed = True

    while changed:
        changed = False
        neighbor_count = ndi.convolve(pruned.astype(np.int32), kernel, mode='constant')

        endpoints = list(zip(*np.where(pruned & (neighbor_count == 1))))

        for (r, c) in endpoints:
            if not pruned[r, c]:
                continue

            path = [(r, c)]
            prev_r, prev_c = r, c
            cr, cc = r, c

            while True:
                nc = neighbor_count[cr, cc]
                if nc >= 3:
                    break
                if nc == 0:
                    break

                nbrs = [
                    (cr + dr, cc + dc)
                    for dr in [-1, 0, 1]
                    for dc in [-1, 0, 1]
                    if (dr, dc) != (0, 0)
                    and 0 <= cr + dr < pruned.shape[0]
                    and 0 <= cc + dc < pruned.shape[1]
                    and pruned[cr + dr, cc + dc]
                    and (cr + dr, cc + dc) != (prev_r, prev_c)
                ]

                if len(nbrs) != 1:
                    path.append((cr, cc))
                    break

                prev_r, prev_c = cr, cc
                cr, cc = nbrs[0]
                path.append((cr, cc))

                if len(path) >= min_branch_length:
                    break

            if len(path) < min_branch_length:
                for (pr, pc) in path[:-1]:
                    pruned[pr, pc] = False
                changed = True

    return pruned
```

```python
def convex_filter(anomaly_mask, lower_tresh=50, line_tresh=200, min_physical_length=100):
    labeled = label(anomaly_mask)
    props = regionprops(labeled)
    cracks = np.zeros_like(anomaly_mask)
    bubbles = np.zeros_like(anomaly_mask)
    
    for prop in props:
        if prop.area < lower_tresh:
            continue
            
        solidity = prop.solidity
        eccentricity = prop.eccentricity

        if solidity >= 0.70 and eccentricity < 0.95:
            coords = prop.coords
            bubbles[coords[:, 0], coords[:, 1]] = 1
            continue

        is_long_enough = (prop.axis_major_length > min_physical_length)
        is_crack = (solidity < 0.30) or (eccentricity > 0.95 and solidity < 0.8)
        
        if is_crack and prop.area > line_tresh and is_long_enough:
            coords = prop.coords
            cracks[coords[:, 0], coords[:, 1]] = 1
            
    return cracks, bubbles
```

```python

def synth_iqr(stack, crop_mask):
    q05_proj = np.percentile(stack, 5, axis=0) * crop_mask
    
    max_concrete_val = np.percentile(q05_proj[crop_mask == 1], 99.5)
    inverted_proj = max_concrete_val - q05_proj
    inverted_proj = np.clip(inverted_proj, 0, None) * crop_mask
    
    macro_background = ndi.gaussian_filter(inverted_proj, sigma=35)
    flat_proj = inverted_proj - macro_background
    flat_proj = np.clip(flat_proj, 0, None) * crop_mask
    
    noise_floor = np.percentile(flat_proj[crop_mask == 1], 60)
    clean_proj = np.clip(flat_proj - noise_floor, 0, None)
    
    crack_peak = np.percentile(clean_proj[crop_mask == 1], 99.9)
    synthetic_iqr = np.clip(clean_proj / (crack_peak + 1e-8), 0, 1)
    
    return (synthetic_iqr ** 0.8) * crop_mask
```

```python
def detect_valleys_tophat(proj, crop_mask, max_feature_size=50):
    smoothed_img = opening(proj, disk(3))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_feature_size, max_feature_size))
    black_hat = cv2.morphologyEx(smoothed_img, cv2.MORPH_BLACKHAT, kernel)
    
    active_pixels = black_hat[crop_mask == 1]
    active_pixels = active_pixels[active_pixels > 0] 
    
    threshold = np.percentile(active_pixels, 90)
    
    return (black_hat > threshold) & crop_mask

def detect_hybrid_cracks(crop_mask, proj, bubbles_to_ignore=None):
    if bubbles_to_ignore is not None:
        shield = dilation(bubbles_to_ignore, disk(5))
        proj[shield] = np.median(proj[crop_mask == 1])
        
    smooth_diff_thin = ndi.gaussian_filter(proj, sigma=2.0)
    norm_thin = (smooth_diff_thin - np.min(smooth_diff_thin)) / (np.max(smooth_diff_thin) - np.min(smooth_diff_thin) + 1e-8)
    
    tubeness_meijering = meijering(norm_thin, sigmas=range(2, 10, 1), black_ridges=False) * crop_mask
    
    active_m = tubeness_meijering[crop_mask == 1]
    mask_meijering = apply_hysteresis_threshold(
        tubeness_meijering, 
        np.percentile(active_m, 70), 
        np.percentile(active_m, 95)
    )
    mask_meijering = prune_skeleton(mask_meijering) 

    smooth_std_wide = ndi.gaussian_filter(proj, sigma=5.0)
    norm_wide = (smooth_std_wide - np.min(smooth_std_wide)) / (np.max(smooth_std_wide) - np.min(smooth_std_wide) + 1e-8)
    
    tubeness_frangi = frangi(norm_wide, sigmas=range(6, 14, 1), black_ridges=False, beta=0.8) * crop_mask
    
    active_f = tubeness_frangi[crop_mask == 1]
    mask_frangi = tubeness_frangi > np.percentile(active_f, 85)

    fused_mask = mask_meijering | mask_frangi
    
    closed_cracks = closing(fused_mask, disk(3))
    
    clean_cracks = remove_small_objects(closed_cracks, max_size=100)
    
    return clean_cracks & crop_mask


def detect_anomalies(proj, iqr, crop_mask):
    mask_tophat = detect_valleys_tophat(proj, crop_mask)
    mask_variance = detect_hybrid_cracks(crop_mask, iqr, bubbles_to_ignore=mask_tophat)
    
    combined_anomalies = mask_variance | mask_tophat
    combined_anomalies = combined_anomalies & crop_mask
    
    return remove_small_objects(combined_anomalies.astype(bool), max_size=60)

def refine_cracks_by_skeleton(thick_cracks, min_branch_length=30):
    skel = medial_axis(thick_cracks)
    
    pruned_skel = prune_skeleton(skel, min_branch_length=min_branch_length)

    dilated = dilation(pruned_skel, disk(3))

    return dilated
    # return pruned_skel 

def segment_pipeline_geometric(stack, circle_mask, small_circle):
    y_idx, x_idx = np.where(circle_mask == 1)
        
    ymin, ymax = np.min(y_idx), np.max(y_idx)
    xmin, xmax = np.min(x_idx), np.max(x_idx)
    
    crop_mask = circle_mask[ymin:ymax+1, xmin:xmax+1]
    crop_stack = stack[:, ymin:ymax+1, xmin:xmax+1]
    
    subset_stack = crop_stack[::40, :, :] # avoid sorting the entire stack
    q_low_proj = np.quantile(subset_stack, 0.05, axis=0).astype(np.float32)
    q_high_proj = np.quantile(subset_stack, 0.5, axis=0).astype(np.float32)
    # diff_proj = q_high_proj - q_low_proj
    # diff_proj = np.var(subset_stack, axis=0).astype(np.float32)
    diff_proj = synth_iqr(subset_stack, crop_mask)

    anomalies = detect_anomalies(q_low_proj, diff_proj, crop_mask)
    
    crop_cracks, crop_bubbles = convex_filter(anomalies, lower_tresh=15)

    refined_crop_cracks = refine_cracks_by_skeleton(crop_cracks, min_branch_length=30)
    
    full_cracks = np.zeros_like(circle_mask)
    full_bubbles = np.zeros_like(circle_mask)
    
    full_cracks[ymin:ymax+1, xmin:xmax+1] = refined_crop_cracks 
    full_bubbles[ymin:ymax+1, xmin:xmax+1] = crop_bubbles 

    full_cracks = full_cracks & small_circle
    full_bubbles = full_bubbles & small_circle

    full_diff_proj = np.zeros(circle_mask.shape, dtype=np.float32)
    full_diff_proj[ymin:ymax+1, xmin:xmax+1] = diff_proj 
    
    return full_cracks, full_bubbles, full_diff_proj
```

```python
# circle params returned either as a default parameter (if after) or the computed value (if after)

def process_npy_to_features(npy_path, save_path=None):
    if not isinstance(npy_path, str) or not os.path.exists(npy_path):
        return pd.DataFrame()

    is_exp = "after" in npy_path
    stack = np.load(npy_path)

    mask, _, _ = adaptive_closing(stack)
    peek(mask, "adaptive closing")
    outer, inner = heal_chipped_mask(mask, inner_scale=0.8)

    peek(outer, "outer")
    peek(inner, "inner")
    
    cracks, bubbles, diff_proj = segment_pipeline_geometric(stack, outer, inner)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(diff_proj, cmap="gray")
    ax.imshow(np.where(cracks > 0, 1, np.nan), cmap="Reds", vmin=0, vmax=1, alpha=0.8)
    ax.imshow(np.where(bubbles > 0, 1, np.nan), cmap="Blues", vmin=0, vmax=1, alpha=0.8)
    ax.set_title(f"{os.path.basename(npy_path)} {'After' if is_exp else 'Before'}", fontsize=12)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()

    table_cracks = features.compute(cracks)
    df_cracks = features.global_summary(pd.DataFrame(table_cracks))

    print(f"Image: {npy_path} Stage: {'AFTER' if is_exp else 'before'} \n{df_cracks}")
    
    del stack, mask, outer, inner, cracks, bubbles
    gc.collect()

    return df_cracks
```

```python
pairs_df = pd.read_csv(PAIRS_CSV)
results_dict = {}
col_before = 'Sample before exposure'
col_after = 'Sample after exposure'   
BASE_OUTPUT_DIR = "26_05_04"

for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Comparing pairs"):
    file_before = os.path.join(ROOT_DIR, str(row.get(col_before, '')))
    file_after = os.path.join(ROOT_DIR, str(row.get(col_after, '')))

    if not os.path.exists(file_before) or not os.path.exists(file_after):
        print("Incomplete pair: are both loaded?")
        continue

    pair_dir = os.path.join(BASE_OUTPUT_DIR, "{:02d}".format(idx))
    os.makedirs(pair_dir, exist_ok=True)
    
    name_before = os.path.splitext(os.path.basename(file_before))[0] + ".png"
    name_after = os.path.splitext(os.path.basename(file_after))[0] + ".png"
    
    save_before = os.path.join(pair_dir, name_before)
    save_after = os.path.join(pair_dir, name_after)
    
    df_before  = process_npy_to_features(file_before, save_before)
    df_after = process_npy_to_features(file_after, save_after)
    
    df_before['stage'] = 'before'
    df_before['pair_id'] = idx
        
    df_after['stage'] = 'after'
    df_after['pair_id'] = idx
    
    print("pair loaded")
        
    results_dict[idx] = {
        'before': df_before,
        'after': df_after
    }
```

```python
all_before = [v['before'] for v in results_dict.values() if not v['before'].empty]
all_after = [v['after'] for v in results_dict.values() if not v['after'].empty]

df_before_combined = pd.concat(all_before)
df_after_combined = pd.concat(all_after)

agg_before = df_before_combined.groupby('pair_id').sum(numeric_only=True)
agg_after = df_after_combined.groupby('pair_id').sum(numeric_only=True)

diff_df = agg_after - agg_before

agg_before_prefixed = agg_before.add_prefix('before_')
agg_after_prefixed = agg_after.add_prefix('after_')
diff_df_prefixed = diff_df.add_prefix('diff_')

side_by_side = pd.concat([agg_before_prefixed, agg_after_prefixed, diff_df_prefixed], axis=1)
display(side_by_side)

side_by_side.to_csv("2026_05_04_cracks_hybrid.csv")
```
