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
from scipy.optimize import minimize
from skimage.morphology import label, disk, remove_small_objects, dilation, skeletonize, opening, closing
from skimage.filters import apply_hysteresis_threshold
from phasepack import phasecong
from skimage.measure import regionprops
from tqdm.auto import tqdm

from cracks import features 

ROOT_DIR = "../../experiment3_registered"
PAIRS_CSV = os.path.join(ROOT_DIR, "before_after_pairs.csv")
```

```python
def adaptive_closing(stack, threshold=10):
    morph_size = 1
    intensity_threshold = threshold
    raw_mask = np.min(stack, axis=0) < intensity_threshold
    closed = np.copy(raw_mask)
    while np.unique(label(1 - closed)).size > 2:
        morph_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
        closed = cv2.morphologyEx((np.min(stack, axis=0) >= 10).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed, morph_size, raw_mask

def fit_circle(mask, min_radius=500, margin_reduction=0.0):
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
    opt.x[2] = opt.x[2] * (1 - margin_reduction)
    return opt

def denoise_swt(image, level=2, wavelet='haar'):
    pad_y = (2**level) - (image.shape[0] % (2**level))
    pad_x = (2**level) - (image.shape[1] % (2**level))
    padded_img = np.pad(image, ((0, pad_y), (0, pad_x)), mode='reflect')
    
    coeffs = pywt.swt2(padded_img, wavelet, level=level)
    modified_coeffs = list(coeffs)
    
    cA1, (cH1, cV1, cD1) = modified_coeffs[0]
    modified_coeffs[0] = (cA1, (np.zeros_like(cH1), np.zeros_like(cV1), np.zeros_like(cD1)))
    
    if level >= 2:
        cA2, (cH2, cV2, cD2) = modified_coeffs[1]
        modified_coeffs[1] = (cA2, (np.zeros_like(cH2), np.zeros_like(cV2), np.zeros_like(cD2)))
    
    clean_padded = pywt.iswt2(modified_coeffs, wavelet)
    return clean_padded[:image.shape[0], :image.shape[1]]

def prune_skeleton(skel, num_iter=12):
    pruned = skel.copy()
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    for _ in range(num_iter):
        neighbor_count = ndi.convolve(pruned.astype(int), kernel, mode='constant')
        endpoints = pruned & (neighbor_count == 1)
        pruned = pruned & ~endpoints
    return pruned
```

```python
def detect_valleys_tophat(min_proj, crop_mask, max_feature_size=25):
    smoothed_min = opening(min_proj, disk(3))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_feature_size, max_feature_size))
    bottom_hat = cv2.morphologyEx(smoothed_min, cv2.MORPH_BLACKHAT, kernel)
    
    active_pixels = bottom_hat[crop_mask == 1]
    active_pixels = active_pixels[active_pixels > 0]
    
    threshold = np.percentile(active_pixels, 95)
    return (bottom_hat > threshold) & crop_mask

def detect_structural_phase(crop_stack, crop_mask, bubbles_to_ignore=None):
    std_proj = denoise_swt(np.std(crop_stack, axis=0))
    
    if bubbles_to_ignore is not None:
        shield = dilation(bubbles_to_ignore, disk(5))
        background_median = np.median(std_proj[crop_mask == 1])
        std_proj[shield] = background_median
        
    PC = phasecong(std_proj, nscale=7, norient=8, minWaveLength=3, mult=1.8, sigmaOnf=0.55, k=1.5)
    phase_map = PC[0] * crop_mask
    
    active_pixels = phase_map[crop_mask == 1]

    high_thresh = np.percentile(active_pixels, 96) 
    low_thresh = np.percentile(active_pixels, 90)  
    
    raw_seeds = phase_map > high_thresh
    clean_seeds = remove_small_objects(raw_seeds, min_size=50)
    deleted_noise = raw_seeds & ~clean_seeds
    
    noise_footprint = dilation(deleted_noise, disk(3))
    phase_map[noise_footprint] = 0.0
    
    hyst_mask = apply_hysteresis_threshold(phase_map, low_thresh, high_thresh)
    
    hyst_mask = remove_small_objects(hyst_mask, min_size=45)
    
    thinned_cracks = skeletonize(hyst_mask)
    thinned_cracks = prune_skeleton(thinned_cracks, num_iter=20) 
    
    flow_restricted_cracks = dilation(thinned_cracks, disk(5))

    return flow_restricted_cracks & crop_mask

def detect_anomalies(crop_stack, min_proj, crop_mask):
    mask_tophat = detect_valleys_tophat(min_proj, crop_mask)
    mask_variance = detect_structural_phase(crop_stack, crop_mask, bubbles_to_ignore=mask_tophat)
    
    combined_anomalies = mask_variance | mask_tophat
    combined_anomalies = combined_anomalies & crop_mask
    
    return remove_small_objects(combined_anomalies, min_size=60)

def convex_filter(anomaly_mask, lower_tresh=50, line_tresh=150, min_physical_length=100):
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

        is_long_enough = (prop.major_axis_length > min_physical_length)
        is_crack = (solidity < 0.30) or (eccentricity > 0.95 and solidity < 0.8)
        
        if is_crack and prop.area > line_tresh and is_long_enough:
            coords = prop.coords
            cracks[coords[:, 0], coords[:, 1]] = 1
            
    return cracks, bubbles

def segment_pipeline_geometric(stack, circle_mask):
    y_idx, x_idx = np.where(circle_mask == 1)
    if len(y_idx) == 0:
        return np.zeros_like(circle_mask), np.zeros_like(circle_mask)
        
    ymin, ymax = np.min(y_idx), np.max(y_idx)
    xmin, xmax = np.min(x_idx), np.max(x_idx)
    
    crop_mask = circle_mask[ymin:ymax+1, xmin:xmax+1]
    crop_stack = stack[:, ymin:ymax+1, xmin:xmax+1]
    min_proj = np.min(crop_stack, axis=0)
    
    anomalies = detect_anomalies(crop_stack, min_proj, crop_mask)
    crop_cracks, crop_bubbles = convex_filter(anomalies, lower_tresh=15)
    
    full_cracks = np.zeros_like(circle_mask)
    full_bubbles = np.zeros_like(circle_mask)
    full_cracks[ymin:ymax+1, xmin:xmax+1] = crop_cracks
    full_bubbles[ymin:ymax+1, xmin:xmax+1] = crop_bubbles
    
    return full_cracks, full_bubbles
```

```python
def process_npy_to_features(npy_path):
    if not isinstance(npy_path, str) or not os.path.exists(npy_path):
        return pd.DataFrame()

    stack = np.load(npy_path)
    
    mask, _, _ = adaptive_closing(stack, 3)
    circle = fit_circle(mask, margin_reduction=0.05) 
    
    y_grid, x_grid = np.ogrid[:mask.shape[0], :mask.shape[1]]
    circle_mask = (((x_grid - circle.x[0]) ** 2 + (y_grid - circle.x[1]) ** 2) <= circle.x[2] ** 2).astype(np.uint8)

    cracks, bubbles = segment_pipeline_geometric(stack, circle_mask)
    
    table_cracks = features.compute(cracks)
    df_cracks = pd.DataFrame(table_cracks)
    
    del stack, mask, circle_mask, cracks, bubbles
    gc.collect()
    
    return df_cracks
```

```python
pairs_df = pd.read_csv(PAIRS_CSV)

# Storing pairs of dataframes
results_dict = {}

# Assuming columns are 'before_expo_path' and 'after_expo_path'. Adjust to match your CSV.
col_before = 'Sample before exposure'
col_after = 'Sample after exposure'   

for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Processing Pair Pipelines"):
    file_before = os.path.join(ROOT_DIR, str(row.get(col_before, '')))
    file_after = os.path.join(ROOT_DIR, str(row.get(col_after, '')))
    
    df_before = process_npy_to_features(file_before)
    if not df_before.empty:
        df_before['stage'] = 'before'
        df_before['pair_id'] = idx
        
    df_after = process_npy_to_features(file_after)
    if not df_after.empty:
        df_after['stage'] = 'after'
        df_after['pair_id'] = idx
        
    results_dict[idx] = {
        'before': df_before,
        'after': df_after
    }
```

```python
all_before = pd.concat([v['before'] for v in results_dict.values() if not v['before'].empty], ignore_index=True)
all_after = pd.concat([v['after'] for v in results_dict.values() if not v['after'].empty], ignore_index=True)

# Side-by-side view grouped by pair index
# This merges aggregate metrics for a quick diff per pair
if not all_before.empty and not all_after.empty:
    agg_before = all_before.groupby('pair_id').sum(numeric_only=True).add_prefix('before_')
    agg_after = all_after.groupby('pair_id').sum(numeric_only=True).add_prefix('after_')

    side_by_side = pd.concat([agg_before, agg_after], axis=1)
    display(side_by_side)
else:
    print("No features extracted. Check paths and segmentation outputs.")
```
