import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import disk, remove_small_objects, dilation , opening, closing, medial_axis, label 
from skimage.measure import regionprops
from skimage.filters import apply_hysteresis_threshold, meijering, frangi 
from cracks.projection import synth_iqr
from cracks.utils import peek

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

def detect_valleys_tophat(proj, crop_mask, max_feature_size=50):
    smoothed_img = opening(proj, disk(3))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_feature_size, max_feature_size))
    black_hat = cv2.morphologyEx(smoothed_img, cv2.MORPH_BLACKHAT, kernel)
    
    active_pixels = black_hat[crop_mask == 1]
    active_pixels = active_pixels[active_pixels > 0] 
    
    threshold = np.percentile(active_pixels, 97)
    
    return (black_hat > threshold) & crop_mask

def detect_hybrid_cracks(crop_mask, proj, bubbles_to_ignore=None):
    if bubbles_to_ignore is not None:
        shield = dilation(bubbles_to_ignore, disk(5))
        proj[shield] = np.median(proj[crop_mask == 1])
        
    smooth_diff_thin = ndi.gaussian_filter(proj, sigma=1.0)
    norm_thin = (smooth_diff_thin - np.min(smooth_diff_thin)) / (np.max(smooth_diff_thin) - np.min(smooth_diff_thin) + 1e-8)
    
    tubeness_meijering = meijering(norm_thin, sigmas=range(1, 10, 1), black_ridges=False) * crop_mask
    
    active_m = tubeness_meijering[crop_mask == 1]
    mask_meijering = apply_hysteresis_threshold(
        tubeness_meijering, 
        np.percentile(active_m, 80), 
        np.percentile(active_m, 90)
    )
    mask_meijering = prune_skeleton(mask_meijering, min_branch_length=25) 

    smooth_std_wide = ndi.gaussian_filter(proj, sigma=2.0)
    norm_wide = (smooth_std_wide - np.min(smooth_std_wide)) / (np.max(smooth_std_wide) - np.min(smooth_std_wide) + 1e-8)
    
    tubeness_frangi = frangi(norm_wide, sigmas=range(6, 14, 1), black_ridges=False, beta=0.8) * crop_mask
    
    active_f = tubeness_frangi[crop_mask == 1]
    mask_frangi = tubeness_frangi > np.percentile(active_f, 85)

    fused_mask = mask_meijering | mask_frangi
    
    closed_cracks = closing(fused_mask, disk(3))
    
    clean_cracks = remove_small_objects(closed_cracks, max_size=80)
    
    return clean_cracks & crop_mask

def detect_anomalies(proj, iqr, crop_mask):
    mask_tophat = detect_valleys_tophat(proj, crop_mask)
    mask_variance = detect_hybrid_cracks(crop_mask, iqr, bubbles_to_ignore=mask_tophat)
    
    combined_anomalies = mask_variance | mask_tophat  
    combined_anomalies = combined_anomalies & crop_mask
    
    return remove_small_objects(combined_anomalies.astype(bool), max_size=50)

def refine_cracks_by_skeleton(thick_cracks, min_branch_length):
    skel = medial_axis(thick_cracks)
    
    pruned_skel = prune_skeleton(skel, min_branch_length=min_branch_length)

    dilated = dilation(pruned_skel, disk(3))

    return dilated

def pipeline_geometric(stack, circle_mask, small_circle, GLOBAL_IQR_SCALE=99.7):
    y_idx, x_idx = np.where(circle_mask == 1)

    if len(y_idx) == 0:
        return np.zeros_like(circle_mask), np.zeros_like(circle_mask), np.zeros_like(circle_mask)
        
    ymin, ymax = np.min(y_idx), np.max(y_idx)
    xmin, xmax = np.min(x_idx), np.max(x_idx)
    
    crop_mask = circle_mask[ymin:ymax+1, xmin:xmax+1]
    crop_stack = stack[:, ymin:ymax+1, xmin:xmax+1]
    
    subset_stack = crop_stack[::40, :, :]
    q_low_proj = np.quantile(subset_stack, 0.10, axis=0).astype(np.float32)
    raw_diff_proj = synth_iqr(subset_stack, crop_mask)
    diff_proj = (raw_diff_proj / GLOBAL_IQR_SCALE) * 255.0
    diff_proj = np.clip(diff_proj, 0, 255).astype(np.float32)

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