
import numpy as np
import scipy.ndimage as ndi
from cracks.utils import peek
import cv2
from skimage.morphology import label
import math
import shapely

def synth_iqr(stack, crop_mask):
    q05_proj = np.percentile(stack, 5, axis=0) * crop_mask
    
    max_concrete_val = np.percentile(q05_proj[crop_mask == 1], 99.5)
    inverted_proj = np.clip(max_concrete_val - q05_proj, 0, None) * crop_mask
    
    macro_background = ndi.gaussian_filter(inverted_proj, sigma=35)
    flat_proj = np.clip(inverted_proj - macro_background, 0, None) * crop_mask
    
    denoised_proj = ndi.median_filter(flat_proj, 3)
    
    noise_floor = np.percentile(denoised_proj[crop_mask == 1], 40)
    clean_proj = np.clip(denoised_proj - noise_floor, 0, None)
    
    crack_peak = np.percentile(clean_proj[crop_mask == 1], 99.0)
    synthetic_iqr = np.clip(clean_proj / (crack_peak + 1e-8), 0, 1) ** 0.9
    
    peek(synthetic_iqr)
    return synthetic_iqr * crop_mask

def adaptive_closing(stack, threshold=10):
    morph_size = 1
    intensity_threshold = threshold
    
    raw_mask = (np.min(stack, axis=0) >= intensity_threshold).astype(np.uint8)
    closed = np.copy(raw_mask)
    
    while np.unique(label(closed)).size > 2 or morph_size > 100:
        morph_size += 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        closed = cv2.morphologyEx((np.min(stack, axis=0) >= 10).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
    return closed, morph_size, raw_mask

def heal_chipped_mask(binary_mask, inner_scale=0.8):
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros_like(binary_mask), np.zeros_like(binary_mask)

    largest_contour = max(contours, key=cv2.contourArea)
    
    hull = cv2.convexHull(largest_contour)
    polygon = shapely.Polygon(np.squeeze(hull))
    mic = shapely.maximum_inscribed_circle(polygon)
    
    center_x, center_y = mic.coords[0]
    edge_x, edge_y = mic.coords[1]
    
    radius = math.sqrt((center_x - edge_x)**2 + (center_y - edge_y)**2)
    
    healed_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.circle(healed_mask, (int(center_x), int(center_y)), int(radius), 1, thickness=cv2.FILLED)
    
    small_radius = int(radius * inner_scale)
    inner_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.circle(inner_mask, (int(center_x), int(center_y)), small_radius, 1, thickness=cv2.FILLED)
    
    return healed_mask, inner_mask