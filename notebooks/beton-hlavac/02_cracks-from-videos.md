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
%load_ext autoreload
%autoreload 2
```

Erik připravil data to superpozice. Nyní je potřeba:

- vysegmentovat betonový válec
- identifikovat pixely patřící trhlinám
- napočítat vlastnosti trhlin na válci

```python
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import label
from scipy.optimize import minimize
from matplotlib.patches import Circle
from tqdm.auto import tqdm
from skimage.filters.rank import median
from skimage.morphology import disk, binary_erosion, closing
```

Data bylo nutné rozdělit na jumbo vzorky a malé vzorky. Vyrobil jsem csv, kde jsou označeny.
Níže je filtrace podle třídy vzorku (pracuji jen se smallv1)

```python
ROOT = "../../erik"
metadata = pd.read_csv(os.path.join(ROOT, "metadata.csv"), header=None, names=["file path", "type"])
print(f"Types of samples: {np.unique(metadata['type']).tolist()}")
```

```python
# go thru directory and select only samples "smallv1"
my_type = "smallv1"
stack_files = metadata[metadata["type"] == my_type]["file path"].values.tolist()
print(stack_files)
```

Hypotéza je, že stack obsahuje pouze dvě třídy betonový vzorek (popředí) a pozadí.

Protože na rozhraní budou pixely, které mohou být identifikovány špatně, tak je potřeba provést morfologii finální masky.

Aby morfologie měla co nejméně drastický dopad, tak se velikost kruhového kernelu postupně zvětšuje, dokud nevznikne pouze jeden segment (popředí) ve finální masce.

```python
def adaptive_closing(stack, threshold=10):
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
        closed = cv2.morphologyEx((np.min(stack, axis=0) >= 10).astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return closed, morph_size, raw_mask
```

Hypotéza: vzorek má kruhový tvar.

Aplikace: fitneme kruh na masku popředí. Penalizuji počtem pixelů, které mají být uvnitř masky a nejsou a počtem pixelů, které jsou vně kruhu a nemají být.

```python
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


```

```python
import pywt
import numpy as np

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

import scipy.ndimage as ndi

def prune_skeleton(skel, num_iter=12):
    pruned = skel.copy()
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    for _ in range(num_iter):
        neighbor_count = ndi.convolve(pruned.astype(int), kernel, mode='constant')
        endpoints = pruned & (neighbor_count == 1)
        pruned = pruned & ~endpoints
        
    return pruned
```

```python
import cv2
import numpy as np
from skimage.filters import frangi, apply_hysteresis_threshold
from skimage.morphology import disk, remove_small_objects, dilation, skeletonize, opening
from phasepack import phasecong

def detect_valleys_tophat(min_proj, crop_mask, max_feature_size=25):
    smoothed_min = opening(min_proj, disk(3))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_feature_size, max_feature_size))
    bottom_hat = cv2.morphologyEx(smoothed_min, cv2.MORPH_BLACKHAT, kernel)
    
    active_pixels = bottom_hat[crop_mask == 1]
    threshold = np.percentile(active_pixels[active_pixels > 0], 98)
    return (bottom_hat > threshold) & crop_mask

def detect_structural_variance(crop_stack, crop_mask, bubbles_to_ignore=None):
    std_proj = np.std(crop_stack, axis=0)
    
    if bubbles_to_ignore is not None:
        shield = dilation(bubbles_to_ignore, disk(3))
        background_median = np.median(std_proj[crop_mask == 1])
        std_proj[shield] = background_median
        
    norm_std = (std_proj - np.min(std_proj)) / (np.max(std_proj) - np.min(std_proj) + 1e-8)
    
    vesselness = frangi(norm_std, sigmas=(1, 2, 3, 4), black_ridges=False, beta=0.10)
    vesselness_masked = vesselness * crop_mask
    
    if bubbles_to_ignore is not None:
        lead_wall = dilation(bubbles_to_ignore, disk(3))
        vesselness_masked[lead_wall] = 0.0
    
    active_pixels = vesselness_masked[crop_mask == 1]
        
    high_thresh = np.percentile(active_pixels, 97) 
    low_thresh = np.percentile(active_pixels, 88)  
    
    hyst_mask = apply_hysteresis_threshold(phase_map, low_thresh, high_thresh)
    
    thinned_cracks = skeletonize(hyst_mask)
    thinned_cracks = prune_skeleton(thinned_cracks, num_iter=8) 
    flow_restricted_cracks = dilation(thinned_cracks, disk(1))

    return flow_restricted_cracks & crop_mask



def detect_structural_phase(crop_stack, crop_mask, bubbles_to_ignore=None):
    std_proj = denoise_swt(np.std(crop_stack, axis=0))
    
    if bubbles_to_ignore is not None:
        shield = dilation(bubbles_to_ignore, disk(5))
        background_median = np.median(std_proj[crop_mask == 1])
        std_proj[shield] = background_median
        
    PC = phasecong(std_proj, nscale=6, norient=8, minWaveLength=3, mult=1.8, sigmaOnf=0.55, k=1.2, )
    phase_map = PC[0] * crop_mask
    
    active_pixels = phase_map[crop_mask == 1]

    high_thresh = np.percentile(active_pixels, 98) 
    low_thresh = np.percentile(active_pixels, 80)  
    
    # remove small seeds around the cracks
    raw_seeds = phase_map > high_thresh
    clean_seeds = remove_small_objects(raw_seeds, min_size=50)
    deleted_noise = raw_seeds & ~clean_seeds
    phase_map[deleted_noise] = 0.0
    
    hyst_mask = apply_hysteresis_threshold(phase_map, low_thresh, high_thresh)
    thinned_cracks = skeletonize(hyst_mask)
    
    flow_restricted_cracks = dilation(thinned_cracks, disk(1))

    return hyst_mask & crop_mask

def detect_anomalies(crop_stack, min_proj, crop_mask):
    mask_tophat = detect_valleys_tophat(min_proj, crop_mask)
    # mask_variance = detect_structural_variance(crop_stack, crop_mask, bubbles_to_ignore=mask_tophat)
    mask_variance = detect_structural_phase(crop_stack, crop_mask, bubbles_to_ignore=mask_tophat)
    

    combined_anomalies = mask_variance | mask_tophat
    combined_anomalies = combined_anomalies & crop_mask
    
    return remove_small_objects(combined_anomalies, min_size=100)
```

```python
from skimage.measure import label, regionprops
import numpy as np

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
```

```python
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
rows = np.ceil(len(stack_files) / 2).astype(int)
layers = []
plt.figure(figsize=(15, rows * 5))

stack_files = stack_files[:5]

for i, stack_file in tqdm(enumerate(stack_files), total=len(stack_files), desc="computing masks"):
    try:
        stack = np.load(stack_file)
        mask, kernel_size, raw_mask = adaptive_closing(stack, 3)
        circle = fit_circle(mask, margin_reduction=0.05) 
        
        y_grid, x_grid = np.ogrid[:mask.shape[0], :mask.shape[1]]
        circle_mask = (((x_grid - circle.x[0]) ** 2 + (y_grid - circle.x[1]) ** 2) <= circle.x[2] ** 2).astype(np.uint8)

        cracks, bubbles = segment_pipeline_geometric(stack, circle_mask)
        layers.append([np.min(stack, axis=0), circle_mask, cracks, bubbles])
        
        ax = plt.subplot(rows, 2, i + 1)
        ax.imshow(stack[100], cmap="gray")
        ax.add_patch(Circle((circle.x[0], circle.x[1]), circle.x[2], edgecolor='white', facecolor='none', lw=1, alpha=0.5))
        ax.imshow(np.where(cracks > 0, 1, np.nan), cmap="Reds", vmin=0, vmax=1, alpha=0.8)
        ax.imshow(np.where(bubbles > 0, 1, np.nan), cmap="Blues", vmin=0, vmax=1, alpha=0.8)
        ax.set_title(f"{os.path.basename(stack_file)}: [{circle.x[0]:.0f},{circle.x[1]:.0f}] r={circle.x[2]:.0f}px", fontsize=10)
        ax.axis('off')
        
    except Exception as e:
        print(f"{os.path.basename(stack_file)}: processing failed with {e}")

plt.tight_layout()
plt.show()
```

**Batch run:** adaptive closing pro oprahované snímky a fitnutí kruhu na morfologicky upravenou masku.


# Signifikantní rozdíl v intenzitě

Vysegmentujeme pixely, kde rozdíl v intenzitě je zásadní. Tyto pixely tvoří segmenty:
- Kulaté (a malé) segmenty jsou póry.
- Čárové segmenty jsou trhliny.

Je nutné zajistit spojitost/nejspojitost segmentů, nebo jinak řešit co je segment...

Níže je naivní řešení bez dělení segmentů.

```python
max = np.max(stack, axis=0)
min = np.min(stack, axis=0)
# Tady je otázka čím by se to mělo rozmazávat ... velikost disku je poměrně zásadní pro finální výsledek
# Velikost zřejmě souvisí s velikostí objektů, které se mají ve výsledku detekovat, tj. bude nutné ji nastavit podle typu vzorků
# Nabízí se otázka jak tento parametr určit
max_med = median(max, footprint=disk(51), mask=layers[-1][2])
min_med = median(min, footprint=disk(51), mask=layers[-1][2])

minmax_diff_norm = max.astype(float) - max_med.astype(float) - (min.astype(float) - min_med.astype(float))
```

```python
plt.figure(figsize=(15, 13))

ax = plt.subplot(2, 1, 1)
ax.hist(minmax_diff_norm[layers[-1][2].astype(bool)].reshape(-1), bins=200)
ax.set_yscale("log")
ax.axvline(-2, color="red")
ax.set_title("Histogram of differences between max and min values (normalized)")

ax = plt.subplot(2, 1, 2)
ax.imshow(np.logical_or(
    minmax_diff_norm * layers[-1][2] > 15,
    minmax_diff_norm * layers[-1][2] < -15
), cmap="Reds")
ax.imshow(minmax_diff_norm * layers[-1][2], cmap="gray", alpha=0.3)
ax.set_xlim(80,1180)
ax.set_title("Threshold of differences between max and min values (normalized)")
plt.show()
```

```python
cracks_and_holes = label(np.logical_or(
    minmax_diff_norm * layers[-1][2] > 15,
    minmax_diff_norm * layers[-1][2] < -15
))
```

Measure circularity of each segment and split them according to a threshold ...

Hypothesis: Circular are pores, non-circular cracks and missing parts of the sample.

There is definitely a lot of errors (fused segments, noise around the threshold) but rough idea should be valid

```python
CIRCULARITY_THRESHOLD = 1.7
```

```python
segments = []
for l in tqdm(np.arange(1, np.max(cracks_and_holes)), total=np.max(cracks_and_holes) - 1, desc="cracks and holes features"):
    x, y = np.where(cracks_and_holes == l)
    segments.append([l, np.max(x), np.min(x), np.max(y), np.min(y), len(x)])
```

```python
circularity = [(segment_id, (((maxx - minx + 1) + (maxy - miny + 1)) / 4) ** 2 * np.pi / pixel_count)
               for segment_id, maxx, minx, maxy, miny, pixel_count in segments]
```

```python
border = binary_erosion(layers[-1][2], disk(5))
broken_edge_segments = set(np.unique(cracks_and_holes * (layers[-1][2] - border)).tolist())
```

```python
circular = set([seg_id for seg_id, ratio in circularity if ratio < CIRCULARITY_THRESHOLD]) - broken_edge_segments
non_circular = set([seg_id for seg_id, ratio in circularity if ratio >= CIRCULARITY_THRESHOLD]) - broken_edge_segments
```

```python
plt.imshow(cracks_and_holes * (layers[-1][2] - border), cmap="gray")
plt.show()
```

```python
plt.figure(figsize=(10, 10))
plt.imshow(np.isin(cracks_and_holes, list(non_circular)), cmap="Reds")
plt.imshow(np.isin(cracks_and_holes, list(circular)), cmap="Greens", alpha=0.5)
plt.imshow(np.isin(cracks_and_holes, list(broken_edge_segments)[1:]), cmap="Blues", alpha=0.5)
plt.xlim(80,1180)
plt.show()
```

Pozorování:
- trhliny do kterých nesvítí vůbec jsou černé (minimální hodnota)
- na okrajích vzorku je problém, nejspíš kvůli mediánovému filtru, který zde funguje asi dost omezeně, nebo kvůli náběhové hraně vzorku (vysoká hodnota, velká plocha na okraji masky)


## Úloha do 31.8.

- zpracovat crack_and_holes pro všechny zregistrovaná videa
- vytvořit tabulku, které zpracuje masky
- volitelně odštípnutý okraj definovat jako crack neznámé tloušťky (odpadlý okraj nezapočítávat)

```python
cracks_A = np.isin(cracks_and_holes, list(non_circular))
pores = np.isin(cracks_and_holes, list(circular))
boundary = np.isin(cracks_and_holes, list(broken_edge_segments)[1:])
inner = closing(circle_mask - np.isin(cracks_and_holes, list(broken_edge_segments)[1:]), disk(8))
cracks_B = np.logical_and(boundary, inner)
cracks = np.logical_or(cracks_A, cracks_B)
boundary = np.logical_and(boundary, np.logical_not(inner))
# => cracks, pores, boundary
```

```python
plt.figure(figsize=(10, 10))
plt.imshow(cracks, cmap="Reds")
plt.imshow(pores, cmap="Greens", alpha=0.5)
plt.imshow(boundary, cmap="gray", alpha=0.5)
plt.xlim(80,1180)
plt.show()
```

```python
from cracks import features
import pandas as pd

table_cracks = features.compute(cracks)
df_cracks = pd.DataFrame(table_cracks).sort_values(by="crackSize_px", ascending=False)
df_cracks.to_csv("swt_phase.csv")
```

a# Úhel dopadu světla

**Hypotéza 3:** pokud není pixel součástí povrchu, který je paralelní ke snímači, bude mít jeho intenzita v rámci stacku velký rozptyl (nebo rozdíl max-min, ...). Tímto způsobem lze identifikovat plošky, které mají velkou odchylku od rovnoběžné plochy.

**Hypotéza 4:** Plocha, která je natočená vůči rovině snímače bude mít pro jeden konkrétní úhel maximální intenzitu, pro úhel opačný minimální. Tím lze zjistit úhel této plochy.

```python
direction = (np.argmax(stack, axis=0) / len(stack) * 256).astype(np.uint8) * layers[-1][2]
dir_med = median(direction, disk(11), mask=layers[-1][2])
angle =  np.logical_and(np.abs(direction.astype(float) - dir_med.astype(float)) < 180, np.abs(direction.astype(float) - dir_med.astype(float)) > 60)
```

```python
plt.figure(figsize=(15, 9))
plt.imshow(angle, cmap="gray")
plt.show()
```

Morfologie může mít problém v případě velkých děr (ale asi ne v případě velkých trhlin). Možná je výhodou, když díry budou mimo masku a nebudou se počítat do trhlin.

Snímek je dobré normalizovat
- oříznout podle masky
- mediánovým filtrem (řeší různé plošky na vzorku)

Úhel největší reflexivity má smysl řešit pouze pro pixely, kde je největší reflexivita signifikantně větší (tj. je tam nějaký reliéf).

Bude nutné řešit kontext okolních pixelů (například floodfill s nějakou zajímavou podmínkou (v rámci segmentu se úhel mění o nějakou větší hodnotu => pór))


Dle Zbyni je potřeba rozlišovat:
- póry (nemají vliv na pevnost a jsou součástí betonu
- trhliny (to je to co chceme měřit)
- uštípnuté okraje (nezajímavé pro vyhodnocení)

```python
var = np.var(stack, axis=0)
```

```python
max = np.max(stack, axis=0)
min = np.min(stack, axis=0)
max_med = median(max, footprint=disk(51), mask=layers[-1][2])
min_med = median(min, footprint=disk(51), mask=layers[-1][2])
```

```python
plt.figure(figsize=(15, 10))
#plt.hist((max.astype(float) - medfilt.astype(float)).reshape(-1, 1), bins=100)
#plt.imshow(np.logical_and(np.abs(max.astype(float) - medfilt.astype(float)) > 16, layers[-1][2]), cmap="gray")
plt.imshow((max.astype(float) - max_med.astype(float) - (min.astype(float) - min_med.astype(float))) < 10, cmap="gray")
#plt.imshow(layers[-1][2], cmap="gray")
#plt.yscale("log")
plt.show()
```

```python
plt.figure(figsize=(15, 10))
plt.imshow(np.min(stack, axis=0) < 20, cmap="gray")
plt.show()
```

```python
plt.figure(figsize=(15,10))
plt.imshow(stack[0], cmap="gray")
plt.xlim(800, 1000)
plt.ylim(200, 400)
plt.axhline(302, color="red")
plt.axvline(950, color="red")

plt.axhline(290, color="orange")
plt.axvline(975, color="orange")

plt.axhline(275, color="green")
plt.axvline(922, color="green")

plt.axhline(270, color="blue")
plt.axvline(900, color="blue")
plt.show()
```

<!-- #region -->
Nerovnosti:

Nerovnosti detekujeme pomocí prahování variability (max-min) nebo var.


- trhliny - netriviální délka (segmentace podle velikosti?)
- špína - totéž co výstupky
- výstupky - světlé tečky, stín je velmi tenký, není vidět n a var, ale je vidět na max-min
- bubliny - má uprostřed tmavou skvrnu
<!-- #endregion -->

```python
points = [
    [370, 980, "red", "hole boundary"],
    [290, 975, "red", "hole boundary"],
    [302, 950, "orange", "thick crack"],
    [275, 922, "green", "thin crack"],
    [270, 900, "blue", "flat surface"]
]
```

```python
plt.figure(figsize=(15, 10))
ax = plt.subplot(121)
ax.imshow(np.max(stack, axis=0) - np.min(stack, axis=0))
ax.set_ylim(200, 400)
ax.set_xlim(800, 1000)
ax = plt.subplot(122)
ax.imshow(var)
ax.set_ylim(200, 400)
ax.set_xlim(800, 1000)
plt.scatter([x for _, x, _, _ in points], [y for y, _, _, _ in points], marker="o", c=[color for _, _, color, _ in points], s=70)
plt.show()
```

```python
for y, x, color,label in points:
    plt.plot(stack[:, y, x], color=color, label=label)
plt.legend()
plt.show()
```

```python
from scipy.ndimage import gaussian_filter1d
```

```python
for y, x, color, label in points:
    approx = gaussian_filter1d(stack[:, y, x].astype(np.float32), 5)
    plt.plot(gaussian_filter1d(np.abs(stack[:, y, x].astype(np.float32) - approx), 11), color=color, label=label)
plt.legend()
plt.show()
```

```python
from sklearn.cluster import KMeans
```

```python
kmeans = KMeans(3).fit(stack[:, 200:900, 800:1500].reshape(stack.shape[0], -1).T)
```

```python
stack_labels = kmeans.labels_.reshape(700, 700)
```

```python
plt.imshow(stack_labels)
```

```python
sample = stack[:, 200:900, 800:1500].reshape(stack.shape[0], -1).T
```

```python
sample.shape
```

```python
plt.figure(figsize=(15,5))
plt.plot(np.argmax(gaussian_filter1d(sample[:700], 15, axis=1), axis=1))
plt.plot(np.argmin(gaussian_filter1d(sample[:700], 15, axis=1), axis=1))
plt.show()
```

```python
plt.imshow(np.argmax(gaussian_filter1d(sample, 15, axis=1), axis=1).reshape(700, 700))
```

```python
plt.figure(figsize=(10,10))
plt.imshow(stack[0], cmap="gray")
plt.xlim(80,1180)
plt.show()
```

```python

```
