---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import cv2
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.segmentation import flood, flood_fill, mark_boundaries
from skimage.morphology import medial_axis, skeletonize
```

```python
rootdir = "/Users/gimli/cvr/data/OneDrive - UJV/analýza obrazu/Ozářené_kamenivo/F_serie/detekce_trhlin_z_BSE_snímků/"
```

```python
tifs = []
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        if file[-4:] == ".tif" or file[-5:] == ".tiff":
            tifs.append(os.path.join(root, file))
            #print(root[len(rootdir):], file)
```

```python
tif = imageio.imread(tifs[0])
```

```python
tif.shape
```

```python
plt.figure(figsize=(15,15))
plt.imshow(tif)
plt.show()
```

```python
np.argmax(np.histogram(tif[:,:,0].reshape(-1),bins=256)[0])
```

# Preliminary thresholding

```python
colored = np.copy(tif[:,:,:])
colored[tif[:,:,0]<=15] = np.array([0,0,0])
colored[np.logical_and(tif[:,:,0]>15, tif[:,:,0]<80)] = np.array([255,50,50])
colored[np.logical_and(tif[:,:,0]>80, tif[:,:,0]<140)] = np.array([200,200,200])
colored[np.logical_and(tif[:,:,0]>140, tif[:,:,0]<230)] = np.array([0,255,50])
#colored[np.logical_and(tif[:,:,0]>=80, tif[:,:,0]<135)] = np.array([100,100,100])
colored[tif[:,:,0]>=230] = np.array([255,255,255])
```

```python
plt.figure(figsize=(15,15))
plt.imshow(colored)
plt.show()
```

# Crack seeds

The darkest values in the image

```python
seeds = np.where(tif[:,:,0]<=15)
```

Now redefine map of crack tracking. Not only the darkest pixels are cracks, but they will be found from seeds.

```python
presegment = np.zeros((tif.shape[0], tif.shape[1]))
```

We take into account lighter pixels, floodfill will be started from seeds (the darkest).

```python
presegment[tif[:,:,0]<80] = 1
```

```python
for seed in zip(seeds[0], seeds[1]):
    if presegment[seed[0], seed[1]] == 1:
        presegment = flood_fill(presegment, seed, 2)
```

As a map of cracks we now create binary image

```python
segmented = np.copy(presegment)
segmented[segmented == 1] = 0
segmented[segmented == 2] = 1
```

```python
plt.figure(figsize=(15,15))
plt.imshow(segmented, cmap="gray")
plt.title("mask of cracks found")
plt.show()
```

```python
skeleton, distance = medial_axis(segmented, return_distance=True)
```

```python
plt.figure(figsize=(15,15))
plt.imshow(distance, cmap="jet")
plt.show()
```

```python
plt.hist(distance.reshape(-1))
plt.title("Histogram of crack thickness (for each pixel, this is not maxima)")
plt.yscale("log")
plt.show()
```

```python
labeled = label(segmented)
```

```python
print(f"Number of cracks {np.max(labeled)}")
```

```python
plt.hist(np.unique(labeled, return_counts=True)[1][1:])
plt.title(f"Crack size histogram")
plt.show()
```

```python
print("For correct length measurement we need to know what length means. Here number of pixels of a skeleton")
plt.hist(np.unique(label(skeleton), return_counts=True)[1][1:])
plt.title(f"Crack length (approx) histogram")
plt.show()
```

```python
bounds = []
for l in range(1, np.max(labeled)):
    print(l)
    crack = np.zeros(segmented.shape, dtype=int)
    crack[labeled == l] = 1    
    skeleton, distance = medial_axis(crack, return_distance=True)
    
    marked = mark_boundaries(segmented, crack, outline_color=(0.5,0,0),mode="outer")
    boundary = np.where(marked[:,:,0] == 0.5)
    counter = 0
    for px in boundary:
        if colored[px[0],px[1],1] == 255:
            counter += 1
    bounds.append({
        "label": l,
        "size": np.sum(crack),
        "length": np.sum(skeleton),
        "maxWidth": np.max(distance),
        "avgWidth": np.mean(distance[distance != 0]),
        "boundaryLength": len(boundary[0]),
        "precipitates": counter
    })
```

```python
boundary
```

```python
plt.imshow(colored[2020:2060, 410:460, :])
plt.show()
```

```python
plt.figure(figsize=(15,15))
plt.imshow(crack[2020:2060, 410:460])
plt.show()
```

```python
import pandas as pd
```

```python
pd.DataFrame(bounds)
```

```python
plt.figure(figsize=(15,15))
plt.imshow(medial_axis(segmented), cmap="gray")
plt.show()

```

```python

```
