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
from tqdm import tqdm
import pandas as pd
import pickle
```

```python
rootdir = "/Users/gimli/cvr/data/microscopy/all-data"
```

```python
OUT_DIR = "/Users/gimli/cvr/data/microscopy/F"
os.makedirs(OUT_DIR, exist_ok=True)
```

```python
tifs = []
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        if file[-4:] == ".tif" or file[-5:] == ".tiff":
            tifs.append(os.path.join(root, file))
```

Zpracování všech snímků

```python
len(tifs)
```

```python
cimgs = []
```

```python
for file in tqdm(tifs[36:]):
    tif = imageio.imread(file).astype(int)
    mid = np.argmax(np.histogram(tif[:,:,0].reshape(-1),bins=256)[0])
    tif[tif <= mid] = tif[tif <= mid] / mid * 95
    tif[tif > mid] = (tif[tif > mid] - mid) / (255 - mid) * (255 - 95) + 95
    colored = np.copy(tif[:,:,:])
    colored[tif[:,:,0]<=15] = np.array([0,0,0])
    colored[np.logical_and(tif[:,:,0]>15, tif[:,:,0]<80)] = np.array([255,50,50])
    colored[np.logical_and(tif[:,:,0]>80, tif[:,:,0]<140)] = np.array([200,200,200])
    colored[np.logical_and(tif[:,:,0]>140, tif[:,:,0]<230)] = np.array([0,255,50])
    colored[tif[:,:,0]>=230] = np.array([255,255,255])
    os.makedirs(os.path.dirname(file).replace(rootdir, OUT_DIR), exist_ok=True)
    imageio.imwrite(file.replace(rootdir, OUT_DIR).replace(".tif", "segmented.png"), colored.astype(np.uint8))
    cimgs.append({
        "name": file,
        "tif": tif.astype(np.uint8),
        "cimg": colored.astype(np.uint8)
    })
    pickle.dump(cimgs, open(os.path.join(OUT_DIR, ".cache.pkl"), "wb"))
```

```python
for c in tqdm(cimgs):
    pickle.dump(c, open(c["name"].replace(rootdir, OUT_DIR).replace(".tif", "-cache.pkl"), "wb"))
```

```python
del cimgs
```

# Crack seeds
The darkest values in the image

```python
for tif in tqdm(tifs):
    c = pickle.load(open(tif.replace(rootdir, OUT_DIR).replace(".tif", "-cache.pkl"), "rb"))
    seeds = np.where(c["tif"][:,:,0]<=15)
    presegment = np.zeros((c["tif"].shape[0], c["tif"].shape[1]))
    presegment[c["tif"][:,:,0]<80] = 1
    
    for seed in zip(seeds[0], seeds[1]):
        if presegment[seed[0], seed[1]] == 1:
            presegment = flood_fill(presegment, seed, 2)
    segmented = np.copy(presegment)
    segmented[segmented == 1] = 0
    segmented[segmented == 2] = 1
    c["mask"] = segmented
    pickle.dump(c, open(c["name"].replace(rootdir, OUT_DIR).replace(".tif", "-cache.pkl"), "wb"))
    imageio.imwrite(c["name"].replace(rootdir, OUT_DIR).replace(".tif", "-mask.png"), 
                    (segmented * 255).astype(np.uint8))
    
```

# Skeletons

```python
from tqdm.notebook import tqdm as tq
```

```python
for tif in tqdm(tifs):
    c = pickle.load(open(tif.replace(rootdir, OUT_DIR).replace(".tif", "-cache.pkl"), "rb"))    
    skeleton, distance = medial_axis(c["mask"], return_distance=True)
    c["skeleton"] = skeleton
    c["labels"] = label(c["mask"])

    bounds = []    
    for l in tqdm(range(1, np.max(c["labels"])), leave=False):
        crack = np.zeros(c["mask"].shape, dtype=int)
        crack[c["labels"] == l] = 1    
        skeleton, distance = medial_axis(crack, return_distance=True)

        marked = mark_boundaries(c["mask"], crack, outline_color=(0.5,0,0),mode="outer")
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
    
    c["bounds"] = bounds
    pickle.dump(c, open(c["name"].replace(rootdir, OUT_DIR).replace(".tif", "-cache.pkl"), "wb"))
    
```

TODO: c["skeleton"] je blbě, je potřeba fixnout

```python
for tif in tq(tifs):
    c = pickle.load(open(tif.replace(rootdir, OUT_DIR).replace(".tif", "-cache.pkl"), "rb"))    
    skeleton, distance = medial_axis(c["mask"], return_distance=True)
    c["skeleton"] = skeleton
    imageio.imwrite(c["name"].replace(rootdir, OUT_DIR).replace(".tif", "-skeleton.png"), 
                    (c["skeleton"] * 255).astype(np.uint8))
    pickle.dump(c, open(c["name"].replace(rootdir, OUT_DIR).replace(".tif", "-cache.pkl"), "wb"))
```

```python

```
