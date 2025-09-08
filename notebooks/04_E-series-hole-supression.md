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
%pwd
```

```python
%cd ..
```

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
from tqdm.notebook import tqdm
import pandas as pd
import pickle
import cracks
```

Define input and output directory. From the input folder all \*.tif files are loaded.

The output folder will contain python cache \*.pkl file and image output.

Filenames and directory names are preserved between input/output.

```python
rootdir = "/Users/gimli/cvr/data/microscopy/LargeAreas"
OUT_DIR = "/Users/gimli/cvr/data/microscopy/dip-results-LA"
```

Read all TIFF files

```python
tifs = []
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        if file[-4:] == ".tif" or file[-5:] == ".tiff":
            tifs.append(os.path.join(root, file))
```

```python
tifs
```

```python
img = imageio.imread(tifs[0])
```

```python
tif = cracks.normalize_hist(img)
```

```python
colored = cracks.threshold_image(tif[:,:,:3])
```

```python
plt.figure(figsize=(15,15))
plt.imshow(colored)
plt.show()
```

```python
for tif in tqdm(tifs):
    cracks.process_file(tif, rootdir, OUT_DIR, recompute=True)
```

```python
tifs[2]
```

```python
c = cracks.process_file(tifs[2], rootdir, OUT_DIR, levels=(50,90,160,230))
```

Focus on just one series

```python
e07 = [tif for tif in tifs if "E07" in tif]
e19 = [tif for tif in tifs if "E19" in tif]
e37 = [tif for tif in tifs if "E37" in tif]
```

# Test level of the cleanup of small cracks

In the code below one image is processed with different level of cleanup. This way will be created samples for evaluation by an expert.

```python
for tif in tqdm(e37[:1]):
    print(tif)
    c = pickle.load(open(tif.replace(rootdir, OUT_DIR).replace(".tif", "-cache.pkl"), "rb"))
```

The most important is number of segments and their sizes in the mask image.
Command below extract this information (for each segment, number of pixels is extracted).

```python
# E37
t1 = 50
t2 = 90
t3 = 160
t4 = 230
c = cracks.process_file(e37[1], rootdir, OUT_DIR, (t1, t2, t3, t4), False)
img = imageio.imread(e37[1])
for LENGTH in tqdm(range(10, 30)):
    cleaned = np.copy(img)
    for value, (count, length) in tqdm(c["label_stats"].items(), leave=False):
        if length > LENGTH:
            cleaned[c["labels"]==value] = np.array([255,0,0])
    imageio.imwrite(c["name"].replace(rootdir, OUT_DIR).replace(".tif", f"-cleaned{LENGTH}.png"), cleaned)
```

```python
# E07
t1 = 15
t2 = 65
t3 = 140
t4 = 230
c = cracks.process_file(e07[1], rootdir, OUT_DIR, (t1, t2, t3, t4), True)
img = imageio.imread(e07[1])
for LENGTH in tqdm(range(10, 30)):
    cleaned = np.copy(img)
    for value, (count, length) in tqdm(c["label_stats"].items(), leave=False):
        if length > LENGTH:
            cleaned[c["labels"]==value] = np.array([255,0,0])
    imageio.imwrite(c["name"].replace(rootdir, OUT_DIR).replace(".tif", f"-cleaned{LENGTH}.png"), cleaned)
```

```python
tifs
```

```python
for i in tqdm([2]):
    c = cracks._load_cache(tifs[i], rootdir, OUT_DIR)
    if "crops" not in c:
        img = imageio.imread(tifs[i])
    else:
        img = imageio.imread(tifs[i])[c["crops"][0][0]:c["crops"][0][1], c["crops"][1][0]:c["crops"][1][1], c["crops"][2][0]:c["crops"][2][1]]
    for LENGTH in tqdm(range(10, 30), leave=False):
        cleaned = np.copy(img[:,:,:3])
        for value, (count, length) in tqdm(c["label_stats"].items(), leave=False):
            if length > LENGTH:
                cleaned[c["labels"]==value] = np.array([255,0,0])
        imageio.imwrite(c["name"].replace(rootdir, OUT_DIR).replace(".tiff", f"-cleaned{LENGTH}.png"), cleaned)
```

```python
c["name"].replace(rootdir, OUT_DIR).replace(".tif", f"-cleaned{LENGTH}.png")
```

```python

```
