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
%cd ..
%load_ext autoreload
%autoreload 2
```

# Extrakce malých bodů z vnitřku zrn

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import cracks
import imageio
from skimage.morphology import disk, square, erosion, closing, opening, label
from tqdm.auto import tqdm
```

```python
OUT_DIR = "/Users/gimli/cvr/data/microscopy/micro-F-samples-out"
rootdir = "/Users/gimli/cvr/data/microscopy/micro-F-samples"
```

```python
tifs = [os.path.join(rootdir, file) for file in os.listdir("/Users/gimli/cvr/data/microscopy/micro-F-samples") if file[-3:] == "tif"]
```

```python
tifs
```

```python
for tif in tifs[:1]:
    c = cracks.process_file(tif, rootdir, OUT_DIR, levels=(40, 82, 160, 220), recompute=False, crop=(2048, 2048))
    img = np.copy(c["tif"])
    img[np.logical_and(c["cimg"][:,:,0] == 255, c["cimg"][:,:,1] != 255)] = np.array([255,0,0])
    img[c["mask"] == 1.0] = [0,0,0]
    plt.figure(figsize=(15,15))
    plt.imshow(img, cmap="gray")
    plt.title(os.path.basename(c["name"]))
    plt.show()
```

```python
labels = label(img[:,:,0] != img[:,:,1])
ll, lc = np.unique(labels, return_counts=True)
im = np.copy(c["tif"])
for l, count in tqdm(zip(ll, lc), total=len(ll)):
    if 10 < count < 3000:
        im[labels==l] = [255,0,0]
im[c["mask"] == 1] = [0,0,250]
plt.figure(figsize=(15,15))
plt.title(c["name"])
plt.imshow(im)
plt.show()
```

```python
import cv2
```

```python
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(5,5))
```

```python
plt.figure(figsize=(15,15))
plt.title(c["name"])
plt.imshow(clahe.apply(c["tif"][:,:,0]), cmap="gray")
plt.show()
```

```python

```
