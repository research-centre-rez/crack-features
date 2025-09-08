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
```

```python
import os
import cracks
import matplotlib.pyplot as plt
import imageio
import numpy as np
from skimage.filters import gaussian
import pickle
from tqdm.auto import tqdm
from skimage.measure import label
from skimage.segmentation import flood, flood_fill, mark_boundaries
from skimage.morphology import medial_axis, skeletonize


```

```python
rootdir = "/Users/gimli/cvr/data/microscopy/LargeAreas"
OUT_DIR = "/Users/gimli/cvr/data/microscopy/dip-results-LA"
```

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
file = tifs[2]
```

```python
plt.figure(figsize=(15,15))
plt.imshow(imageio.imread(tifs[2])[9:-37,23:-21,:3])
plt.axvline()
plt.show()
```

```python
# Here is necessary to crop the boundaries ...
# At the end of the processing it is necessary to return boundaries back into all images
crops = ((9,-37),(23,-21),(0,-1))
img = imageio.imread(tifs[2])[crops[0][0]:crops[0][1],crops[1][0]: crops[1][1],crops[2][0]:crops[2][1]]
```

```python
# here are defined areas of image stitching
# Due to the issue of electron microscopy, the images are lighter on the right and darker on the left
# We try to compensate this, based on statistical error (med)
cuts = [0, 1000, 1022, 1920, 1945, 2840, 2868, 3759, 3800, 3895, 4050, tif.shape[1]]
med = np.median(img[:,:,0], 0)
```

```python
norms = []
for i in range(len(cuts) - 1): 
    norms.extend(np.linspace(med[cuts[i]], med[cuts[i+1] - 1], cuts[i+1] - cuts[i]).tolist())
    
plt.figure(figsize=(15,7))
plt.plot(np.median(img,0))
plt.plot(norms)
plt.title("Normalization of the lightness from the left to the right of the LA image")
plt.show()
```

```python
tif = np.copy(img[:,:,0])
#mid = np.apply_along_axis(lambda a: np.argmax(np.histogram(a, bins=255)[0]), 0, tif[:,:,0])
med = norms

for c in range(tif.shape[1]):
    column = tif[:, c]
    
    column[column <= med[c]] = column[column <= med[c]] / med[c] * 95
    column[column > med[c]] = (column[column > med[c]] - med[c]) / (255 - med[c]) * (255 - 95) + 95    
```

```python
t1 = 50
t2 = 75
t3 = 160
t4 = 230
```

```python
colored = np.zeros((tif.shape[0], tif.shape[1], 3), dtype=np.uint8)
colored[tif[:,:]<=t1] = np.array([0,0,0])
colored[np.logical_and(tif[:,:]>t1, tif[:,:]<=t2)] = np.array([255,50,50])
colored[np.logical_and(tif[:,:]>t2, tif[:,:]<=t3)] = np.array([200,200,200])
colored[np.logical_and(tif[:,:]>t3, tif[:,:]<t4)] = np.array([0,255,50])
colored[tif[:,:]>=230] = np.array([255,255,255])
```

```python
plt.figure(figsize=(15,15))
plt.imshow(c["cimg"])
plt.show()
```

```python
os.makedirs(os.path.dirname(file).replace(rootdir, OUT_DIR), exist_ok=True)
imageio.imwrite(file.replace(rootdir, OUT_DIR).replace(".tiff", "segmented.png"), colored.astype(np.uint8))
c = {
    "name": file,
    "crops": crops,
    "tif": np.stack([tif, tif, tif], axis=2),
    "cimg": colored.astype(np.uint8),
    "levels": [t1, t2, t3, t4]
}
pickle.dump(c, open(c["name"].replace(rootdir, OUT_DIR).replace(".tiff", "-cache.pkl"), "wb"))
```

```python
c = cracks.process_file(file, rootdir, OUT_DIR, levels=[t1, t2, t3, t4])
```

```python

```
