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
import os
import imageio.v3 as iio
from skimage.measure import label
from skimage.morphology import medial_axis
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
```

```python
ROOT = "/Users/gimli/cvr/data/beton/labels/cvat-antoine"
```

```python
def compute_stats(file_path):
    segmented = iio.imread(file_path)[:,:,0] == 255
    labeled = label(segmented)
    
    bounds = []
    crack_images = []
    skeletons = []
    distances = []
    boundaries = []
    for l in range(1, np.max(labeled)):        
        crack = np.zeros(segmented.shape, dtype=int)
        crack[labeled == l] = 1    
        skeleton, distance = medial_axis(crack, return_distance=True)
        
        marked = mark_boundaries(segmented, crack, outline_color=(0.5,0,0),mode="outer")
        boundary = np.where(marked[:,:,0] == 0.5)
        bounds.append({
            "label": l,
            "size": np.sum(crack),
            "length": np.sum(skeleton),
            "maxWidth": np.max(distance),
            "avgWidth": np.mean(distance[distance != 0]),
            "boundaryLength": len(boundary[0])
        })

        skeletons.append((l, skeleton))
        distances.append((l, distance))
        boundaries.append((l, boundary))

        crack_image = np.stack([segmented, segmented, segmented], axis=2)
        crack_image[labeled==l, 0:2] = 0
        crack_images.append((l, crack_image))
    return labeled, skeletons, distances, boundaries, bounds, crack_images
```

```python
stats = {}
for root, dirs, files in os.walk(ROOT):
    for file in files:
        if file.endswith(".png"):
            dir = root.split(os.path.sep)[-1]
            print(f"Processing {dir}-{file}")
            stats[f"{dir}-{file}"] = compute_stats(os.path.join(root, file))            
```

```python
for stat_key in stats:    
    sample = stat_key.split("-")[0]
    scan = stat_key.split("-")[1][:-4]
    for crack_id, image in stats[stat_key][-1]:
        folder = os.path.join("/Users/gimli/Downloads/cracks-2/crack-maps", sample, scan)
        os.makedirs(folder, exist_ok=True)
        iio.imwrite(os.path.join("/Users/gimli/Downloads/cracks-2/crack-maps", sample, scan, f"{crack_id:03d}-crack.png"), 
                    cv2.cvtColor(image.astype(np.uint8)*255, cv2.COLOR_BGR2RGB))
```

```python
import cv2
```

```python
plt.imshow(cv2.cvtColor(stats["3A-after_1.png"][-1][0][1].astype(np.uint8)*255, cv2.COLOR_BGR2RGB))
plt.show()
```

```python
import pandas as pd
```

```python

```

```python
for key in stats.keys():
    os.makedirs(os.path.join(ROOT, "stats"), exist_ok=True)
    pd.DataFrame(stats[key]).to_csv(f"{ROOT}/stats/{key.split('.')[0]}.csv")
```

```python

```
