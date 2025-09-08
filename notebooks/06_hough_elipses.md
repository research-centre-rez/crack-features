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
import imageio
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import numpy as np
```

```python
img = imageio.imread("/Users/gimli/cvr/data/microscopy/circles/MicrosoftTeams-image.png")
```

```python
plt.figure(figsize=(15,15))
plt.imshow(img[:1024,:,1], cmap="gray")
plt.grid()
plt.xticks(np.arange(0,1024,32))
plt.show()
```

```python
edges = canny(img[:1024,:,1], sigma=3, low_threshold=10, high_threshold=50)
```

```python
plt.imshow(edges[:128,:128], cmap="gray")
plt.show()
```

```python
%%time
hough_radii = range(10, 50)
result = hough_circle(edges,
                     hough_radii, normalize=False)
accums, cx, cy, radii = hough_circle_peaks(result, hough_radii,
                                           total_num_peaks=1000)
```

```python
image.shape
```

```python
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
image = np.copy(img[:1024,:,:3])
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=(image.shape[0], image.shape[1]))
    image[circy, circx, :] = [220, 20, 20]

ax.imshow(image, cmap=plt.cm.gray)
plt.show()
```

```python
image_rgb = np.copy(img[:1024,:,:3])
e = np.copy(edges)
```

```python
# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
print(yc, xc, a, b, orientation)
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
e = color.gray2rgb(img_as_ubyte(e))
e[cy + 60, cx + 80] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 15),
                                sharex=True, sharey=True)

ax1.set_title('Original picture')
ax1.imshow(img[:128,:128,:])

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(e[:128,:128,:])

plt.show()
```

```python
ellipse_perimeter(yc, xc, a, b, orientation)
```

```python
plt.figure(figsize=(10,10))
plt.imshow(1-edges, cmap="gray")
plt.show()
```

```python
hough_radii = np.arange(20, 35, 2)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent 3 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=3)

# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 10))
image = color.gray2rgb(img[:1024,:,1])
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)
    image[circy, circx] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()
```

```python

```
