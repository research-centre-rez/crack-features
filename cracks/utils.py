
import matplotlib.pyplot as plt
import numpy as np

def peek(img, title="Pipeline Step", cmap=None, size=6):
    plt.figure(figsize=(size, size))
    
    if cmap is None:
        if img.dtype == bool or np.unique(img).size <= 2:
            cmap = "viridis"
        else:
            cmap = "gray"
            
    plt.imshow(img, cmap=cmap)
    plt.title(f"DEBUG: {title} {img.shape}")
    plt.axis("off")
    plt.show()
