---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import math, copy, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import os
from tqdm.auto import tqdm
from skimage.morphology import label
```

```python
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)
```

```python
@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, dfs: List[pd.DataFrame], feature_cols: Optional[Sequence[str]]=None):
        if feature_cols is not None:
            X = [df[feature_cols].to_numpy(dtype=np.float32) for df in dfs]
        else:
            X = [df.to_numpy(dtype=np.float32) for df in dfs]
        Xcat = np.concatenate(X, axis=0)  # [sum(Ni), 6]
        mean = Xcat.mean(axis=0)
        std = Xcat.std(axis=0)
        std[std < 1e-8] = 1.0  # ochrana proti nulové varianci
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]]=None):
        X = df[feature_cols].to_numpy(dtype=np.float32) if feature_cols else df.to_numpy(dtype=np.float32)
        return (X - self.mean) / self.std
```

```python
class SetDataset(Dataset):
    def __init__(self, dfs: List[pd.DataFrame], y: List[float],
                 standardizer: Optional[Standardizer]=None,
                 feature_cols: Optional[Sequence[str]]=None):
        assert len(dfs) == len(y)
        self.dfs = dfs
        self.y = np.asarray(y, dtype=np.float32)
        self.feature_cols = feature_cols
        self.standardizer = standardizer

    def __len__(self): return len(self.dfs)

    def __getitem__(self, idx):
        df = self.dfs[idx]
        X = df[self.feature_cols].to_numpy(dtype=np.float32) if self.feature_cols else df.to_numpy(dtype=np.float32)
        if self.standardizer is not None:
            X = (X - self.standardizer.mean) / self.standardizer.std
        x = torch.from_numpy(X)             # [Ni, 6]
        y = torch.tensor(self.y[idx])       # []
        return x, y
```

```python
def collate_pad(batch):
    # batch = list of (x:[Ni,6], y:[])
    xs, ys = zip(*batch)
    Nmax = max(x.shape[0] for x in xs)
    B = len(xs)
    feat = xs[0].shape[1]
    Xpad = torch.zeros(B, Nmax, feat, dtype=torch.float32)
    mask = torch.zeros(B, Nmax, dtype=torch.float32)
    for i, x in enumerate(xs):
        n = x.shape[0]
        Xpad[i, :n] = x
        mask[i, :n] = 1.0
    y = torch.stack(list(ys))  # [B]
    return Xpad, mask, y
```

```python
class TinyDeepSets(nn.Module):
    def __init__(self, in_dim=6, hid=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid)
        )
        self.head = nn.Sequential(
            nn.Linear(2*hid, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, X, mask=None):
        # X: [B,N,6], mask: [B,N]
        H = self.enc(X)  # [B,N,hid]
        if mask is None:
            mean_pool = H.mean(dim=1)
            max_pool = H.max(dim=1).values
        else:
            # mean s maskou
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
            mean_pool = (H * mask.unsqueeze(-1)).sum(dim=1) / denom
            # max s maskou (vyřadíme neplatné pozice -inf)
            H_masked = H.masked_fill((mask==0).unsqueeze(-1), float('-inf'))
            max_pool = torch.max(H_masked, dim=1).values
            # když má sample N=0 (nemělo by nastat), nahradíme -inf nulou
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B,2*hid]
        out = self.head(pooled).squeeze(-1)                # [B]
        return out
```

```python
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-3
    epochs: int = 1000
    patience: int = 50
    batch_size: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    huber_delta: float = 1.0  # Huber loss pro robustnost
```

```python
def train_model(train_ds, val_ds, cfg: TrainConfig):
    model = TinyDeepSets(in_dim=6, hid=16).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.SmoothL1Loss(beta=cfg.huber_delta)  # Huber

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_pad)
    val_ld   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_pad)

    best_state, best_val = None, float('inf')
    wait = 0
    for ep in range(cfg.epochs):
        model.train()
        for X, mask, y in train_ld:
            X, mask, y = X.to(cfg.device), mask.to(cfg.device), y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            pred = model(X, mask)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # validace
        model.eval()
        with torch.no_grad():
            vals, ns = 0.0, 0
            for X, mask, y in val_ld:
                X, mask, y = X.to(cfg.device), mask.to(cfg.device), y.to(cfg.device)
                pred = model(X, mask)
                vals += torch.mean((pred - y)**2).item() * X.size(0)  # MSE pro monitorování
                ns += X.size(0)
            val_mse = vals / max(ns,1)
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val
```

```python
# ------- helper: K-fold (volitelné) -------
def kfold_cv(dfs, y, feature_cols=None, k=5):
    assert len(dfs) >= k, "Na k-fold potřebuješ aspoň k vzorků."
    idx = np.arange(len(dfs))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    for tr, va in kf.split(idx):
        std = Standardizer.fit([dfs[i] for i in tr], feature_cols)
        tr_ds = SetDataset([dfs[i] for i in tr], [y[i] for i in tr], std, feature_cols)
        va_ds = SetDataset([dfs[i] for i in va], [y[i] for i in va], std, feature_cols)
        model, val = train_model(tr_ds, va_ds, TrainConfig())
        scores.append(val)
    return scores
```

<!-- #region -->
# Usage

1) Připrav vstup:
    - dfs: List[pd.DataFrame], každý má 6 sloupců (3 int, 3 float)
    - targets: List[float]
```python
feature_cols = ["i1","i2","i3","f1","f2","f3"]  # pokud máš názvy sloupců
std = Standardizer.fit(dfs_train, feature_cols)
train_ds = SetDataset(dfs_train, y_train, std, feature_cols)
val_ds   = SetDataset(dfs_val,   y_val,   std, feature_cols)
model, best_val_mse = train_model(train_ds, val_ds, TrainConfig())
```

2) Infer:
```python
new_df_set: List[pd.DataFrame]  # nové množiny
std # musí být ten z tréninku!
test_ds = SetDataset(new_df_set, [0.0]*len(new_df_set), std, feature_cols)
test_ld = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_pad)
model.eval(); preds=[]
with torch.no_grad():
    for X, mask, _ in test_ld:
        p = model(X.to(TrainConfig().device), mask.to(TrainConfig().device)).cpu().numpy().tolist()
        preds += p
print(preds)
```

<!-- #endregion -->

```python
dfs = []
keys = []
ROOT = "/Users/gimli/cvr/data/beton/erik-features"
for file in os.listdir("/Users/gimli/cvr/data/beton/erik-features"):
    if file.endswith("cracks.csv"):
        keys.append(file[:2])
        dfs.append(pd.read_csv(os.path.join(ROOT, file)))
```

```python
dfs[0]
```

```python
vzorky_cem = pd.read_excel("/Users/gimli/cvr/data/beton/Vzorky_cem1.xlsx", header=[1,2])
```

```python
metadata = vzorky_cem[(vzorky_cem["filename"]["před"] != "-") & (vzorky_cem["filename"]["po"] != "-") & (vzorky_cem["UZ [µs]"]["před"].notna())]
```

```python
plt.figure(figsize=(15,5))
ax = plt.subplot(1,2,1)
ax.scatter(vzorky_cem["UZ [µs]"]["před"], vzorky_cem["UZ [µs]"]["po"])
x = vzorky_cem["UZ [µs]"]["před"]
y = vzorky_cem["UZ [µs]"]["po"]
idx = np.isfinite(x) & np.isfinite(y)
fit = np.polyfit(x[idx], y[idx] , 1)
ax.plot(np.linspace(np.min(x), np.max(x), 100), np.polyval(fit, np.linspace(np.min(x), np.max(x), 100)), label="fit", color="red")
ax.scatter(metadata["UZ [µs]"]["před"], metadata["UZ [µs]"]["po"], label="selected samples")
ax.set_title("Difference in UZ", fontsize=15)
ax.set_xlabel("UZ [µs] before", fontsize=14)
ax.set_ylabel("UZ [µs] after", fontsize=14)
ax.legend(fontsize=13)

ax = plt.subplot(1,2,2)
ax.scatter(vzorky_cem["UZ [µs]"]["před"]/vzorky_cem["expozice"]["teplota [°C]"], vzorky_cem["UZ [µs]"]["po"]/vzorky_cem["expozice"]["teplota [°C]"])
x = vzorky_cem["UZ [µs]"]["před"]/vzorky_cem["expozice"]["teplota [°C]"]
y = vzorky_cem["UZ [µs]"]["po"]/vzorky_cem["expozice"]["teplota [°C]"]
idx = np.isfinite(x) & np.isfinite(y)
fit = np.polyfit(x[idx], y[idx] , 1)
ax.plot(np.linspace(np.min(x), np.max(x), 100), np.polyval(fit, np.linspace(np.min(x), np.max(x), 100)), label="fit", color="red")
ax.scatter(metadata["UZ [µs]"]["před"]/metadata["expozice"]["teplota [°C]"], metadata["UZ [µs]"]["po"]/metadata["expozice"]["teplota [°C]"], label="selected samples")
ax.set_title("UZ diff normalized by exposure temperature", fontsize=15)
ax.set_xlabel("UZ [µs]/temp [°C] before", fontsize=14)
ax.set_ylabel("UZ [µs]/temp [°C] after", fontsize=14)
ax.legend(fontsize=13)
plt.show()
```

```python
cracks_dir = "/Users/gimli/cvr/data/beton/erik-features"
stacks_dir = "/Users/gimli/cvr/data/beton/erik"
dfs = []
stacks = []
for prefix_before, prefix_after in zip(metadata["filename"]["před"], metadata["filename"]["po"]):
    sample = []
    for suffix in ["cracks.csv", "pores.csv", "boundary.csv"]:
        sample.append([
            pd.read_csv(os.path.join(cracks_dir, prefix_before + suffix)),
            pd.read_csv(os.path.join(cracks_dir, prefix_after + suffix))
        ])
    stacks.append([
        np.load(os.path.join(stacks_dir, prefix_before[:-1] + ".npy")),
        np.load(os.path.join(stacks_dir, prefix_after[:-1] + ".npy"))
    ])
    dfs.append(sample)

```

```python
metadata["filename"]["před"][5]
```

```python
dfs[2][0][0].mean()
```

```python
dfs[2][0][1].mean()
```

```python
colors=["red", "blue", "green", "orange"]
plt.figure(figsize=(15,8))
ax = plt.subplot(1,2,1)
ax.imshow(stacks[2][0][500], cmap="gray")
ax.scatter([465, 980, 1382, 1071],[570, 990, 539, 53], color=colors, alpha=0.5, marker="+", s=20)
ax.set_xlim(400, 1400)
#ax.set_ylim(500, 700)
ax = plt.subplot(1,2,2)
ax.imshow(stacks[2][1][0], cmap="gray")
ax.scatter([456, 163, 695, 1100],[118, 718, 1005, 590], color=colors, alpha=0.5, marker="+", s=20)
ax.set_xlim(100, 1200)
#ax.set_ylim(500, 700)
plt.show()
```

```python
import cv2
```

```python
ptsSrc = np.array([[465, 980, 1382, 1071],[570, 990, 539, 53]]).astype(np.float32).T
ptsDst = np.array([[456, 163, 695, 1100],[118, 718, 1005, 590]]).astype(np.float32).T
```

```python
tform = cv2.getPerspectiveTransform(ptsSrc, ptsDst)
```

```python
np.savetxt("/Users/gimli/cvr/data/beton/erik-features/3C-part2_processed_registered_stack-perspective-params.txt",tform)
```

```python
before = cv2.warpPerspective(stacks[2][0][0], tform, (1920, 1080))
```

```python
colors=["red", "blue", "green", "orange"]
plt.figure(figsize=(15,8))
ax = plt.subplot(1,2,1)
ax.imshow(before, cmap="gray")
ax.set_xlim(100, 1200)
ax = plt.subplot(1,2,2)
ax.imshow(stacks[2][1][0], cmap="gray")
ax.set_xlim(100, 1200)
#ax.set_ylim(500, 700)
plt.show()
```

```python
from cracks.concrete_cylinders import detection
```

```python
sample_no = 5
stack_files = [
    os.path.join(stacks_dir, metadata["filename"]["před"][sample_no][:-1] + ".npy"),
    os.path.join(stacks_dir, metadata["filename"]["po"][sample_no][:-1] + ".npy")
]
```

```python
masks_1 = detection.circular_mask(stack_files[0], 3)
```

```python
masks_2 = detection.circular_mask(stack_files[1], 3)
```

```python
layers = [
    masks_1, masks_2
]
```

```python
plt.figure(figsize=(15,5))
ax = plt.subplot(1,2,1)
ax.imshow(stacks[2][0][0], cmap="gray")
ax.imshow(masks_1[2], cmap="gray", alpha=0.5)
ax = plt.subplot(1,2,2)
ax.imshow(stacks[2][1][0], cmap="gray")
ax.imshow(masks_2[2], cmap="gray", alpha=0.5)
plt.show()
```

```python
normalized_stacks = [detection.normalize_lightness(stack_file, layer)
                         for stack_file, layer in tqdm(zip(stack_files, layers), desc="Lightness normalization")
                         if layer[0] is not None]
```

```python
cracks_and_holes = [
        label(np.logical_or(normalized_stack * layer[2] > 20, normalized_stack * layer[2] < -20))
        for normalized_stack, layer in tqdm(zip(normalized_stacks, layers), desc="Rough defect segmentation")
        if layer[0] is not None
    ]
```

```python
labeled = label((cracks_and_holes[1] != 0))
```

```python
np.unique(labeled)
```

```python
mask = (cracks_and_holes[1] != 0)
for lid in np.unique(labeled):
    if np.sum(labeled == lid) < 40:
        mask[labeled == lid] = 0
```

```python
img = np.stack([
    cv2.warpPerspective((cracks_and_holes[0] != 0).astype(np.uint8) * 255, tform, (1920, 1080)),
    mask.astype(np.uint8) * 255,
    (cracks_and_holes[1] != 0).astype(np.uint8) * 255,

], axis=2)
```

```python
plt.figure(figsize=(10, 10))
plt.imshow(stacks[2][1][0], cmap="gray")
plt.imshow(img, alpha=0.3)
plt.xlim(90, 1150)
plt.show()
```

```python
plt.figure(figsize=(15, 10))
ax = plt.subplot(1,2,1)
ax.imshow(cv2.warpPerspective(stacks[2][0][0].astype(np.uint8), tform, (1920, 1080)), cmap="gray")
ax.set_xlim(90, 1150)
ax = plt.subplot(1,2,2)
ax.imshow(stacks[2][1][0], cmap="gray")
ax.set_xlim(90, 1150)
ax.set_yticks([])
plt.show()
```

```python
plt.figure(figsize=(15,5))
ax = plt.subplot(1,2,1)
ax.imshow(cv2.warpPerspective((cracks_and_holes[0] != 0).astype(np.uint8), tform, (1920, 1080)))
ax = plt.subplot(1,2,2)
ax.imshow(cracks_and_holes[1] != 0)
plt.show()
```

DEPTH MAP

```python
stacks[2][0].shape
```

```python
import numpy as np

# -----------------------
# Scene & acquisition parameters
# -----------------------
K = 655                         # number of lighting positions
h_cm = 2.0                      # light height above surface [cm]
R_cm = 7.0                      # circular path radius [cm]
sample_diam_cm = 2.0            # concrete disk diameter [cm]
px_per_cm = 1000.0 / 2.0        # 2 cm -> 1000 px => 500 px/cm

H, W = 1080, 1920               # image size (FullHD)
cx, cy = (W - 1) / 2.0, (H - 1) / 2.0  # image center in pixels
sample_radius_px = (sample_diam_cm / 2.0) * px_per_cm

# -----------------------
# 1) Far-field-style L (Kx3) at the surface center
#    Each row is a unit vector pointing from the surface toward the light.
#    Useful for "dummy L" in classic photometric stereo code.
# -----------------------
angles = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)  # 0 .. 360 deg

# Light positions in world coords [cm], moving on a horizontal circle at z = h_cm
light_pos = np.stack([
    R_cm * np.cos(angles),      # x
    R_cm * np.sin(angles),      # y
    np.full_like(angles, h_cm)  # z
], axis=1)  # shape: (K, 3)

# Direction from the surface CENTER (0,0,0) toward the light, then normalized
L = light_pos / np.linalg.norm(light_pos, axis=1, keepdims=True)  # (K,3), unit vectors

# Optional: relative 1/r^2 gain at the center (useful if you want near-field roll-off)
r_center = np.linalg.norm(light_pos, axis=1)     # distance from center to light [cm]
gain_center = 1.0 / (r_center ** 2)              # shape: (K,)

# L is your "dummy" light matrix
print("L shape:", L.shape)         # (655, 3)
print("Example first 3 rows of L:\n", np.round(L[:3], 6))
print("Center gains (first 3):", np.round(gain_center[:3], 6))

# -----------------------
# 2) Near-field (per-pixel) helpers — compute d_k(x,y) and a_k(x,y) on demand
#    (Do this only on the sample mask to avoid huge memory.)
# -----------------------
# Build a circular mask for the concrete disk (optional but recommended)
yy, xx = np.mgrid[0:H, 0:W]
rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
sample_mask = rr <= sample_radius_px  # boolean mask of the specimen area

# Precompute world coords [cm] for each pixel (orthographic camera)
# Set the world origin at the disk center; +x right, +y down (image convention), z=0 on the surface.
X_cm = (xx - cx) / px_per_cm
Y_cm = (yy - cy) / px_per_cm
Z_cm = np.zeros_like(X_cm)  # surface is z = 0 everywhere

def per_light_direction_and_gain(k, mask=None):
    """
    Compute, for a single light index k:
      - d(x,y): unit vector from surface point (x,y,0) to light position (xL,yL,h), shape (H,W,3)
      - a(x,y): inverse-square falloff 1/r^2, shape (H,W)

    Use `mask` to restrict computation (e.g., to the concrete disk) for speed/memory.
    """
    xL, yL, zL = light_pos[k]  # cm
    # Vector from surface point to light
    vx = xL - X_cm
    vy = yL - Y_cm
    vz = zL - Z_cm
    r = np.sqrt(vx * vx + vy * vy + vz * vz) + 1e-12
    d_unit = np.stack([vx / r, vy / r, vz / r], axis=-1)  # (H,W,3)
    a_inv_r2 = 1.0 / (r * r)                              # (H,W)

    if mask is not None:
        # Return compact arrays limited to the mask’s bounding box to save RAM
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        d_crop = d_unit[y0:y1, x0:x1, :].copy()
        a_crop = a_inv_r2[y0:y1, x0:x1].copy()
        mask_crop = mask[y0:y1, x0:x1]
        return (d_crop, a_crop, mask_crop, (y0, y1, x0, x1))
    else:
        return (d_unit, a_inv_r2)

# Example: get near-field direction & gain for light #0 over the specimen only
d0_crop, a0_crop, mask_crop, bbox = per_light_direction_and_gain(0, mask=sample_mask)
print("Per-pixel near-field d for k=0 (crop) shape:", d0_crop.shape)  # (h,w,3)
print("Per-pixel near-field a for k=0 (crop) shape:", a0_crop.shape)  # (h,w)

# -----------------------
# Notes / usage:
# - Use L (Kx3) directly if you want a simple, far-field-like photometric stereo dummy.
# - For physically correct near-field PS, use per_light_direction_and_gain(k, ...) inside your solver:
#     I_k(x,y) ≈ ρ(x,y) * a_k(x,y) * d_k(x,y) · n(x,y)
# - Units are centimeters; consistent units matter, the absolute scale does not (except for 1/r^2).
# - Surface normal (camera-facing) is +z; camera is orthographic and parallel to the slab.
# -----------------------
```

```python
import numpy as np
import cv2 as cv
from numpy.fft import fft2, ifft2, fftfreq

# imgs: list of HxW float32 images in [0,1], same viewpoint
# L: Kx3 array of light directions scaled by relative intensity (rows l_k^T)                # (K,3)
I = stacks[2][0].astype(np.float32)/255.0
K,H,W = I.shape

# Shadow/saturation masks
shadow = I < 0.02
sat    = I > 0.98
valid  = ~(shadow | sat)

# IRLS to solve g = rho*n per pixel
LtL = L.T @ L
Lt  = L.T
LtL_inv = np.linalg.inv(LtL)  # if L well-conditioned; else use np.linalg.pinv

g = np.zeros((H,W,3), np.float32)
for y in tqdm(range(H)):
    Ik = I[:,y,:]                  # (K,W)
    Vk = valid[:,y,:]              # (K,W)
    # Initialize with ordinary LS using only valid obs
    for x in range(W):
        v = Vk[:,x]
        if v.sum() >= 3:
            Lv = L[v]
            b  = Ik[v,x]
            # IRLS: 3-5 iterations with Huber weights
            w = np.ones(v.sum(), np.float32)
            gpx = np.zeros(3, np.float32)
            for _ in range(4):
                Wm = np.diag(w)
                A  = Lv.T @ Wm @ Lv
                bW = Lv.T @ Wm @ b
                gpx = np.linalg.solve(A, bW)
                r   = b - Lv @ gpx
                # Huber weighting
                s = 1.4826*np.median(np.abs(r)) + 1e-6
                t = np.abs(r)/(1.345*s)
                w = 1/np.maximum(1, t)
            g[y,x,:] = gpx
        else:
            g[y,x,:] = 0

rho = np.linalg.norm(g, axis=2) + 1e-8
n   = g / rho[...,None]
nz  = np.clip(n[...,2], 1e-4, 1)   # avoid division by 0
p = -n[...,0] / nz
q = -n[...,1] / nz

# Frankot–Chellappa integration
def integrate_fc(p, q):
    H,W = p.shape
    wx = 2*np.pi*fftfreq(W)
    wy = 2*np.pi*fftfreq(H)
    WX, WY = np.meshgrid(wx, wy)
    denom = WX**2 + WY**2
    Px = fft2(p)
    Qy = fft2(q)
    Z  = (-1j*WX*Px - 1j*WY*Qy) / np.where(denom==0, 1, denom)
    z  = np.real(ifft2(Z))
    z -= np.median(z)        # fix constant
    return z

z = integrate_fc(p, q)

# Remove best-fit plane to get relative depth (grooves)
YY, XX = np.mgrid[0:H, 0:W]
A = np.c_[XX.ravel(), YY.ravel(), np.ones(H*W)]
coef, _, _, _ = np.linalg.lstsq(A, z.ravel(), rcond=None)
plane = (A @ coef).reshape(H,W)
z_rel = z - plane

# z_rel is your depth map (up to an overall scale if light intensities unknown)

```

```python
plt.imshow(z_rel)
plt.show()
```

```python
y_dummy = np.random.rand(len(dfs)) * 10
```

```python
feature_cols = ["crackSize_px","skeleton_px", "boundaryLength_px", "maxWidth_px","avgWidth_px","farthestPoints_px"]  # pokud máš názvy sloupců
std = Standardizer.fit(dfs, feature_cols)
train_ds = SetDataset(dfs[:10], y_dummy[:10], std, feature_cols)
val_ds   = SetDataset(dfs[10:], y_dummy[10:], std, feature_cols)
model, best_val_mse = train_model(train_ds, val_ds, TrainConfig())
```

```python
best_val_mse
```

```python
new_df_set = dfs[10:]
test_ds = SetDataset(new_df_set, [0.0]*len(new_df_set), std, feature_cols)
test_ld = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_pad)
model.eval()

preds=[]
with torch.no_grad():
    for X, mask, _ in test_ld:
        p = model(X.to(TrainConfig().device), mask.to(TrainConfig().device)).cpu().numpy().tolist()
        preds += p
print(preds)
```

```python
y_dummy[10:]
```

```python
", ".join(np.unique(sorted(keys)))
```

```python
pd.read_excel("/Users/gimli/cvr/data/beton/Vzorky_cem1.xlsx", header=[1, 2])
```
