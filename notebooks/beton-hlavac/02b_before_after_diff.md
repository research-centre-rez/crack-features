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
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys

sys.path.append("../..")

from cracks import features 
from cracks.projection import adaptive_closing, heal_chipped_mask
from cracks.segment import pipeline_geometric

ROOT_DIR = "../../experiment3_registered"
PAIRS_CSV = os.path.join(ROOT_DIR, "before_after_pairs.csv")
```

```python

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
```

```python
def process_npy_to_features(npy_path, save_path=None):
    if not isinstance(npy_path, str) or not os.path.exists(npy_path):
        return pd.DataFrame()
    
    name = npy_path.split("/")[4]

    is_exp = "after" in npy_path
    stack = np.load(npy_path)

    mask, _, _ = adaptive_closing(stack)
    outer, inner = heal_chipped_mask(mask, inner_scale=0.85)

    cracks, bubbles, diff_proj = pipeline_geometric(stack, outer, inner)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(diff_proj, cmap="gray")
    ax.imshow(np.where(cracks > 0, 1, np.nan), cmap="Reds", vmin=0, vmax=1, alpha=0.8)
    ax.imshow(np.where(bubbles > 0, 1, np.nan), cmap="Blues", vmin=0, vmax=1, alpha=0.8)
    ax.set_title(f"{os.path.basename(npy_path)} {'After' if is_exp else 'Before'}", fontsize=12)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()

    table_cracks = features.compute_cracks(cracks)
    table_bubbles = features.compute_bubbles(bubbles)
    df_cracks = features.global_summary(pd.DataFrame(table_cracks), pd.DataFrame(table_bubbles))

    df_cracks['sample_id'] = name

    print(f"Image: {npy_path} Stage: {'AFTER' if is_exp else 'before'} \n{df_cracks}")
    
    del stack, mask, outer, inner, cracks, bubbles
    gc.collect()

    return df_cracks
```

```python
pairs_df = pd.read_csv(PAIRS_CSV)
results_dict = {}
col_before = 'Sample before exposure'
col_after = 'Sample after exposure'   
BASE_OUTPUT_DIR = "2026_05_17"

for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Comparing pairs"):
    file_before = os.path.join(ROOT_DIR, str(row.get(col_before, '')))
    file_after = os.path.join(ROOT_DIR, str(row.get(col_after, '')))

    if not os.path.exists(file_before) or not os.path.exists(file_after):
        print("Incomplete pair: are both loaded?")
        continue

    pair_dir = os.path.join(BASE_OUTPUT_DIR, "{:02d}".format(idx))
    os.makedirs(pair_dir, exist_ok=True)
    
    name_before = os.path.splitext(os.path.basename(file_before))[0] + ".png"
    name_after = os.path.splitext(os.path.basename(file_after))[0] + ".png"
    
    # save_before = os.path.join(pair_dir, name_before)
    # save_after = os.path.join(pair_dir, name_after)
    
    df_before  = process_npy_to_features(file_before)
    df_after = process_npy_to_features(file_after)
    
    df_before['stage'] = 'before'
    df_before['pair_id'] = idx
        
    df_after['stage'] = 'after'
    df_after['pair_id'] = idx
    
    print("pair loaded")
        
    results_dict[idx] = {
        'before': df_before,
        'after': df_after
    }
```

```python
temperature = {"1A1": 175, "1A2": 175, "1A4": 200, "1A5": 225, "1A6": 225, "1A7": 250, "1A8": 250, "1A9": 275, "1A10": 275, "1A11": 300, "1A12": 300}


all_before = [v['before'] for v in results_dict.values() if not v['before'].empty]
all_after = [v['after'] for v in results_dict.values() if not v['after'].empty]

df_before_combined = pd.concat(all_before)
df_after_combined = pd.concat(all_after)

agg_before = df_before_combined.groupby('pair_id').agg({
    **{col: 'sum' for col in df_before_combined.columns if col not in ['pair_id', 'sample_id']},
    'sample_id': 'first'
})

agg_after = df_after_combined.groupby('pair_id').agg({
    **{col: 'sum' for col in df_after_combined.columns if col not in ['pair_id', 'sample_id']},
    'sample_id': 'first'
})

numeric_cols = agg_after.select_dtypes(include=[np.number]).columns
diff_df = agg_after[numeric_cols] - agg_before[numeric_cols]

agg_before_prefixed = agg_before[numeric_cols].add_prefix('before_')
agg_after_prefixed = agg_after[numeric_cols].add_prefix('after_')
diff_df_prefixed = diff_df.add_prefix('diff_')

side_by_side = pd.concat([agg_before_prefixed, agg_after_prefixed, diff_df_prefixed], axis=1)

side_by_side['sample_id'] = agg_before['sample_id']
side_by_side['temperature'] = side_by_side['sample_id'].map(temperature)

cols = ['sample_id', 'temperature'] + [c for c in side_by_side.columns if c not in ['sample_id', 'temperature']]
side_by_side = side_by_side[cols]

display(side_by_side)
side_by_side.to_csv("2026_05_18_cracks_hybrid.csv")
```
