import os
import glob
import numpy as np
from tqdm import tqdm

def calculate_global_constants(root_dir):
    # Find all .npy files in any subdirectory
    search_path = os.path.join(root_dir, "**", "*.npy")
    npy_files = glob.glob(search_path, recursive=True)
    
    if not npy_files:
        print(f"No .npy files found in {root_dir}")
        return None, None

    print(f"Found {len(npy_files)} files. Calculating global baseline...")
    
    low_percentiles = []
    iqr_values = []
    
    for file_path in tqdm(npy_files, desc="Analyzing dataset lighting"):
        try:
            # Memory-map the file to avoid loading massive arrays entirely into RAM
            stack = np.load(file_path, mmap_mode='r')
            
            # Spatial downsample (every 40th frame) to speed up math
            subset = stack[::40, :, :]
            
            # 1. Measure the baseline low intensity (5th percentile)
            low_p = np.percentile(subset, 5)
            low_percentiles.append(low_p)
            
            # 2. Measure the IQR spread (75th - 25th percentile)
            q75, q25 = np.percentile(subset, [75, 25])
            iqr_values.append(q75 - q25)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    global_low_intensity = np.mean(low_percentiles)
    global_iqr_scale = np.mean(iqr_values)
    
    print("\n" + "="*40)
    print("CALCULATED GLOBAL CONSTANTS")
    print("="*40)
    print(f"GLOBAL_LOW_INTENSITY = {global_low_intensity:.2f}")
    print(f"GLOBAL_IQR_SCALE     = {global_iqr_scale:.2f}")
    print("="*40)
    
    return global_low_intensity, global_iqr_scale

if __name__ == "__main__":
    # Target directory path
    target_dir = "../experiment3_registered"
    calculate_global_constants(target_dir)