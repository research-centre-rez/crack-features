import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

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

def get_date_name():
    return datetime.now().strftime('%Y_%m_%d-%H%M%S')

def load_temp_spec(spec_path: str) -> dict:
    if not os.path.exists(spec_path):
        logger.error(f"Temperature specification file not found at: {spec_path}")
        raise FileNotFoundError(f"Specification file {spec_path} does not exist.")
        
    try:
        with open(spec_path, 'r', encoding='utf-8') as f:
            temp_map = json.load(f)
            
        temp_map = {str(k): float(v) for k, v in temp_map.items()}
        logger.info(f"Successfully loaded {len(temp_map)} entries from temperature spec.")
        return temp_map
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON spec file {spec_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading spec file {spec_path}: {e}")
        raise