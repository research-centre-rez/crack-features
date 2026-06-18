import os
import glob
import argparse
import logging
import pandas as pd
from cracks.features import process_groups
from cracks.utils import get_date_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Processing concrete groups")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the experiment root directory")
    
    args = parser.parse_args()
    root_dir = args.root_dir

    search_pattern = os.path.join(root_dir, "**", "*.npy")
    matrices = glob.glob(search_pattern, recursive=True)
    
    if not matrices:
        logger.error(f"No matricies within the root subtree: {root_dir}")
        return

    groups = {}
    for m in matrices:
        grp = os.path.dirname(m)
        groups.setdefault(grp, []).append(m)
    
    logger.info(f"Initializing processing with ROOT_DIR: {root_dir}")
    logger.info(f"Loaded {len(groups)} groups")

    out_dir = get_date_name()
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Saving outputs to directory: {out_dir}")

    process_groups(groups, out_dir, "concrete_processed.csv")

if __name__ == "__main__":
    main()