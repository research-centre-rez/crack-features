import os
import argparse
import logging
import pandas as pd
from cracks.features import process_pairs
from cracks.utils import get_date_name, load_temp_spec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Process crack before/after pairs.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the experiment root directory")
    
    args = parser.parse_args()

    root_dir = args.root_dir
    pairs_csv = os.path.join(root_dir, "before_after_pairs.csv")
    
    logger.info(f"Initializing processing with ROOT_DIR: {root_dir}")
    logger.info(f"Using pairs CSV: {pairs_csv}")
    
    temperature_map = load_temp_spec(root_dir + "temp_spec.json")
    logger.info(f"Loaded temperature specification for {len(temperature_map)} samples.")

    if not os.path.exists(pairs_csv):
        logger.error(f"Pairs CSV file not found at {pairs_csv}")
        return

    pairs_df = pd.read_csv(pairs_csv)
    logger.info(f"Loaded {len(pairs_df)} before/after pairs from CSV.")

    out_dir = get_date_name()
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Saving outputs to directory: {out_dir}")

    process_pairs(pairs_df, root_dir, out_dir, "concrete_processed.csv", temperature_map)

if __name__ == "__main__":
    main()