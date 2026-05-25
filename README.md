# Crack-features
## Installation

### Local
Clone the repository, create a fresh virtual environment. Install the editable version of the package with

```bash
pip install -e .
```

### Docker
Build the image
```bash
docker build -t cracks .
```

Run it with

```bash
docker run --rm     -v $(pwd)/<experiment_path>:/data     cracks     --root_dir /data/
```
replacting `<experiment_path>` with the path to your prepared folder. The folder has the following structure.

## Experiment structure
- `before_after.csv` - measured pair paths
- `temp_spec.json` - concrete samples matched to their exposed temperatures
- `after_expo` - containting .npy stacks
- `before_expo` - containting .npy stacks

## Output format
Contents created during each experiment run

`concrete_processed.csv` - cracks/bubbles measurements

- `temperature` - specified by `temp_spec.json`
- `before_skeletonSum_px` - area of the skeleton before exposure
- `after_skeletonSum_px` - area of the skeleton after exposure
- `diff_skeletonSum_px` - difference, invariant: should grow

- `before_crackBoundaryLengthSum_px` - length of the boundary before exposure
- `after_crackBoundaryLengthSum_px` - length of the boundary after exposure
- `diff_crackBoundaryLengthSum_px` - difference, no invariant for now

- `before_bubbleAreaSum_px` 
- `after_bubbleAreaSum_px` 
- `diff_bubbleAreaSum_px` 

- `before_bubbleBoundaryLengthSum_px` - length of the boundary before exposure
- `after_bubbleBoundaryLengthSum_px` - length of the boundary after exposure
- `diff_bubbleBoundaryLengthSum_px` - difference, no invariant for now

- `before_bubbleCount_px` - number of bubble connected components before
- `after_bubbleCount_px` - number of bubble connected components after 
- `diff_bubbleCount_px` - difference

For each pair:
- before/after IQR projection
- segmented cracks (red) and bubbles (blue) for both stages