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
docker run --rm \
    -v $(pwd):/workspace \
    -v <path_to_data_on_host>:/data \
    -w /workspace \
    cracks \
    --root_dir /workspace<experiment_path>/
```

replacting `<experiment_path>` with the path to your prepared folder. The folder has the following structure.

## Experiment structure

- `after` - containting .npy stacks
- `before` - containting .npy stacks

## Output format

Contents created during each experiment run

`concrete_processed.csv` - cracks/bubbles measurements

- `group_name` - name of the subdirectory

- `skeletonSum_px` - skeleton area measurements for the group

- `crackBoundaryLengthSum_px` - boundary length measurements for the group

- `bubbleAreaSum_px` - bubble area measurements for the group

- `bubbleBoundaryLengthSum_px` - bubble boundary length measurements for the group

- `bubbleCount_px` - number of uniquely segmented bubbles in the group

For each processed file:

Synthetic IQR projection visualization
Segmented cracks (red) and bubbles (blue) masks

