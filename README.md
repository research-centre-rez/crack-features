# Crack features

Toolbox for feature computation for crack and phase maps. Repository contains image processing algorithms for analysis of electron microscopy images.

## Installation
Follow standard virtual environment creation by using `venv`, `poetry` or `conda`. This software was developed by using Python version 3.10, but hopefully works for wide range of Python versions.

3rd party packages are defined in `requirements.txt` and should be installed by pip:
```bash
pip install -r requirements.txt
```
However, any dependency hell can be easily resolved by removing specific versions mentioned there. The software itself do not use any of known Python-version issues like pickle serialization or tensorflow an PyTorch version-specific formats. 

## Crack detection
Applies two step thresholding for crack detection. 
First threshold creates seeds. 
The second threshold define areas where flood fill is applied.

This functionality covers `cracks/detection.py`. For usage see:
```bash
python cracks/detection.py --help
```

Generates:
- `crack_mask.png` (binary image) - {0, 1} for each pixel, stored as PNG
- user outputs with `[user]` prefix:
  - `[user]colored.png` - black are crack seed, red are crack flooded areas, gray and green are other pixels (gray are darker, green lighter)
  - `[user]crack-mask.png` - the same as crack mask but values are {0, 255}
  - `[user]input-grayscale.png` - input converted to grayscale image values in range \[0,255\]
  - `[user]input-normalized.png` - input with applied normalization. Normalization stretches values into {0, ..., 255} if they are not and center histogram around specified value. Each side of the histogram is stretched different way.

## Phase maps

Phase map is a special output from an electron microscope. Electron microscope produces a spectrum for each pixel. But spectra are very hard to visualize due to high number of dimensions. The spectra correspond to elements and elements can be encoded by colors. Product of this visualization technique we call **phase map** and it is fulfilled by phase definition (@see [config/phases_config.json]).

Phase maps are noisy for direct usage and contain segments with unclear phase. For this reason we do segmentation of phase map first with some morphology for fixing small segments. Segmentation is based on La*b* color distance of reference phase colors [@see phase config](config/phase_config.json) and color in the phase map (according to the closest phase distance the phase association is done).

This functionality is in `phase_map/__init__.py`. For usage see:
```bash
python -m phase_map --help
```

Generates:
- user outputs
  - `[user]grain_map_filtered.tiff` - 16bit int grayscale image where each grain has own id. Grains are filtered in this image, grains smaller than `grain_size_px_limit` pixels are dropped
  - `[user]grain_map.tiff` - as previous but before applying `grain_size_px_limit`
  - `[user]matrix_grains.png` - Binary image where 1 denotes matrix phase (after grain size filtration), 0 denotes non-matrix phase. Matrix denotes phase which is mixed from other phases.
  - `[user]non_matrix_grains.png` - Negative of the previous
  - `[user]phase_map-${PHASE["label"]}.png` - Binary image where 1 correspond with present of the `${PHASE}`
  - `[user]phase_map.png` - Visualization of the phase map after cleaning, i.e. segmented phase map.
- `grain_map_filtered.tiff` - 16bit int image where each grain has own ID after application of `grain_size_px_limit`
- `grain_map.tiff` - 16bit int image where each grain has own ID (no size filter is applied)
- `phase_img_lab.png` - phase map converted to La*b* color space (png should be interpreted as La*b* not RGB)
- `phase_map_adjusted.png` - if input phase map is lightness-adjusted this output is generated. This means that the lightness of the input is stretched to maximum.
- `phase_map_filtered.png` - grayscale image where each pixel is attached to one of the phases. New phase **matrix** is introduced with the highest ID. **Matrix** is a mixture of other phases with too small grains. However, the `grain_size_px_limit` is applied and all segments under limit are attached to **matrix** phase.
- `phase_map_gaussian_blur.png` - for handling nosiy phase edges the blur can be applied (@see parameter `--gaussian_blur`), this image is produces for checking how blurred image looks like
- `phase_map_rgb` - the input phase map image
- `phase_map` - same as `phase_map_filtered.png` but without applying `grain_size_px_limit`

## Grains / phases cracking analysis 

Main part of the toolbox is analysis of cracks. This can be executed by `full_crack_analysis.py` script. For usage @see:
```bash
python full_crack_analysis.py --help
```
There are two possible flags implemented {crack_features, phase_features}. The first one (crack_features), is focused only on cracks itself (their length, max width, average width, distance between endpoints and length of their boundary). Second part (phase_features) puts cracks into context of phase map and measure wheather crack is between two phases or through some phase. This way we can quantify weak compounds of the scanned material (i.e. phases which crack more often or interface between two phases which is week.)

Generates:
- `cracks_features.csv` - statistics about the cracks containing:
  - label - i.e. crack ID which is used in `crack_mask.png`
  - crackSize_px - number of pixels belonging to the crack
  - skeleton_px - number of pixels belonging to the crack skeleton (medial axis)
  - maxWidth_px - the biggest distance from medial axis of the crack to the closest boundary point of the crack
  - avgWidth_px - average of the crack width (i.e. distance from medial axis to the boundary)
  - boundaryLength_px - crack boundary length
  - farthestPoints_px - distance of two points of the crack which have the biggest distance between them
- `phase-edge.png` - A stacked bar chart showing number of crack's skeleton pixels between each pair of phases
- `phase-through.png` - A bar chart showing number of crack's skeleton pixels going through a single phase
- `phases_cracks.csv` - raw data for the previous two charts (numbers of cracks skeletons pixels between each pair of phases as well as going through single phase).
- `skeleton-neighbors.csv` - full overview of the analysis. Each row of this table contains:
  - crack skeleton coord X - x position of evaluated skeleton pixel
  - crack skeleton coord Y - y position of evaluated skeleton pixel
  - crack ID - the ID of the crack used in `crack_mask.png`
  - branch ID - the ID of the crack branch (each crack is split into branches - a single pixel wide path)
  - gradient X - $\delta x$ in the skeleton point $[X, Y]$
  - gradient Y - $\delta y$ in the skeleton point $[X, Y]$
  - left neighbor X - x-coordinate of the first non-crack pixel in the direction perpendicular (clockwise) to gradient for the point $[X,Y]$
  - left neighbor Y - y-coordinate of the first non-crack pixel in the direction perpendicular (clockwise) to gradient for the point $[X,Y]$
  - right neighbor X - x-coordinate of the first non-crack pixel in the direction perpendicular (counter-clockwise) to gradient for the point $[X,Y]$
  - right neighbor Y - y-coordinate of the first non-crack pixel in the direction perpendicular (counter-clockwise) to gradient for the point $[X,Y]$
  - left neighbor phase ID - phase belonging to the left neighbor
  - right neighbor phase ID - phase belonging to the right neighbor
