# `batch_run` Workflow

## Inputs
For full functionality we need:
- `E10-phases_x6.mat` - which contains phases description, i.e.:
  - `phase_color` - RGB coordinates corresponding to a phase in phase image
  - `phase_label` - textual description of this phase
- EDX layered images - which are stored in sample subdir (aka `phase_img`)
- JBmasks - crack mask created from a gray scan
- JBskeletons - medial axes for cracks in a form of mask

## Steps

- crop of the inputs according to `CROP_RANGE` - TODO: create cli parameter for this
- phase map
  - cleanup `phase_thr` function
  - from cleaned phase map create grains `refine_grains`, `build_grains_metadata`
  - filter grains by `grain_filter` smaller than `SIZE_LIMIT` - TODO: create cli parameter for this
- crack analysis
  - make cracks more smooth `crack_id` - crack dilation and then medial axis
  - split cracks to branches
  - evaluation of branches 
    - for each pixel find normal and in the direction of the normal find out first non-crack pixel, save its `phase_id`
    - for each pair of `phase_id` distinguish *passing thru grain* and *going on edge*
- join branches and create statistics for each crack-branch-phase

# Postprocessing - evaluation statistics