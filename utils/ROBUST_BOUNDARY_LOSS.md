# Robust signed-distance boundary loss experiment

This implementation follows the publicly described mechanism of Li et al.,
*A Robust Boundary Loss for High Accuracy Medical Image Segmentation*, IJCNN
2026, DOI: 10.1109/IJCNN69854.2026.11728433: signed distance maps,
percentile truncation, and normalization. The publicly accessible abstract
does not specify the full formula or hyperparameters, so this is an explicit
experimental implementation, not a claim of exact paper reproduction.

For each training image and each foreground class present in that image, let
`d_in` and `d_out` be Euclidean distance transforms of the ground-truth mask
and its complement. Each is divided by its own chosen percentile of nonzero
distances and clipped at 1. The signed map `phi` is negative inside the
foreground and positive outside, with range `[-1, 1]`. Empty and full-image
foreground masks have no two-sided boundary and are omitted from this term.

The auxiliary loss is the mean of `(p - y) * phi` across pixels, then the mean
across valid image/class pairs. `p` is sigmoid foreground probability for a
one-logit model or softmax foreground probability for a multiclass model;
`y` is the corresponding ground-truth indicator. Subtracting `y * phi` is
constant with respect to model parameters and gives the same gradient as the
signed-distance integral in Kervadec et al., *Boundary loss for highly
unbalanced segmentation*, MIDL 2019.

The loss is applied only to the final segmentation output and added to the
existing region losses and optional area-adaptive Tversky loss. It does not
change model architecture, inference, or metric definitions. The training
shell scripts expose `USE_ROBUST_BOUNDARY_LOSS=0|1` (default `1`),
`ROBUST_BOUNDARY_WEIGHT` (default `0.1`), and
`ROBUST_BOUNDARY_PERCENTILE` (default `95`). Direct Python CLI defaults to
off. The weight and percentile are experiment choices, **not** reported
paper settings. The parsed values are saved in each run's `config.json`.

Distance transforms use 2D pixel units on the augmented training masks; no
physical voxel spacing is inferred. GPU predictions are differentiable; the
target-only distance transforms run on CPU and add training time. Claims of
better HD95 require matched-seed ablations and the existing full-volume or
full-image test protocols.
