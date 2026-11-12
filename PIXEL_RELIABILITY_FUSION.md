# EMCAD with pixel reliability fusion

This branch restores the original EMCAD EUCB upsampling and adds a selectable
four-head readout based on the repository's `try5` branch. The fusion design is
an adaptation of that prior local implementation, not a claim of a new
independent invention. The single-logit binary correctness target now uses a
sigmoid threshold; applying `argmax` to a single channel would always predict
class zero.

`USE_PIXEL_RELIABILITY_FUSION=1` enables fusion in the six training launchers;
`=0` uses the original P1 output and creates no fusion parameters. Synapse and
ACDC use multiclass supervision. BKAI, ClinicDB, ColonDB, ETIS, Kvasir,
ISIC2017, ISIC2018, BUSI, DSB18, and EM use the shared binary training path.
Cell forwards the switch to the Polyp launcher. The effective switch and loss
weights appear in `PARAM_NAMES` logs and training `config.json`; tests restore
the architecture switch from the selected checkpoint's configuration.

The enabled readout predicts four per-pixel reliability scores from the four
full-resolution logits. Softmax across heads yields weights that sum to one;
their weighted logits form the final prediction. Training retains the original
deep supervision loss and adds a fused segmentation loss plus a BCE term for
per-head pixel correctness. Inference uses only predictions and learned
reliability heads, with no mask or validation statistic.

For ablation, compare switch 0 and 1 with the same data split, initial encoder
weights, augmentations, optimizer, supervision, schedule, checkpoint rule, and
prespecified seeds. Report per-dataset and per-class results, variation across
seeds, parameter count, inference time, and memory. Use validation data for
model selection and keep test data sealed. In the current Synapse trainer,
`test_vol` is used for best-checkpoint selection; verify that this list is a
validation split before treating a separate test evaluation as unbiased.
