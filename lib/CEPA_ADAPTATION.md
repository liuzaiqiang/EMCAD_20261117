# CEPA adaptation in EMCAD

Source: Wu et al., *MCEPANet: a connectivity-edge guided attention network for
robust medical image segmentation with multi-scale boundary preservation*,
Biomedical Physics & Engineering Express 12 (2026) 045020,
DOI: 10.1088/2057-1976/ae7c06.

The publicly accessible abstract and module figure describe a connectivity
path, a Sobel edge path, channel attention, and multi-scale insertion. The
full paper equations and author code were unavailable for verification. This
is therefore an **EMCAD adaptation of that published module concept**, not
an exact reproduction of the paper's CEPA or its reported measurements.

At each of the three skip fusion stages, EMCAD first upsamples through EUCB
(including CAA when enabled) and filters the encoder skip through LGAG. CEPA
then takes that skip and the same-resolution decoder feature. Its connectivity
cue averages cosine similarity to the eight neighboring pixels in a learned
projection of the skip feature. Its edge cue applies fixed Sobel kernels to
the decoder feature's channel mean. Separate projected feature branches are
weighted by these cues, fused with channel attention, and mapped to a
spatial/channel gate. A bounded residual scale is initialized to zero, so the
new module initially leaves the skip unchanged. No ground-truth mask or new
loss is used by CEPA; it runs during both training and inference.

`CEPA_ENABLED=0|1` is present in every training launcher and defaults to 1.
The Python CLI default is 0 for legacy checkpoints. The value is saved in
`config.json` and restored when testing the corresponding checkpoint. A
checkpoint trained with CEPA must be tested with CEPA enabled; the test
scripts reconstruct it from the saved config. Switching off CEPA constructs
no CEPA parameters and follows the previous EMCAD path.

The three model paths cover Synapse, ACDC, and the shared binary trainer for
ClinicDB, Kvasir, ColonDB, ETIS, BKAI, ISIC2017, ISIC2018, BUSI, DSB18, and
EM. Existing area-adaptive Tversky loss and prediction/metric code are
unchanged. Whether this adaptation improves Dice or HD95 must be measured
using matched-seed ablations; the published MCEPANet scores are not EMCAD
results.
