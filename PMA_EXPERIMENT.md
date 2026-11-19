# EMCAD + LB-UNet PMA experiment

The training launchers accept `PMA_ENABLED=0|1` and
`SMALL_TARGET_HARD_SUPERVISION=0|1`.  Both zero leave the EMCAD model parameter
keys, decoder computation, four output logits, and original segmentation loss
unchanged.  The flags are recorded in launcher logs and in `config.json`.

PMA uses the RBP residual prediction and PIF skip fusion of Xu and Tong,
*LB-UNet*, MICCAI 2024, equations (1) to (3).  EMCAD has four decoder stages:
region prediction applies to all four, and boundary prediction to the three
stages with skips.  Paper weights are mapped deep to shallow as region
`0.2,0.3,0.4,0.5` and boundary `0.1,0.2,0.3`.  Auxiliary BCE/Dice terms use
the paper's `0.5 * BCE + Dice` with the same stage weights; EMCAD's original
segmentation loss remains the primary loss.  The encoder, GSA, and main
segmentation loss of LB-UNet are not transplanted.

**Boundary target adaptation:** The source GBG is an offline genetic search for
six key points on a single 256 × 256 skin lesion.  Its released implementation
assumes one contour and fixed image size.  This project instead creates a
two-sided class-transition boundary from each augmented training mask.  This
supports arbitrary image sizes, multiple disconnected organs, and shared
inter-class boundaries without unaligned offline labels.  It is *not* the
paper's GA key-point target, so results must be reported as an EMCAD PMA
adaptation, not an exact LB-UNet reproduction.

For an ablation, use the same data split, seed, encoder weights, image size,
threshold, and TopN settings in all four flag combinations.  Compare paired
per-case Dice and HD95 distributions, including cases with empty or tiny masks.
LB-UNet reported Dice and IoU on ISIC 2017/2018; it did not report HD95, and
its reported gain cannot establish an HD95 improvement for these 12 datasets.
