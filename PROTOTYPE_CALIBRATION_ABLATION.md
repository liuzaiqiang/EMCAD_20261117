# Cross-scale prototype calibration ablation

`USE_CONTENT_AWARE_ANTIALIAS` and `USE_CROSS_SCALE_PROTOTYPES` are independent
binary training-launcher switches. They are recorded in the launcher log; the
model configuration records `caa_mode` and `use_cross_scale_prototypes`.

| CAA | Prototypes | Architecture |
| --- | --- | --- |
| 0 | 0 | Original EMCAD decoder and heads |
| 1 | 0 | EMCAD with content-aware anti-aliasing upsampling |
| 0 | 1 | EMCAD with cross-scale prototype calibration |
| 1 | 1 | Both modules |

For example, use `USE_CONTENT_AWARE_ANTIALIAS=0 USE_CROSS_SCALE_PROTOTYPES=1`
before `bash sh/start_train_synapse.sh`. The same variables work with ACDC,
ISIC, BUSI, Polyp, and Cell launchers. Cell passes them to the Polyp launcher.
The ISIC, Polyp, and Cell launchers select their subdataset with `DATASET_NAME`.

The calibration block projects each decoder scale to a common feature space.
Per-image class probabilities from the native segmentation heads weight each
scale's class centroid. A target scale receives the valid centroids from the
other three scales; cosine similarity supplies a bounded, learned residual to
its logits. Binary segmentation forms internal background and foreground
probabilities from its single foreground logit. The block uses no ground-truth
mask, validation statistic, or cross-image information at inference. Its gain
starts at zero, so enabling it initially leaves the original logits unchanged.

For an honest comparison, keep data splits, encoder weights, augmentation,
loss, optimizer, schedule, validation frequency, checkpoint selection, and
seeds fixed across the four rows. Run multiple prespecified seeds and report
mean and spread for each dataset, along with parameter count, FLOPs, latency,
and memory. Select hyperparameters on validation data only. Keep test data
sealed until the protocol is fixed. A positive metric change is an empirical
question; this implementation does not establish one by itself.
