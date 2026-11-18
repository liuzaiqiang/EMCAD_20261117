# EMCAD + EFB implementation note

## Sources and scope

- Yue et al., **CD-PolypNet**, MICCAI 2025, Section 2.3, Equations (7)–(10): https://papers.miccai.org/miccai-2025/paper/2291_paper.pdf
- Authors' released implementation: https://github.com/ChangpengYue/CD-PolypNet/tree/main/train/efb_net
- Dai et al., **Attentional Feature Fusion**, WACV 2021 (two-stage iAFF): https://openaccess.thecvf.com/content/WACV2021/html/Dai_Attentional_Feature_Fusion_WACV_2021_paper.html

This is an **EMCAD adaptation of the published EFB**, not a verbatim reproduction of CD-PolypNet. The published branch consumes SAM features and is tied to that architecture's channel count and resolution. Here `lib/edge_guided_feature.py` receives EMCAD's shallowest decoded `d1` feature before the final prediction head, plus the original input image. The paper's Canny prior, fixed 7×7 Gaussian low-pass with sigma 3, high-frequency residual, learned edge gate, 3×3 semantic convolution, and two-stage iAFF are retained. The fixed Canny and Gaussian operations have no learned parameters.

For grayscale CT/MRI, Canny uses the input slice directly; for RGB images, it uses the channel mean. Per-image min–max conversion to uint8 makes Canny usable for the different image intensity ranges in this project. The Canny map is resized to `d1` resolution using nearest-neighbor interpolation. This normalization and resize are integration choices, not a claim that the original SAM code handles all 12 datasets. There is no extra boundary loss or supervision in this adaptation.

The switch `EFB_ENABLED=0|1` in each training launcher maps to `--efb_enabled`. `0` is the unchanged EMCAD network path; `1` registers and executes EFB. Existing small-target hard-region supervision remains independently switchable. Training saves `efb_enabled` in its config; testing restores it from the training config, with absent legacy fields interpreted as `0`. Synapse's topN candidate selection and Dice/HD95 history are unchanged.

## Complexity reporting

`lib/model_complexity.py` calls the existing EMCAD `utils/utils.py::CalParams(model, input_tensor)` once before training. The reported **FLOPs** and **Params** are exactly that method's THOP `profile` and `clever_format` convention for a batch of one and the dataset's configured image size and channel count. The file also records the exact direct count of registered parameters for auditing. THOP leaves temporary counting buffers on modules; these are removed after reporting so checkpoints are unaffected. THOP's estimate excludes OpenCV Canny and some tensor-level functional operations, so it is an estimate rather than exact end-to-end EFB cost. Use the same input resolution, channel count, THOP version, and method for paper comparisons. The authors' reported gain on their SAM-based polyp model is not evidence of a gain on EMCAD or on every dataset; that requires experiments.
