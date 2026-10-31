# Progress Log

## Session: Pix2Pix data pipeline + baseline metric correction

### Context
Starting implementation of the Pix2Pix GAN for mask-conditioned synthetic
defect generation. Before writing the generator/discriminator, verified the
assumptions behind the planned oversampling strategy and the reported
baseline segmentation numbers - both turned out to need revision.

---

### Finding 1: class_3 is not a rare class

Original assumption (from earlier work): class_3 is rare and its weak
baseline IoU (0.61, macro-imagewise) reflects a data-scarcity problem that
synthetic augmentation should address.

`scripts/analyze_class_distribution.py` on the raw Severstal `train.csv`
shows the opposite:

| class | % of defect images | % of total defect pixel area | avg instance area |
|---|---|---|---|
| class_1 | 13.5% | 2.4% | 1.06% of a frame |
| class_2 | **3.7%** | **0.5%** | 0.82% of a frame |
| class_3 | **77.3%** | **80.3%** | 6.22% of a frame |
| class_4 | 12.0% | 16.8% | 8.39% of a frame |

class_3 is the dominant class by both image presence and pixel area.
class_2 is the genuinely rare class on both axes.

---

### Finding 2: baseline per-class IoU was inflated for rare classes

`src/segmentation/evaluate.py` originally computed per-class IoU with
`smp.metrics.iou_score(..., reduction="none")` per batch, then averaged
those per-image ratios across images ("average of ratios").

`smp.metrics.iou_score` defaults to `zero_division=1`: any image that
doesn't contain a given class (and isn't falsely predicted as that class)
scores a trivial IoU=1.0 for it. For rare classes (class_1: present in
13.5% of images, class_2: 3.7%), the vast majority of images contribute
this trivial 1.0, massively inflating the reported per-class average.

Fixed by accumulating `tp/fp/fn/tn` across the entire test set first, then
computing IoU once from the totals ("sum then divide" - standard
dataset-level per-class IoU, equivalent to reading IoU off the diagonal of
a full confusion matrix). Cross-checked independently against
`src/segmentation/analyze_errors.py` (built from a manual confusion
matrix) - both methods agree.

**Corrected baseline (U-Net, ResNet34 encoder) - this is now the official
reported number:**

| class | corrected IoU | legacy macro-imagewise IoU (inflated) |
|---|---|---|
| background | 0.9724 | 0.9697 |
| class_1 | **0.4329** | 0.9094 |
| class_2 | **0.4887** | 0.9615 |
| class_3 | 0.6283 | 0.6103 |
| class_4 | 0.6454 | 0.9455 |
| **mIoU** | **0.6336** | 0.8793 |

`src/segmentation/train.py`'s `compute_miou()` (used for checkpoint
selection / `ReduceLROnPlateau` during training) was left on
macro-imagewise deliberately - it's an internal, single-run training
signal, not a cross-experiment comparison metric. All final reported
numbers (baseline vs. every GAN model x real:synthetic ratio) must come
from the corrected `evaluate.py` for the comparison to be valid.

---

### Revised thesis narrative

- class_1 and class_2 are the genuinely weak classes (both rare *and*
  poorly segmented).
- class_1 is notable on its own: ~3.6x more data than class_2 but a worse
  corrected IoU (0.43 vs 0.49) - suggests its weakness isn't purely a data
  scarcity problem (visual ambiguity / high intra-class variability is a
  plausible factor, not yet investigated further).
- class_3 and class_4, once measured honestly, are comparable to or better
  than class_1/class_2 - not the weak link.

### Decision: oversampling target

Chose **class_2 only** as the oversampling target for the Pix2Pix (and
later SPADE / StyleGAN2-ADA) training data. Rationale: cleanest,
most defensible "data scarcity -> synthetic augmentation helps" story,
lowest interpretation risk given remaining project scope (3 GAN
architectures x 3 real:synthetic ratios). class_1's weakness is
noted as a discussion point / possible limitation, not pursued as a
separate GAN-targeted experiment for now.

---

### Artifacts produced this session

- `src/gans/pix2pix/dataset.py` - `Pix2PixSteelDataset`, multi-channel
  (4-channel) mask conditioning from the 0-4 label-map PNGs, native
  1600x256 resolution (no crop/resize, matches segmentation baseline),
  `get_class3_image_ids()` / `build_class3_weighted_sampler()` helpers
  (**to be renamed/generalized to class_2 - see Next Steps**).
- `scripts/analyze_class_distribution.py` - image-level presence vs
  pixel-level area per class, from raw `train.csv`.
- `src/segmentation/analyze_errors.py` - confusion matrix, per-class
  precision/recall/F1, error breakdown by target class.
- `src/segmentation/evaluate.py` - corrected per-class IoU computation
  (dataset-level accumulation instead of per-image averaging).

### Architecture decisions locked in for Pix2Pix

- Generator: U-Net, 6 downsampling blocks (not the standard 8 - 1600x256
  isn't a power-of-two square; 6 blocks keeps bottleneck at 4x25 without
  collapsing the height dimension).
- Discriminator: PatchGAN, depth matched to the 6-block generator.
- Loss: adversarial (BCE) + L1, lambda=100 (standard Pix2Pix).
- Mask conditioning: multi-channel one-hot (4 channels, classes 1-4),
  weighted oversampling for class_2 (target fraction TBD - was 0.25 for
  the class_3 version, needs reconsidering given class_2's natural
  frequency is only 3.7% vs class_3's 77.3%).
- Native resolution 1600x256, no crop/resize, to match the segmentation
  baseline's evaluation format.

### Update: Pix2Pix pipeline complete, 25-epoch smoke test healthy

`dataset.py` generalized to `get_class_image_ids()` /
`build_class_weighted_sampler()` (configurable class_id, defaults to
class_2). class_2 oversampling `target_fraction` set to 0.15 (not 0.25 -
class_2 has only ~247 unique images dataset-wide / ~160 in train split;
an aggressive target risked the GAN memorizing that small pool instead of
generalizing, more of a concern for a generative model than it would be
for segmentation-style class balancing).

Implemented:
- `generator.py` - U-Net, 6 downs (bottleneck 4x25 for native 1600x256
  input), InstanceNorm (small batch size, 2-4, makes BatchNorm noisy),
  Dropout(0.5) near the bottleneck for stochasticity. 29.24M params.
- `discriminator.py` - standard 70x70 PatchGAN, 3 layers, conditioned on
  concatenated (mask, image), 7 input channels. 2.77M params. Left at
  standard depth - fully convolutional, doesn't need to match generator
  depth.
- `train.py` - adversarial (BCEWithLogits) + L1 (lambda=100), Adam
  lr=2e-4 betas=(0.5, 0.999) for both G/D, separate AMP GradScalers,
  MLflow logging, periodic visual sample grids (mask/fake/real).

25-epoch run on the class_2-weighted train loader: `g_loss` 29.3->25.3,
`l1` 27.1->22.6, both trending down smoothly; `d_loss` stable ~0.29-0.34
(no sign of D overpowering G or vice versa). Visual samples show the
generator starting to respond to mask shape - large blob-shaped defects
(class_3/4-like) develop plausible speckled texture matching real images
by epoch 20-25; thin line defects (1-2px wide) still barely visible,
expected at this stage (less gradient signal per pixel early in
training).

**Known issue, not yet fixed:** visible checkerboard/grid artifact in the
background texture across all samples - classic `ConvTranspose2d`
(kernel=4, stride=2) artifact from uneven upsampling overlap. Standard
fix is replacing `ConvTranspose2d` in the decoder with
`nn.Upsample(mode="nearest") + Conv2d`. Not blocking further training,
but should be addressed before FID comparison against SPADE/StyleGAN2-ADA
- the artifact could distort FID scores independent of actual sample
quality.

### Next steps

1. Full Pix2Pix training run (100+ epochs) - the 25-epoch smoke test
   trend is healthy enough to proceed.
2. Consider fixing the `ConvTranspose2d` checkerboard artifact before
   final FID comparisons (see above).
3. Implement FID computation script (needs InceptionV3, real vs.
   generated sample sets).
4. Move to SPADE, then StyleGAN2-ADA.
5. Real:synthetic ratio experiments (75:25 / 50:50 / 25:75) evaluated via
   the corrected `evaluate.py` only.


## Session: Checkerboard artifact fixes - three attempts, original architecture wins

### Context

Carried over from the previous session: the 25-epoch Pix2Pix smoke test
(generator.py, U-Net, 6 downs, InstanceNorm, 29.24M params - see earlier
entry "Update: Pix2Pix pipeline complete, 25-epoch smoke test healthy")
had healthy loss dynamics but a visible checkerboard/grid artifact in the
background texture, attributed to `ConvTranspose2d(kernel=4, stride=2)`
uneven upsampling overlap. This session tried three fixes before running
the full 100-epoch training, evaluated by short smoke tests (10-30
epochs) and visual comparison of sample grids across epochs.

---

### Attempt 1: `Upsample(nearest) + Conv2d(kernel=3)`

Replaced all three `ConvTranspose2d` upsampling layers (outermost,
innermost, middle blocks) in `generator.py` with
`nn.Upsample(scale_factor=2, mode="nearest")` followed by
`nn.Conv2d(kernel_size=3, stride=1, padding=1)`.

Result: checkerboard pattern eliminated, but generator params dropped
29.24M -> 21.32M (smaller kernel = fewer weights per up-block). On a
full 100-epoch run, output quality visibly *worsened* over training -
background texture became increasingly blurry/blocky, with mosaic-like
artifacts near defect regions by epoch 100. Quantitative losses looked
healthy (`l1` dropped steadily to ~14.1, `d_loss` stable ~0.17-0.18) -
the loss/visual-quality mismatch indicates the network was satisfying L1
via blur/local averaging rather than learning real texture, likely due
to reduced capacity from the smaller conv kernel.

**Rejected.**

### Attempt 2: `ConvTranspose2d(kernel=2, stride=2, padding=0)`

Hypothesis: kernel=stride means no overlap between upsampling kernels,
which is the textbook cause of checkerboard artifacts - should fix the
artifact while staying closer to the original param count than Attempt 1.

Wrong on the param count: kernel=2 means 4 weights per channel pair
(2x2) vs. kernel=4's 16 weights per channel pair (4x4) - actually *more*
aggressive parameter reduction than Attempt 1. Params dropped to 15.67M.

**Rejected immediately without a training run**, given the capacity
concern already observed in Attempt 1.

### Attempt 3: ICNR initialization, original `ConvTranspose2d(kernel=4)` restored

ICNR (Aitken et al., 2017): initializes ConvTranspose2d sub-kernels
identically at the start of training (rather than independently at
random), targeting the root cause of checkerboard artifacts without
changing kernel size, stride, or param count (29.24M preserved).

30-epoch smoke test: looked promising - `d_loss` stable ~0.27-0.37, no
checkerboard visible, background texture sharp (not blurred like Attempt
1).

Full 100-epoch run told a different story. Sampled epochs 1/25/50/75/100:
- Epoch 1: near-clean background.
- Epoch 25: waffle/grid pattern clearly emerged.
- Epochs 50-75: pattern held roughly steady.
- Epoch 100: pattern intensified further, ending up *more* prominent
  than the original un-fixed checkerboard from the first 25-epoch smoke
  test.

Conclusion: ICNR only fixes the *initialization* of ConvTranspose2d
sub-kernels; nothing prevents them from diverging again during training
via gradient updates. On a short run the fix looks like it's working; on
the full 100-epoch run the pattern re-emerges and grows, making ICNR a
delay rather than a solution - same failure mode observed with the
original architecture's artifact in Attempt 1's rejected fix path, just
on a longer timescale.

**Rejected.**

---

### Final decision: revert to original architecture, accept the artifact

`generator.py` reverted fully to `ConvTranspose2d(kernel=4, stride=2,
padding=1)` in all three upsampling blocks, standard `normal_(0, 0.02)`
init for all Conv/ConvTranspose/Norm layers (ICNR init and `icnr_init()`
helper removed). 29.24M params confirmed.

Full 100-epoch run on this architecture (epochs 1/25/50/75/100 sampled):
mild checkerboard artifact present but stable, does not intensify over
training the way ICNR's pattern did. Large blob-shaped defects develop
plausible speckled texture by epoch 100, comparable to what was seen in
the original 25-epoch smoke test. Result accepted as satisfactory for
the thesis.

**Accepted as final Pix2Pix generator architecture.** The checkerboard
artifact is a known, documented limitation - to be discussed explicitly
when comparing FID scores against SPADE/StyleGAN2-ADA, since it could
inflate Pix2Pix's FID independent of actual sample quality. Worth a
one-line caveat in the thesis discussion section.

### Key learning

For `ConvTranspose2d` checkerboard artifacts: capacity-preserving fixes
(ICNR) only delay the artifact's emergence, they don't prevent it from
re-emerging under longer training with a narrow, oversampled data pool
(~160 class_2 images). Capacity-reducing fixes (`Upsample+Conv`,
smaller-kernel `ConvTranspose2d`) trade the artifact for a different
failure mode (blur/mosaic from insufficient decoder capacity). Given the
project's remaining scope (SPADE, StyleGAN2-ADA, ratio experiments), the
mild original artifact was judged the least-bad option - fixing it
properly would likely require an architecture change beyond swapping the
upsampling op (e.g. sub-pixel conv / PixelShuffle with its own dedicated
capacity, not just an init trick on the existing layer).

### Next steps (updated)

1. FID evaluation comparing real vs. synthetic outputs (Pix2Pix baseline
   ready - checkpoint from the accepted 100-epoch run).
2. SPADE implementation, then StyleGAN2-ADA.
3. Segmentation retraining experiments with varying real:synthetic
   ratios (75:25, 50:50, 25:75), evaluated via corrected `evaluate.py`.
4. Note the checkerboard artifact as a limitation in the FID discussion
   when comparing Pix2Pix against SPADE/StyleGAN2-ADA.


### Backlog: MLflow reporting hygiene (not urgent, before thesis writing)

Raised while building compute_fid.py. Deferred - applies to train.py and
general MLflow usage, not FID specifically:

- train.py's mlflow.start_run() has no run_name/tags - runs are only
  identifiable by random hash in the UI. Same fix as compute_fid.py
  (run_name, tags: stage/generator_variant/git_commit) should be ported
  over, ideally before the SPADE/StyleGAN2-ADA runs start piling up
  alongside Pix2Pix's.
- No periodic backup of mlflow.db (SQLite, single file, single point of
  failure). Worth a cron/manual weekly copy somewhere outside
  /opt/steel-defect-data before results start feeding the thesis.
- For the thesis results chapter: don't rely on MLflow UI screenshots -
  export via mlflow.search_runs() to CSV/markdown as the actual source
  for tables/figures (a snippet for this was sketched in a compute_fid.py
  discussion, not yet written to a file - re-derive when needed).



## Session: Pix2Pix FID evaluation

### Context

Implemented FID (Frechet Inception Distance) evaluation comparing real
vs. Pix2Pix-generated steel defect images, using clean-fid's InceptionV3
feature port (mode="clean", verified ROCm-compatible before implementation
- feature shape (N, 2048) confirmed). Two scopes: "overall" (full test
split) and "class_2" (test-split subset containing the oversampling
target class, via the existing get_class_image_ids() helper).

### Artifacts produced

- `src/gans/pix2pix/compute_fid.py` - generates paired fake images from
  test-split masks (generator kept in `.train()` mode so Dropout stays
  active for stochastic sampling - `--samples-per-mask` controls how many
  fake variants per real mask), extracts InceptionV3 features via
  clean-fid, computes raw FID (`fid_from_feats`) and an attempted
  bias-corrected FID (Chong & Forsyth, 2020 - extrapolation to N->infinity).
  MLflow logging with `run_name`/tags (`stage`, `note`, `git_commit`,
  `checkpoint`) for identifiable runs (git tag needs
  `git config --global --add safe.directory /workspace` inside the
  container - dubious-ownership issue, not a code bug).

### Bugs found and fixed during implementation

1. **cleanfid.fid.frechet_distance too slow for repeated calls.** The
   bias-correction loop calls a Frechet-distance computation
   n_sizes x n_repeats times per scope (160+ by default, 320+ across both
   scopes) on the full 2048x2048 InceptionV3 covariance matrices.
   scipy.linalg.sqrtm (general Schur-based, used internally by clean-fid)
   pegged the CPU for a very long time at that call count. Fixed with a
   custom `fast_frechet_distance()` using the identity
   eig(A@B) == eig(A^0.5 @ B @ A^0.5) for symmetric PSD A, B - only needs
   `eigh`/`eigvalsh` (fast, numerically stable for singular/rank-deficient
   matrices, which our covariance matrices always are - see below), not
   the general Schur method. Cut the bias-correction step from
   "effectively hung" to ~20-45 seconds per scope. Only used inside the
   bias-correction loop - the one-off raw FID computation still uses
   clean-fid's own `fid_from_feats`/`frechet_distance`, since it's called
   only once per scope and isn't the bottleneck.

2. **Fake pool discarded during bias correction.** `generate_fakes()`
   intentionally produces more fake images than real ones
   (`--samples-per-mask > 1`, since Dropout-based stochastic generation is
   cheap and real steel defect photos are the actual scarce resource -
   e.g. overall scope: 1000 real vs 5000 fake). The first version of
   `bias_corrected_fid()` subsampled *both* sides down to a shared
   N = min(n_real, n_fake) at every grid point, discarding most of the
   fake-side sample-size advantage that `raw_fid` (computed on the full
   asymmetric 1000 vs 5000) actually benefits from. This produced
   bias-corrected FID *higher* than raw FID for the overall scope (87-89
   vs raw ~80) - the wrong direction, since more effective sample size
   should only reduce estimated bias, never increase it. Fixed by holding
   the fake side fixed at its full available pool (mu_f/sigma_f computed
   once, reused across every real-subsample size/repeat) and only varying
   n_real in the extrapolation grid - real images are the actual
   scarcity bottleneck for this thesis, not synthetic ones.

3. **Subsample-size grid spaced wrong for the regression.** The
   extrapolation regresses FID(n) against 1/n, but the size grid was
   generated with `np.linspace(min_size, max_size)` - i.e. evenly spaced
   in n, not in 1/n. This concentrates almost the entire useful 1/n range
   into a single small-n point with disproportionate leverage on the
   fitted line, distorting the intercept. Fixed by spacing the grid evenly
   in 1/n instead (`np.linspace(1/max_real, 1/min_size, num=n_sizes)`,
   inverted back to integer sizes) - matches the actual Chong & Forsyth
   methodology.

### Finding: bias-corrected FID is unreliable at our sample sizes -
### methodological limitation, not a further bug

After fixing all three issues above, the diagnostic FID(n_real) curve is
clean and monotonically decreasing (confirmed by printing every grid
point + raw_fid for comparison - the largest-N point matches raw_fid
almost exactly, as expected, since at n_real = full real count the
subsample equals the full set and the fake side is already fixed at its
full pool too). However, the *slope* between consecutive points is
markedly non-constant and non-monotonic even after fixing the grid
spacing (e.g. overall scope: pairwise slopes ~1233 -> 881 -> 834 -> 707
-> 1676 -> 2134 -> 3781 across the 8-point grid) - i.e. FID(n_real) is not
actually linear in 1/n_real over this range. Two different grid designs
(evenly-spaced-in-n vs evenly-spaced-in-1/n) produced two different,
non-converging bias-corrected estimates (83.9 vs 92.8 for the overall
scope) rather than agreeing - if this were a grid-design artifact alone,
both should converge toward the same extrapolated value.

Likely cause: n_real never exceeds 1000 (overall) or 49 (class_2), both
far below 2048 - the InceptionV3 feature dimensionality - so the
2048x2048 covariance matrix is *always* rank-deficient (rank <= n-1)
at every point on the grid, not just at the small-n end. The Chong &
Forsyth linear-in-1/N bias model is empirically validated in regimes
where N spans well above the feature dimensionality; it has no guarantee
of holding when N stays below D for the entire tested range, which is
our situation for both scopes.

**Decision: raw FID is the metric reported for the thesis, not the
bias-corrected extrapolation.** This matches standard practice in the
GAN literature (raw FID is reported even at small N in most papers). The
FID(n_real) curve is kept as a *descriptive* diagnostic (illustrates the
direction/magnitude of small-sample inflation, especially stark for
class_2: 183 at n=20 vs 170 at n=49) rather than a corrective one. To be
stated explicitly in the thesis discussion/limitations section: N is
always below the InceptionV3 feature dimensionality (2048) for both
scopes, so Chong & Forsyth-style bias correction was attempted but found
unreliable in this regime, rather than silently omitted.

### Results (Pix2Pix baseline, 100-epoch accepted-architecture checkpoint)

| scope | raw FID | n_real | n_fake |
|---|---|---|---|
| overall | 80.390 | 1000 | 5000 |
| class_2 | 170.486 | 49 | 245 |

class_2's much worse FID than overall is a substantive, expected result
(mirrors the segmentation baseline's weak class_2 IoU) - consistent with
the small training pool (~160 images) and the accepted checkerboard
artifact, which likely distorts fine/thin-defect texture (class_2's
typical presentation) more than large blob-shaped defects (class_3/4-
like). Worth cross-referencing with the checkerboard-artifact discussion
already in progress_log.md when writing the thesis discussion section.

### Next steps

1. SPADE implementation, then StyleGAN2-ADA - run the *same*
   compute_fid.py (same test split, same --samples-per-mask) against each
   for a like-for-like raw FID comparison across all three architectures.
2. Segmentation ratio experiments (75:25 / 50:50 / 25:75), evaluated via
   the corrected (sum-then-divide) evaluate.py only - unrelated to this
   session's FID bias-correction findings, no changes needed there.
3. When writing the thesis: explicitly document the N < 2048 limitation
   for FID bias correction (see "Finding" above) alongside the checkerboard
   artifact as a joint discussion of what could distort the Pix2Pix FID
   numbers independent of actual sample quality.

## Session: SPADE implementation, discriminator-balance tuning, full training, and FID evaluation

### Context

Following the architecture decisions made when starting this stage (reuse
`src/gans/pix2pix/dataset.py`/`build_class_weighted_sampler()` unchanged,
same train/val/test splits, same `compute_fid.py` pattern), implemented
SPADE (Park et al., 2019) as the second of three GAN architectures for
this thesis, deliberately staying close to the original paper and
deviating only where forced (VRAM, network access, or an empirically
diagnosed training-stability problem) - same philosophy used for Pix2Pix.

### Architecture decisions

- `spade_norm.py` - SPADE normalization (spatially-adaptive gamma/beta
  predicted from the mask via two small conv branches), InstanceNorm2d
  as the parameter-free base rather than the paper's default BatchNorm2d
  - same VRAM/batch-size=2-4 rationale already established for Pix2Pix's
    generator (the paper itself names this configuration, `spadeinstance3x3`,
    so it's a supported variant, not an ad hoc change).
- `spade_resblock.py` - SPADEResnetBlock, spectral norm on every conv
  (paper default), weight init (`normal_(0, 0.02)`) applied *before*
  wrapping in `torch.nn.utils.parametrizations.spectral_norm`, since that
  API turns `.weight` into a computed parametrization - a later blanket
  `model.apply(init_weights)` would silently miss the real underlying
  weight.
- `generator.py` - SPADEGenerator, "more"-upsampling-layer configuration
  (6 `Upsample(nearest)` doublings, 7 SPADEResnetBlocks: head_0,
  G_middle_0, G_middle_1, up_0..up_3), seed resolution 4x25 - chosen
  specifically to match Pix2Pix's 6-downsampling-block bottleneck
  (1600/64=25, 256/64=4 exactly), so the two generators share the same
  effective receptive-field depth as a controlled variable. No VAE image
  encoder; stochastic sampling instead comes from `Dropout2d(0.5)` in the
  three bottleneck-adjacent blocks (head_0, G_middle_0, G_middle_1),
  deliberately mirroring Pix2Pix's "dropout near the bottleneck, model
  kept in `.train()` mode at inference" convention so `compute_fid.py`'s
  `--samples-per-mask` stochastic-variant mechanism needed no SPADE-specific
  handling.
- `discriminator.py` - MultiscaleDiscriminator, num_D=2 (full resolution +
  2x avgpool-downsampled), each scale a 4-layer PatchGAN with spectral
  norm + InstanceNorm2d, returning intermediate activations for feature
  matching. Same 7-channel (mask+image) conditioning as Pix2Pix's
  discriminator, kept as a controlled variable.
- `src/gans/common/losses.py` - hinge GAN loss (G and D), feature
  matching loss (L1 across every discriminator scale/layer, paper's
  `4/(n_layers_D+1)` per-layer weighting), VGG19 perceptual loss (5 tap
  points at relu1_1..relu5_1, paper's `[1/32,1/16,1/8,1/4,1]` schedule).
  VGG19 weights: `download.pytorch.org` isn't reachable from this
  project's network setup, so the official torchvision ImageNet checkpoint
  (`vgg19-dcbb9e9d.pth`) was downloaded manually (outside the container)
  and loaded from `/opt/steel-defect-data/pretrained/vgg19-dcbb9e9d.pth` -
  identical weights to what `torchvision.models.vgg19(pretrained=True)`
  would have fetched, purely a network-access workaround, not an
  architectural deviation.
- Generator params at `ngf=32`: 29.41M (29,407,459 exactly, confirmed via
  MLflow-logged `generator_params`) - deliberately chosen instead of the
  paper's default `ngf=64` (92.09M) specifically to land within ~1M params
  of Pix2Pix's generator (29.24M), keeping generator capacity a controlled
  variable between the two architectures rather than an uncontrolled
  side effect of a VRAM workaround. Discriminator: 5.54M params
  (`ndf=64`, `n_layers_d=4`, `num_D=2`).

### Finding 1: discriminator overpowering generator at baseline hyperparameters

A 25-epoch smoke test at Pix2Pix-equivalent optimizer settings (Adam,
lr=2e-4 for both G and D, betas=(0.5,0.999)) showed a clean, monotonic
divergence with no reversal across all 5 sampled epochs:

| epoch | g_loss | d_loss | vgg_loss |
|---|---|---|---|
| 5 | 16.4014 | 2.0256 | 0.5139 |
| 25 | 19.0410 | 1.0038 | 0.5068 |

`d_loss` fell monotonically, `g_loss` rose monotonically (hinge loss's
`g_loss=-D(fake).mean()` has no upper saturation point, unlike BCE, so it
grows without bound as D's margin widens), `vgg_loss` stayed flat - the
discriminator steadily winning, unlike Pix2Pix's stable 25-epoch dynamics
(`d_loss` ~0.29-0.34 throughout). Diagnosed as a real, non-catastrophic
but self-reinforcing imbalance worth correcting before a full run.

### Attempt 1: TTUR (asymmetric G/D learning rates) - insufficient alone

`lr_g=4e-4`, `lr_d=1e-4` (D slowed relative to G - the standard first fix
for this exact pattern). Result: barely moved the needle.

| epoch | g_loss | d_loss | vgg_loss |
|---|---|---|---|
| 5 | 16.1512 | 2.1218 | 0.5082 |
| 25 | 18.5782 | 1.0520 | 0.5002 |

Same monotonic shape, nearly identical magnitudes to the baseline run.
Honest negative result - kept the TTUR learning rates (no reason to
revert) but moved to a second, independent lever.

### Attempts 2-4: throttling discriminator update frequency - the effective lever

Added `--d-update-every N` to `train.py`: D updates only every N steps
(G still updates every step). Clean dose-response relationship across
three escalations, each halving-or-more the previous divergence:

| d_update_every | g_loss @5 | g_loss @25 | delta g_loss (5->25) | best vgg_loss seen |
|---|---|---|---|---|
| 1 (TTUR only) | 16.1512 | 18.5782 | +2.427 | ~0.500 |
| 2 | 14.4846 | 16.1556 | +1.671 | ~0.489 (epoch 3-5) |
| 4 | 13.8041 | 14.8649 | +1.061 | ~0.470 (epoch 13-15) |
| 8 | 13.7406 | 13.8601 | **+0.120** | 0.4602 (epoch 20) |

Both the divergence slope *and* the best achievable `vgg_loss` improved
with every escalation - not just a flatter version of the same problem,
genuinely better quality ceiling each time. At `d_update_every=8`,
`epoch_fm_loss` and `epoch_vgg_loss` were nearly flat/slightly-improving
for most of the 25 epochs, much closer to Pix2Pix's healthy dynamics;
`epoch_adv_loss` (the purely adversarial component) still crept up
~2x over the run, but is a small enough fraction of total `g_loss`
(dominated by `10*fm_loss + 10*vgg_loss`) that it no longer visibly hurt
perceptual quality within 25 epochs.

**Decision: `d_update_every=8` for the full run.** The jump from
`d_update_every=4` to `8` reduced the divergence by ~9x in one step -
clear diminishing-returns territory, and pushing further risks starving
D of a useful signal entirely (d_loss staying high/uninformative rather
than providing real gradient). Not re-tested beyond 8 given thesis time
budget; a reasonable stopping point given the dose-response curve shape.

### Finding 2: mid-training instability in the full 100-epoch run (invisible in any 25-epoch smoke test)

Full training run (`ngf=32`, `lr_g=4e-4`, `lr_d=1e-4`, `d_update_every=8`,
`batch_size=2`, `checkpoint_every=5`, 100 epochs, ~1.1 days runtime,
class_2 `target_fraction=0.15` unchanged from Pix2Pix) completed without
crashing, but the MLflow curves showed something none of the five
25-epoch smoke tests could have revealed, since they all stopped well
before this point:

- `epoch_d_loss` bottomed around epoch 65-70, then **reversed** and rose
  back up to ~2.3 by epoch 100 (D "catching up" after the long
  under-training imposed by `d_update_every=8`).
- `epoch_vgg_loss` and `epoch_g_loss`, which had held a flat, healthy
  plateau through epochs 20-80 (`vgg_loss` ~0.46-0.47), both spiked
  sharply around epochs 85-90 (`vgg_loss` peaking ~0.555 - above even
  the epoch-1 starting value) - a transient destabilization event,
  coinciding with D's reversal.
- Both recovered by epoch 100, but `vgg_loss` settled at ~0.48-0.49,
  not fully back to the pre-spike plateau.

Checkpoints were saved every 5 epochs specifically so this kind of
late-training event wouldn't be invisible after the fact.

### Finding 3: checkpoint selection by train-loss intuition was wrong - verified against FID instead

Adapted `compute_fid.py` for SPADE (`src/gans/spade/compute_fid.py`):
reused the Pix2Pix version's FID/bias-correction logic completely
unchanged (same `fast_frechet_distance`, fixed-fake-pool and
1/n-spaced-grid fixes - see earlier session). Two load-bearing changes
only: (1) import `SPADEGenerator`/`build_generator` from
`src.gans.spade.generator` instead of Pix2Pix's; (2) added a required
`--ngf` flag (default 32) - `SPADEGenerator`'s own default is `ngf=64`,
which does NOT match what every SPADE training run actually used, and
silently building the wrong-capacity model would fail `load_state_dict`
on a shape mismatch rather than loading something subtly wrong. Also
made `--checkpoint` required (no default, unlike Pix2Pix's
`generator_last.pth`) since there was no single "accepted" SPADE
checkpoint yet going into this comparison.

Given Finding 2, epoch 75 (right before the instability spike, closer to
the train-loss plateau) was hypothesized as possibly a better checkpoint
than epoch 100 (recovered, but not fully back to plateau). Ran
`compute_fid.py` on both:

| scope | epoch 75 (raw FID) | epoch 100 (raw FID) |
|---|---|---|
| overall (1000 real) | 97.702 | **68.125** |
| class_2 (49 real) | 190.609 | **152.108** |

**The train-loss-based hypothesis was wrong.** Epoch 100 is decisively
better on both scopes despite its train `vgg_loss` sitting above the
pre-spike plateau. Train-batch perceptual loss and held-out FID are
correlated but not interchangeable for checkpoint selection - worth
checking FID directly rather than inferring from training curves alone,
whenever that's feasible. Bias-corrected FID was computed as a diagnostic
alongside raw FID for both checkpoints but not trusted for the same
reason already established for Pix2Pix (n_real stays below the 2048
InceptionV3 feature dimensionality for both scopes; the class_2 scope's
bias-corrected estimate landed on the opposite side of raw FID from the
overall scope's, the same inconsistency signature that ruled out
bias-correction for Pix2Pix).

**Decision: epoch 100 (`generator_epoch0100.pth` / `generator_last.pth`)
is the final SPADE checkpoint** for all further comparisons.

### Result: SPADE (epoch 100) vs. Pix2Pix baseline, raw FID

| scope | Pix2Pix | SPADE (epoch 100) |
|---|---|---|
| overall | 80.390 | **68.125** |
| class_2 | 170.486 | **152.108** |

SPADE outperforms Pix2Pix on raw FID at both scopes (~15% better overall,
~11% better on class_2), despite the checkerboard-adjacent artifact
swapping observed during earlier smoke tests (grid/mesh texture <->
saturation blowout, never fully eliminated) and the epoch 80-90
mid-training instability. Worth discussing explicitly in the thesis
results chapter as a genuine architecture-level comparison point, not
just "both models trained successfully."

### Architecture decisions locked in for SPADE (deviations from Park et al., 2019)

1. InstanceNorm2d (not BatchNorm2d) as SPADE's base normalization -
   VRAM/batch-size=2-4 rationale, same as Pix2Pix; the paper's own
   `spadeinstance3x3` naming supports this as a configured variant.
2. `Dropout2d(0.5)` in the three bottleneck-adjacent SPADEResnetBlocks
   instead of a VAE image encoder, for sample diversity - matches
   Pix2Pix's existing stochasticity mechanism exactly, so `compute_fid.py`
   needed no SPADE-specific handling for `--samples-per-mask`.
3. `ngf=32` (not the paper's default 64) - deliberately matched to
   Pix2Pix's generator param count (29.41M vs. 29.24M) as a controlled
   variable between architectures, not merely a VRAM workaround.
4. VGG19 perceptual-loss weights: official torchvision ImageNet checkpoint,
   downloaded manually and loaded from a local path
   (`/opt/steel-defect-data/pretrained/vgg19-dcbb9e9d.pth`) -
   `download.pytorch.org` unreachable from this project's network setup;
   same weights, not an architectural deviation.
5. Asymmetric G/D learning rates (`lr_g=4e-4`, `lr_d=1e-4`) and D updated
   only every 8 steps (`d_update_every=8`) - both needed to correct a
   discriminator-overpowering-generator dynamic diagnosed via a 25-epoch
   smoke test; TTUR alone was insufficient on its own, `d_update_every`
   was the lever that actually worked (see dose-response table above).

### Next steps

1. StyleGAN2-ADA implementation next, reusing the same
   `dataset.py`/`compute_fid.py` pattern established for Pix2Pix -> SPADE.
2. Segmentation ratio experiments (75:25/50:50/25:75) once all three GANs
   are trained and their checkpoints finalized, evaluated via the
   corrected (sum-then-divide) `evaluate.py` only.
3. Thesis discussion points to carry forward: SPADE's checkerboard-adjacent
   artifact swap across training, the epoch 80-90 instability under
   `d_update_every=8`, and SPADE's FID win over Pix2Pix despite both.