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

### Next steps

1. Rename/generalize `get_class3_image_ids()` /
   `build_class3_weighted_sampler()` in `dataset.py` to work off
   `CLASS_ID` generically (or a class_2-specific version) - currently
   still hardcoded to class_3.
2. Re-run the sampler sanity check with class_2 and pick a sensible
   `target_fraction` given its much lower natural frequency (3.7%).
3. Implement Pix2Pix generator (`generator.py`) and discriminator
   (`discriminator.py`).
4. Implement training loop (`train.py`) with MLflow logging, following
   the same conventions as `src/segmentation/train.py`.
5. Later: SPADE, then StyleGAN2-ADA.
6. Real:synthetic ratio experiments (75:25 / 50:50 / 25:75) evaluated via
   the corrected `evaluate.py` only.