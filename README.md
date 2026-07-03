# psf-transcript-correction
Wiener deconvolution of the optical PSF for transcript position
correction in imaging-based spatial transcriptomics.

## The Problem
The optical PSF in Xenium, MERSCOPE, and CosMx is strongly anisotropic:
- Lateral xy: sigma = 0.2 um
- Axial z: sigma = 2.8 um — 14x difference

This blurs transcript axial (z) positions, contributing to systematic
transcript-to-cell misassignment near boundaries. Note: 10x Genomics'
Xenium Onboard Analysis pipeline already applies some instrument-level
PSF-aware refinement of punctum XYZ coordinates before transcripts.parquet
is written (see Limitations). This method is a residual, second-pass
correction on top of that on-instrument step, not a from-scratch
PSF correction of raw, uncorrected data.

## The Method
3D Wiener deconvolution in k-space:

W(k) = H*(k) / (|H(k)|^2 + lambda)

Mathematically equivalent to the Bayes-optimal Kalman estimator.
Lambda estimated adaptively from the local density field (no manual
tuning). Transcript positions are corrected using a noise-robust
weighted centroid of the corrected density field in a local window
around each observed position (a hard argmax was tested and found to
be far more sensitive to deconvolution noise; see validation/ scripts
for the comparison).

## Simulation Results (synthetic data, known ground truth)

Primary metric — population z-std reduction, grouped by simulated cell
of origin, matching the metric used on real data (real data has no
ground truth, so this population-scatter metric is what's actually
measurable there). Ground truth confirms this is genuine recovered
accuracy, not just variance shrinkage: the "gap to true std" column
shows the corrected population converging toward the actual known
biological std, not just tightening arbitrarily.

| Cell type  | n    | True std | Z-std before | Z-std after | Improvement | Gap to truth: before to after |
|------------|------|----------|--------------|-------------|-------------|--------------------------------|
| Neuron     | 1064 | 0.91 um  | 2.94 um      | 1.24 um     | +57.8%      | 2.03 um to 0.33 um |
| Glial      | 23   | 0.28 um  | 2.80 um      | 1.81 um     | +35.3%      | 2.52 um to 1.53 um |
| Macrophage | 40   | 0.27 um  | 2.53 um      | 0.86 um     | +66.1%      | 2.26 um to 0.58 um |

Secondary metric — per-transcript distance to known truth (stricter):
Z improvement +18.4%, XY improvement -2.9% (near-neutral; sigma_xy is
already small, leaving little room to correct, and deconvolution noise
roughly cancels the small achievable gain). The method's benefit is
concentrated in Z, consistent with the 14x PSF anisotropy.

## Real-Data Validation

Two independent public Xenium datasets, different tissue types, same
platform and PSF parameters. Metric: z-position standard deviation
before vs. after correction (population scatter reduction — no ground
truth available on real data, so this is the same metric validated
against ground truth in simulation above).

### DRG (GSE273557, Price Lab / Yu et al., Nature Neuroscience 2024)
Per-neuron z-std, computed within each of 14 annotated neurons'
~10um vicinity (matches the spatial scale of the PSF effect itself,
rather than pooling across a whole tissue region where real biological
z-spread swamps the ~2-3um PSF signal):

**Mean improvement: +2.5% | Median: +1.1% | 13 of 14 neurons improved**

### Breast Cancer (10x Genomics, Xenium FFPE Human Breast Cancer Rep1,
Janesick et al. 2023, Nature Communications)
Per-gene z-std across 7 marker genes spanning distinct compartments,
411,517 transcripts in a dense-tissue window:

| Gene   | Compartment              | n      | Improvement |
|--------|---------------------------|--------|-------------|
| EPCAM  | Epithelial (broad)        | 9,622  | +5.1% |
| ERBB2  | Tumor biomarker (HER2)    | 31,541 | +4.0% |
| ESR1   | Tumor biomarker (ER)      | 1,697  | +5.2% |
| KRT14  | Myoepithelial             | 1,865  | +3.0% |
| ACTA2  | Myoepithelial/sm. muscle  | 3,693  | +3.1% |
| PTPRC  | Immune (CD45)             | 615    | +3.1% |
| PECAM1 | Endothelial               | 462    | +1.6% |

All 7 genes positive. Effect size is modest (1-5%) but consistent in
direction and magnitude across two independently-sourced datasets and
different tissue architectures (ECM-rich DRG neuropil vs. dense
epithelial/ductal breast tissue) — this consistency, not the size of
any single number, is what makes the result credible.

## Limitations
- **XY correction does not work** — near-neutral to slightly negative
  in both simulation and real data. Only Z (the dominant axis of PSF
  anisotropy) shows a reliable, reproducible improvement.
- **On-instrument PSF calibration**: measuring real z-scatter within a
  tight (~10um) neuron vicinity in the DRG data gives 2.31um vs. the
  theoretical raw PSF sigma_z of 2.8um (0.83x ratio) — evidence that
  10x's own on-instrument calibration already removes a meaningful
  fraction of correctable axial error before this method is applied.
  The 1-5% improvements reported above are therefore a residual,
  second-pass gain on top of that, not a from-scratch correction.
- **Effect size is small.** This is not a dramatic result. Earlier
  drafts of this repo reported much larger numbers (24-99%); those
  were found to be derived from circular/hardcoded values in the
  simulation script rather than measurements, and have been corrected.
  The numbers in this README are the actual output of the linked
  scripts, run end-to-end, with no manual adjustment.
- Tested on 500x500um windows, not full tissue sections.
- Density-grid-based correction (0.5um voxels) is coarser than
  per-punctum fitting; a finer-grained approach may recover more signal.

## Reproducing These Results
- `psf_simulation_paper1.py` — simulation with known ground truth
- `validation/psf_validation_per_neuron.py` — DRG per-neuron validation (primary real-data result)
- `validation/psf_validation_v2.py` — DRG per-gene, whole-window validation
- `validation/psf_validation_breast.py` — breast cancer per-gene validation
- `validation/check_oninstrument_correction.py` — on-instrument correction diagnostic
- `validation/download_breast.sh` — downloads the public breast dataset

## Datasets
- DRG: GEO accession GSE273557 (Price Lab / Yu et al. 2024), public
- Breast: 10x Genomics Xenium FFPE Human Breast Cancer Rep1 (Janesick
  et al. 2023, Nature Communications companion dataset), public, CC-licensed

## Generalization to Other Platforms
The underlying problem — anisotropic axial PSF blur from widefield/
epifluorescence-style optics — is not Xenium-specific; CosMx (NanoString)
and MERFISH/MERSCOPE (Vizgen) use comparable imaging architectures and
plausibly have their own axial anisotropy. The method itself is already
platform-agnostic: SIGMA_XY and SIGMA_Z are isolated constants at the
top of each script, so applying this to another platform is in principle
a two-line change once real values are known.

**Not done yet, and deliberately not claimed here**: neither NanoString
nor Vizgen publish a specific Gaussian PSF sigma_z the way 10x documents
for Xenium (checked their public technical documentation directly).
Using the Xenium sigma_z=2.8um for a different platform's optical system
would be exactly the kind of unsupported number this project spent real
effort removing elsewhere. Extending this method to CosMx or MERFISH
requires either (a) sourcing a calibrated PSF value from platform
vendor documentation or a published characterization, or (b) empirical
PSF characterization from bead calibration images (standard practice in
single-molecule localization microscopy) on real platform data — not
yet attempted here.

## Status
Simulation and real-data validation complete on two independent public
Xenium datasets. Effect is modest, Z-only, and consistent with 10x's own
on-instrument PSF calibration already handling most of the correctable
error. Generalization to CosMx/MERFISH is architecturally straightforward
but not yet validated — no trustworthy PSF parameters for those platforms
have been sourced. Preprint in preparation, framed as a small
residual-correction method rather than a from-scratch PSF correction.

## Author
Firas Manasrah — Berlin, June 2026
github.com/firas-manasrah

## License
MIT
