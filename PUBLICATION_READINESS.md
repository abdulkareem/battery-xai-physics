# Publication Readiness Plan (High-Impact Journal)

This document converts the current repository into a publishable, reviewer-ready research package for battery degradation forecasting and explainability.

## 1) What is already strong

- End-to-end pipeline exists: heterogeneous loaders, preprocessing, physics-guided features, model training, transfer evaluation, SHAP analysis, uncertainty, and paper/table generation.
- Reproducibility scaffolding exists: deterministic utility module, config-driven runs, tests, and artifact outputs.
- Cross-dataset framing is present (NASA, CALCE, Oxford/BIL, Iontech style).

## 2) Gaps that typically block high-impact publication

### A. Scientific validation depth
- Add stronger statistical rigor:
  - Multiple random seeds with confidence intervals for every main metric.
  - Paired significance tests for model comparisons (and report effect sizes, not only p-values).
  - Calibration and reliability analysis for uncertainty intervals.

### B. External validity
- Include strict temporal split and leave-one-domain-out protocols per dataset family.
- Report degradation-stage stratified performance (early/mid/late life).
- Add domain-shift diagnostics (feature drift and SHAP drift per transfer direction).

### C. Explainability quality
- Validate explanation faithfulness:
  - Perturbation-based sanity checks.
  - Stability checks across seeds/folds.
  - Counterfactual consistency with known battery physics trends.

### D. Reproducibility + transparency
- Add dataset manifest with per-source licenses and preprocessing provenance.
- Pin exact package versions in a lock file and publish environment export used for final tables.
- Add run metadata card (git SHA, config hash, hardware, runtime, seed).

### E. Manuscript packaging
- Include threat-to-validity section with concrete mitigations.
- Add an ablation ladder:
  1) baseline tabular model,
  2) + physics features,
  3) + temporal model,
  4) + physics-informed losses,
  5) + uncertainty-aware objective.
- Add practical deployment note: inference latency, memory footprint, and expected recalibration cadence.

## 3) Minimum publishable checklist

Before submission, ensure all are complete:

- [ ] Main and supplementary tables generated from a scripted pipeline only.
- [ ] Every key result has mean ± std or CI over >= 5 seeds.
- [ ] Cross-dataset transfer matrix includes all declared source-target pairs.
- [ ] Uncertainty coverage and interval width reported together.
- [ ] SHAP global + temporal + transfer-consistency figures included.
- [ ] Data provenance/ethics/licensing statement included.
- [ ] Reproducibility section includes exact command lines and config files.

## 4) Suggested experiment matrix for the next revision

| Block | Experiments | Outcome |
|---|---|---|
| Robustness | 5-10 seeds x all transfer protocols | Variance-aware claims |
| Generalization | Leave-one-dataset-out + temporal holdout | Stronger out-of-domain evidence |
| Explainability | Faithfulness + stability audits | Defensible XAI claims |
| Uncertainty | Coverage/width tradeoff across models | Actionable reliability claims |
| Physics priors | Controlled ablations of each physics feature group | Mechanistic interpretation |

## 5) Fast path to submission quality

1. Freeze a final benchmark config and 5-seed schedule.
2. Run complete transfer + ablation matrix.
3. Auto-generate tables/figures from artifacts (no manual edits).
4. Add statistical appendix and failure-case analysis.
5. Package reproducibility bundle (configs, commands, environment, checksums).

## 6) Notes on current maintenance fix

- The incremental-capacity feature extraction now uses `numpy.trapezoid` instead of removed `numpy.trapz` to maintain compatibility with NumPy 2.x and keep physics-derived area metrics reproducible.
