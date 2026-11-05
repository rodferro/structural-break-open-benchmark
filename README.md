# ADIA Lab Structural Break Open Benchmark — submission #28, rebuilt

Binary classification on CrunchDAO's
[Structural Break Open Benchmark](https://hub.crunchdao.com/competitions/structural-break-open-benchmark):
for each univariate time series, with a known candidate boundary, predict whether
the data-generating process changes at that boundary. Metric: ROC AUC.

Submission #28 scored **0.8089 on the public leaderboard**. Its out-of-sample
run then failed at inference time, so it never received a final ranking. The
competition has since closed. This repository reproduces the failure, fixes it,
turns the notebook into a tested package, and re-evaluates the model honestly.

> Competition data, trained artifacts and non-public scores are deliberately
> **not** in this repository (see `.gitignore`).

## Approach

1. **Features** (`sbreak/features.py`). Every series is split at the boundary
   into a *before* segment `a` (period 0) and an *after* segment `b` (period 1).
   For each series we compute ~650 hand-crafted comparisons of `a` and `b`:
   moments, quantiles, robust and winsorised statistics, effect sizes
   (Cohen's d), boundary-local differences over windows of 3 to 256 points,
   statistics of transformed series (z-score, rank, |x|, rolling mean/std),
   outlier and jump counts, a Chow F-statistic, Welch t, KS, Levene and
   Wasserstein distances.
2. **Feature selection** (`sbreak/selection.py`). LightGBM gain importance over
   5 folds ranks the raw features. Submission #28 kept the top 80, added
   pairwise interactions (×, ÷, |×|) among the top 20, and kept the top 100
   after a second pass (`submission28_config()`). The current default keeps
   the top 200 raw features with no interactions (see *Validation*).
3. **Base models** (`sbreak/models.py`). LightGBM and CatBoost are each
   trained in 5 stratified folds. Test predictions are the fold average.
   Submission #28 also stacked TabPFN; it has been dropped (see below).
4. **Stacking** (`sbreak/stacking.py`). A class-balanced logistic regression
   is fitted on the out-of-fold probability columns.

**Why TabPFN was dropped.** It is only practical on a GPU (a fold's
prediction takes many minutes on a laptop CPU, and Apple MPS runs out of
memory at this context size), it pulled in torch, and in nested CV it added
only a small gain to the stack. Everything now runs on CPU.

`main.py` is the CrunchDAO entry point (`train` / `infer`). Everything else
lives in the `sbreak` package.

## Post-mortem: why the out-of-sample run failed

**What happened.** `train()` ran in March 2026 and saved each fitted
`TabPFNClassifier` with `joblib.dump`. In April the platform rebuilt the
environment for `infer()` from `requirements.txt`, where `tabpfn` was
unpinned. pip installed tabpfn 7.x instead of the 6.4.1 used at training.
`joblib.load` succeeded, then the first `predict_proba` raised:

```
AttributeError: 'PerFeatureTransformer' object has no attribute 'pre_generated_column_embeddings'
```

**Why.** Unpickling an `nn.Module` restores the *old* instance `__dict__`
but binds it to the *new* class. tabpfn 7.0.0 (released 2026-03-24, between
the two runs) changed `PerFeatureTransformer` so that its forward pass reads an
attribute that 6.4.1's `__init__` never created. Nothing fails at load time,
so a "can the models be loaded?" check would not have caught it.

**Reproduction.** `scripts/repro_pickle_failure.py` dumps a model under
6.4.1 and loads it under another environment. 6.4.1 → 6.4.1 works;
6.4.1 → 7.0.0 and 6.4.1 → 7.1.1 fail with exactly the error above. TabPFN is no
longer a project dependency, so the script has its own pinned environments:
see `legacy/requirements-repro.txt` for the two-venv recipe.

**Root causes**

- Unpinned dependencies. Only `numpy`, `scikit-learn` and `lightgbm` were
  pinned. `tabpfn`, `tabpfn-common-utils`, `torch` and `catboost` floated.
- Pickling library objects whose internals are not a stable interface.
  The same risk applied to the CatBoost models and the scikit-learn stacker,
  which were also joblib pickles.
- No test exercised "train in environment A, infer in a freshly built
  environment B".

**Fixes**

- `requirements.txt` pins every runtime dependency (lightgbm 4.6.0,
  catboost 1.2.8, scikit-learn 1.6.1, numpy 2.0.2, ...).
- No fitted estimator is pickled. LightGBM is saved in its text format,
  CatBoost as `.cbm` and the stacker as JSON coefficients. The test suite
  checks that predictions from a fresh process match the training process
  (to 1e-6).
- While TabPFN was still in the stack, it was *refitted at inference* from its
  saved training context instead of being unpickled. Artifacts trained under
  tabpfn 6.4.1 then loaded and predicted under 7.1.1. Dropping TabPFN removed
  the problem at its source.
- `resources/manifest.json` records library versions. Version drift at
  inference raises a warning naming the package instead of an obscure
  crash.
- CI (`.github/workflows/smoke.yml`) builds a clean environment from
  `requirements.txt` on Python 3.11 (the platform's version) and 3.12, trains
  tiny models on synthetic data, reloads every saved model in a fresh
  interpreter and predicts.
- A weekly **drift canary** job reproduces the conditions of the failure: it
  trains under the pinned versions, upgrades the shipped libraries (LightGBM,
  CatBoost, scikit-learn, SciPy, NumPy, pandas, pyarrow) to their latest
  releases, then runs inference on the same artifacts. It runs on a schedule (and on demand), not on pull requests, so an
  upstream release cannot block unrelated work.

**Two smaller latent bugs found along the way**

- *Train/infer skew.* Some feature names embed the realised window length
  (e.g. `mc_mean_b256_a512`), so different series produce different columns.
  The training matrix had NaN where a column did not apply, but `infer()`
  filled absent columns with `0.0`. The fill is now NaN. Feature set `v2`
  names these windows by their nominal size, which removes the problem.
- `class_weight="balanced"` was passed to `lgb.train`, which ignores
  it (it is a scikit-learn-API parameter). It has been dropped. Behaviour is
  unchanged.

## Validation

Submission #28 was validated optimistically, in three ways:

- Features were ranked on *all* training rows, then cross-validated.
- Each base model was early-stopped on the fold it was scored on. This one
  is still the default inside the stack (`early_stop_on_val_fold=True`): it
  biases the stacker's training data, but not the nested-CV outer scores,
  which are always computed on untouched rows. The alternative was ablated
  and is worse (see below).
- The stacker was scored on the data it was fitted on.

`sbreak/evaluation.py` instead runs **nested CV**: an outer 5-fold split
over series, and inside each outer-train set the *whole* procedure
(feature selection, 5-fold base models, stacker). Only the outer-held-out
rows are scored. `leaky_cv()` reproduces the old protocol so the two can be
compared on the same folds. All ablations share the same outer folds and are
compared with paired per-fold differences (`scripts/compare.py`).

Findings (qualitative; per-fold numbers stay local under `runs/`):

- The old protocol is clearly optimistic. Nested CV lands noticeably
  *below* both the old CV estimate and the 0.8089 public-leaderboard score,
  so that score should not be read as this model's expected out-of-sample AUC.
- **Dropping the pairwise-interaction stage** (keeping the top 200 raw
  features instead of 80 raw + interactions → 100) is the one change that
  helped clearly and consistently across folds, with every feature set
  tried. It is the new default.
- Length-independent feature names (`v2`) are neutral for AUC but remove
  the train/infer skew. They are now the default.
- Extra two-sample and dependence features (`v3`) are selected but add
  nothing measurable: they are redundant with the existing set.
- TabPFN added a small, very consistent gain to the stack, but not enough to
  justify a GPU dependency; it has been dropped. CatBoost is the strongest
  single model. The stack is only slightly better than CatBoost alone, and a
  rank-average stacker matches the logistic one.
- Neither a stronger-regularised LightGBM (it also changes which features
  the selection picks, for the worse) nor a larger final feature set
  helped.

### CPU feature-engineering campaign

Re-run of the default on CPU (LightGBM + CatBoost), then error analysis and
targeted feature work. Every experiment uses the same outer folds and is
compared to the baseline by paired per-fold differences.

**Noise floor.** Re-running the *identical* configuration with a different
seed (which changes the inner folds of selection and stacking and CatBoost's
seed) moves the score by about as much as most of the changes below, and this
seed luck is shared across outer folds, so "worse in every fold" is not by
itself evidence. Three seeds of the default bracket the noise; the original
baseline run turned out to be the luckiest of the three, so single-seed
comparisons against it were biased against every change, and the results
below are read against the range of the three seeds. Averaging the honest
predictions of three seeds helps only marginally, for three times the compute.

**Error analysis** (`scripts/error_analysis.py`, nested-CV out-of-fold
predictions only):

- Segment lengths carry no label information (before/after lengths are
  distributed identically for breaks and non-breaks). AUC rises with the
  length of the after-boundary segment: short after-segments are simply harder.
- The training set is a **mixture of series families** with different scales:
  the scale distribution is multimodal, the positive rate differs between
  families, and the model is clearly weakest on one large family. The model
  already sees scale (raw-scale features are always selected); inside the weak
  family, the best separators are drift *levels* that the base set already
  contains.
- **Missed breaks** look like non-breaks in their marginal distribution (low
  KS, Wasserstein and Levene statistics); many have no detectable mean or
  variance change. **False alarms** are non-breaks whose before/after
  distributions do differ.
- A residual-signal probe (candidate statistics scored where the model is
  uncertain) found only weak extra signal: dependence and spectral changes,
  distribution shape after standardisation, and boundary scan statistics.

**Feature review** (`scripts/feature_audit.py`): no leakage (every feature is
computed per series without labels); 17 constant columns (z-score outlier
counts at |z| > 3 or 4 inside 10-point windows can never fire, since the
largest possible |z| with 10 points is 2.85); 4 exact duplicates; whole-segment
counts and energy sums that grow with segment length even without a break;
`-log10 p` capped at 8 and Levene p-values left raw (saturated for a large
share of series). Feature extraction is cheap; CatBoost training dominates the
CPU budget.

**Experiments** (each group is its own flag: `Config.feature_groups`,
`Config.exclude_features`):

- Changes are judged against the spread of three seeds of the unchanged
  default: inside it means no measurable effect, below its worst seed means
  worse.
- `scan` (how special the given boundary is among all split points: CUSUM of
  mean, ranks, variance and absolute deviation, and a Gaussian likelihood
  ratio): worse. The columns rank at the top of the importance list yet lower
  out-of-sample AUC, including against a matched-width control and in a
  single model in isolation.
- `selfnorm` (boundary contrast relative to contrasts between windows inside
  the before-segment, aimed at the false alarms): worse, and unstable.
- `tests` (unsaturated, length-normalised two-sample statistics): no
  measurable change.
- `@audit` exclusions (constants, duplicates, length-confounded counts):
  no measurable change; the counts still carry usable signal.
- Inner-split early stopping (`early_stop_on_val_fold=False`): clearly
  worse.
- Top 150 or 225 features instead of 200: no measurable change.

**Conclusion.** None of the changes beat the noise floor, so the default is
unchanged. The groups that *looked* most informative (top importance ranks)
hurt the most: on this data, LightGBM gain is a poor guide to whether a new
feature generalises, which is why every candidate is judged by nested CV
rather than by importance. The groups and exclusion presets stay in the code,
off by default, so they can be re-tested (for example averaged over several
seeds) without new code.

### Sealed holdout

`y_test.reduced.parquet` holds labels for the reduced test set. It is used as a
sealed holdout: feature selection, tuning and ablations use nested CV on the
training data only, and the holdout is scored **once**, on the frozen final
configuration, with `--score-holdout`. The runner never reads it otherwise.

## Usage

```bash
python3.12 -m venv .venv        # 3.11 also works (the CrunchDAO runner used 3.11)
.venv/bin/pip install -r requirements-dev.txt

# put X_train.parquet, y_train.parquet, X_test.reduced.parquet, y_test.reduced.parquet in the repo root
.venv/bin/python -m sbreak.runner                    # CrunchDAO-like train + infer (no scoring)
.venv/bin/python -m sbreak.runner --skip-train --determinism-check
.venv/bin/python -m sbreak.runner --skip-train --score-holdout   # once, on the final config
.venv/bin/python -m sbreak.runner --synthetic --smoke  # no data needed

.venv/bin/python -m pytest -q                          # smoke tests

# experiments (outputs under runs/, gitignored; numbers go in reports/private/)
.venv/bin/python -m scripts.build_features --feature-set v2           # base table -> .cache/
.venv/bin/python -m scripts.build_features --groups scan tests selfnorm  # one cache per group
.venv/bin/python -m scripts.evaluate --tag baseline                   # nested CV, default config
.venv/bin/python -m scripts.evaluate --groups tests --tag tests       # default + one group
.venv/bin/python -m scripts.evaluate --set 'exclude_features=["@audit"]' --tag audit_drop
.venv/bin/python -m scripts.evaluate --set seed=12345 --tag baseline_seed2   # noise floor
.venv/bin/python -m scripts.compare baseline tests audit_drop         # paired deltas, folds won, compute
.venv/bin/python -m scripts.seed_bag baseline baseline_seed2          # average of runs
.venv/bin/python -m scripts.error_analysis --tag baseline             # where the model fails
.venv/bin/python -m scripts.feature_audit --feature-set v2            # constants, duplicates, length effects
.venv/bin/python -m scripts.evaluate --preset submission28 --leaky    # submission #28's own CV protocol
```

Long runs: start them with `nohup … > runs/logs/<name>.log 2>&1 &` so they
survive the editor closing. Cache writes are atomic, and an unreadable cache
entry is recomputed, so an interrupted run can simply be restarted.

`main.py`, `sbreak/` and `requirements.txt` are what was submitted to
CrunchDAO; the local runner reproduces the platform's train/infer cycle.
`train()` writes everything into `resources/`. Trained artifacts are
gitignored and should never be shared.

## Layout

```
main.py                    CrunchDAO entry point (train / infer)
sbreak/features.py         per-series features (v1 = submission #28, v2, v3)
sbreak/extra_features.py   additional two-sample / dependence features (v3)
sbreak/feature_groups.py   optional, separately ablatable feature groups (scan, tests, selfnorm)
sbreak/exclusions.py       named lists of base features to drop before selection (@audit)
sbreak/selection.py        LightGBM-importance selection (optional interaction stage)
sbreak/models.py           LightGBM / CatBoost with pickle-free persistence
sbreak/stacking.py         K-fold base models + stacker
sbreak/pipeline.py         train / infer / manifest
sbreak/evaluation.py       nested CV and the original (leaky) protocol
sbreak/runner.py           local CrunchDAO-style runner
scripts/                   feature cache, evaluation, comparison, error analysis, audit, pickle-failure repro
legacy/                    the original submission #28 (setup token redacted)
tests/                     smoke tests (synthetic data only)
```
