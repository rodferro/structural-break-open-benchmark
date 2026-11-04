# ADIA Lab Structural Break Open Benchmark — submission #28, rebuilt

Binary classification on CrunchDAO's
[Structural Break Open Benchmark](https://hub.crunchdao.com/competitions/structural-break-open-benchmark):
for each univariate time series, with a known candidate boundary, predict whether
the data-generating process changes at that boundary. Metric: ROC AUC.

Submission #28 scored **0.8089 on the public leaderboard**. Its out-of-sample
run then failed at inference time. This repository reproduces the failure,
fixes it, and turns the notebook into a tested package.

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
3. **Base models** (`sbreak/models.py`). LightGBM, CatBoost and TabPFN are
   each trained in 5 stratified folds. Test predictions are the fold average.
4. **Stacking** (`sbreak/stacking.py`). A class-balanced logistic regression
   is fitted on the three out-of-fold probability columns.

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
6.4.1 and loads it under another environment. I confirmed that 6.4.1 → 6.4.1
works and that 6.4.1 → 7.0.0 and 6.4.1 → 7.1.1 fail with exactly the error above.

**Root causes**

- Unpinned dependencies. Only `numpy`, `scikit-learn` and `lightgbm` were
  pinned. `tabpfn`, `tabpfn-common-utils`, `torch` and `catboost` floated.
- Pickling library objects whose internals are not a stable interface.
  The same risk applied to the CatBoost models and the scikit-learn stacker,
  which were also joblib pickles.
- No test exercised "train in environment A, infer in a freshly built
  environment B".

**Fixes**

- `requirements.txt` pins every runtime dependency to the versions that
  produced the leaderboard score (tabpfn 6.4.1, tabpfn-common-utils 0.2.18,
  torch 2.9.0, lightgbm 4.6.0, catboost 1.2.8, ...).
- No fitted estimator is pickled. LightGBM is saved in its text format,
  CatBoost as `.cbm` and the stacker as JSON coefficients. TabPFN is
  *refitted at inference*: its "fit" is in-context learning (preprocessing
  plus storing the data), so we save the fold's training matrix and labels
  (parquet) and the checkpoint file, and call `fit` again in `infer()`.
  Refitting is deterministic; the test suite checks that predictions from a
  fresh process match the training process (to 1e-6). Shipping the checkpoint
  means inference does not need network access to the Hugging Face Hub.
- `resources/manifest.json` records library versions. Version drift at
  inference raises a warning naming the package instead of an obscure
  crash. The 6.4.1-trained artifacts load and predict under tabpfn 7.1.1.
- CI (`.github/workflows/smoke.yml`) builds a clean environment from
  `requirements.txt`, trains tiny models on synthetic data, reloads every saved
  model in a fresh interpreter and predicts. It also runs weekly, so a
  dependency that disappears or breaks shows up before a deadline does.

**Two smaller latent bugs found along the way**

- *Train/infer skew.* Some feature names embed the realised window length
  (e.g. `mc_mean_b256_a512`), so different series produce different columns.
  The training matrix had NaN where a column did not apply, but `infer()`
  filled absent columns with `0.0`. The fill is now NaN. Feature set `v2`
  names these windows by their nominal size, which removes the problem.
- `class_weight="balanced"` was passed to `lgb.train`, which ignores
  it (it is a scikit-learn-API parameter). It has been dropped. Behaviour is
  unchanged.

**macOS note.** If `lightgbm` is imported before `torch`, two OpenMP runtimes
are loaded and TabPFN deadlocks on its first multi-threaded op.
`sbreak/__init__.py` imports torch first.

## Validation

Submission #28 was validated optimistically, in three ways:

- Features were ranked on *all* training rows, then cross-validated.
- Each base model was early-stopped on the fold it was scored on.
- The stacker was scored on the data it was fitted on.

`sbreak/evaluation.py` instead runs **nested CV**: an outer 5-fold split
over series, and inside each outer-train set the *whole* procedure
(both selection stages, 5-fold base models, stacker). Only the outer-held-out
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
- TabPFN adds a small, very consistent gain to the stack. CatBoost is the
  strongest single model. The stack is only slightly better than CatBoost
  alone, and a rank-average stacker matches the logistic one.
- Neither a stronger-regularised LightGBM (it also changes which features
  the selection picks, for the worse) nor a larger final feature set
  helped.

Local TabPFN runs on a laptop CPU used a 3,000-row context and 4
estimators. `scripts/kaggle_tabpfn.py` runs the real configuration on a GPU:

```bash
.venv/bin/python -m scripts.make_kaggle_bundle   # dist/kaggle_bundle.zip: code + selected feature names
# on Kaggle (GPU, internet on): attach the bundle as a dataset plus the competition data, then
#   !pip install -q -r /kaggle/input/<bundle>/requirements.txt
#   !cd /kaggle/input/<bundle> && python -m scripts.kaggle_tabpfn --data-dir /kaggle/input/<data> --out /kaggle/working/runs
# download runs.zip, unzip it into this repo, then:
.venv/bin/python -m scripts.evaluate --models lgb cat tabpfn          # uses the GPU TabPFN predictions
.venv/bin/python -m scripts.evaluate --preset submission28 --models lgb cat tabpfn
```

## Usage

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

# put X_train.parquet, y_train.parquet, X_test.reduced.parquet, y_test.reduced.parquet in the repo root
.venv/bin/python -m sbreak.runner                    # CrunchDAO-like train + infer + local score
.venv/bin/python -m sbreak.runner --skip-train --determinism-check
.venv/bin/python -m sbreak.runner --synthetic --smoke  # no data needed

.venv/bin/python -m pytest -q                          # smoke tests

# experiments (outputs under runs/, gitignored)
.venv/bin/python -m scripts.build_features --feature-set v1   # cache features in .cache/
.venv/bin/python -m scripts.evaluate --models lgb cat         # nested CV
.venv/bin/python -m scripts.evaluate --leaky                  # submission #28's own CV protocol
```

The submission directory is `main.py`, `sbreak/` and `requirements.txt`.
`train()` writes everything into `resources/`.

## Layout

```
main.py                    CrunchDAO entry point (train / infer)
sbreak/features.py         per-series features (v1 = submission #28, v2, v3)
sbreak/extra_features.py   additional two-sample / dependence features (v3)
sbreak/selection.py        two-stage LightGBM-importance selection
sbreak/models.py           LightGBM / CatBoost / TabPFN with pickle-free persistence
sbreak/stacking.py         K-fold base models + stacker
sbreak/pipeline.py         train / infer / manifest
sbreak/evaluation.py       nested CV and the original (leaky) protocol
sbreak/runner.py           local CrunchDAO-style runner
scripts/                   feature cache, evaluation CLI, pickle-failure repro
legacy/                    the original submission #28 (setup token redacted)
tests/                     smoke tests (synthetic data only)
```
