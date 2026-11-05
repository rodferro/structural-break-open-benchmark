"""Score the average of several nested-CV runs' out-of-fold predictions (seed bagging).

    python -m scripts.seed_bag baseline baseline_seed2 [baseline_seed3 ...]

Every input is an honest outer-fold prediction for the same rows, so their
average is too; per-fold AUC uses the same outer folds as scripts.evaluate.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sbreak.evaluation import OUTER_SEED


def main() -> None:
    tags = sys.argv[1:]
    oofs = [pd.read_parquet(f"runs/results/{t}.oof.parquet") for t in tags]
    idx = oofs[0].index
    y = pd.read_parquet("y_train.parquet")["structural_breakpoint"].astype(int).loc[idx]
    outer = list(StratifiedKFold(5, shuffle=True, random_state=OUTER_SEED).split(idx, y))
    cols = {"stack (prob mean)": np.mean([o["stack"].loc[idx] for o in oofs], axis=0)}
    for m in ("lgb", "cat"):
        if all(m in o for o in oofs):
            cols[f"{m} (prob mean)"] = np.mean([o[m].loc[idx] for o in oofs], axis=0)
    singles = {t: o["stack"].loc[idx].to_numpy() for t, o in zip(tags, oofs, strict=True)}
    for name, p in {**singles, **cols}.items():
        folds = []
        for _, val in outer:
            # rank within the fold, as the per-fold AUCs do
            folds.append(roc_auc_score(y.iloc[val], rankdata(p[val])))
        print(f"{name:28s} mean fold AUC {np.mean(folds):.4f}  folds {np.round(folds, 4).tolist()}")


if __name__ == "__main__":
    main()
