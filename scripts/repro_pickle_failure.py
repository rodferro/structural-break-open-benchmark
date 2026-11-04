"""Reproduce the April 2026 out-of-sample failure.

Step 1 (old env, tabpfn==6.4.1):  python scripts/repro_pickle_failure.py dump  /path/model.joblib
Step 2 (new env, tabpfn>=7.0):    python scripts/repro_pickle_failure.py load  /path/model.joblib

Under a newer tabpfn, step 2 raises
    AttributeError: 'PerFeatureTransformer' object has no attribute
    'pre_generated_column_embeddings'
because the pickled nn.Module carries the *old* instance __dict__ while the
unpickling process binds it to the *new* class, whose forward() expects
attributes that the old __init__ never created.
"""

import sys

import joblib
import numpy as np


def _toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    return X, y


def main() -> None:
    from importlib.metadata import version

    import tabpfn

    mode, path = sys.argv[1], sys.argv[2]
    print(f"tabpfn=={version('tabpfn')}  tabpfn-common-utils=={version('tabpfn-common-utils')}")
    X, y = _toy()
    if mode == "dump":
        clf = tabpfn.TabPFNClassifier(device="cpu", random_state=0)
        clf.fit(X, y)
        print("fitted proba:", clf.predict_proba(X[:3])[:, 1])
        joblib.dump(clf, path)
        print("dumped to", path)
    elif mode == "load":
        clf = joblib.load(path)
        print("loaded OK; predicting ...")
        print("proba:", clf.predict_proba(X[:3])[:, 1])
    else:
        raise SystemExit("mode must be dump|load")


if __name__ == "__main__":
    main()
