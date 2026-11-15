"""Step 2a - hyperparameter search for every case x model type x part (train -> validation).

Run from the project root:   python scripts/03_tune.py        (long: about 1-2 h on 2 cores)
Each of the three parts of a transistor model (Ig, Ib, channel part) is tuned separately on
its own validation error: mean relative error of the physical quantity (|Ig|, |Ib|, Ich).
For the channel part the transform ("plain" or "conductance") is tuned as well.
The test set is not touched.
Outputs (results/step2/):
  tuning_results.csv          every configuration tried, with its validation errors
  best_hyperparameters.json   chosen settings per case / model type / part (read by 04_train.py)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per fit: reproducible and no oversubscription

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml_lib as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step2"

GB_PER_FIT = {"mlp": 0.3, "xgboost": 0.3, "random_forest": 1.0, "poly_ridge": 2.0}   # rough peak memory

GRIDS = {
    "poly_ridge": [dict(degree=d, alpha=a) for d in (2, 3, 4) for a in (1e-8, 1e-6, 1e-4, 1e-2)],
    "random_forest": [dict(n_estimators=300, max_features=f, min_samples_leaf=m)
                      for f in (0.33, 0.66, 1.0) for m in (1, 3)],
    "xgboost": [dict(n_estimators=n, learning_rate=0.03, max_depth=d, min_child_weight=1,
                     subsample=0.8, colsample_bytree=0.8) for n in (1000, 3000) for d in (4, 6, 8)],
    "mlp": [dict(hidden=h, alpha=a, lr=1e-3, max_iter=4000, patience=150)
            for h, a in (([64, 64, 64], 1e-6), ([128, 128, 128], 1e-6), ([128, 128, 128], 1e-4),
                         ([256, 256], 1e-6))],
}


def _job(case, mtype, part, form, hp):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    d = M.load_case(ROOT, case)
    tr, va, _ = M.split_indices(len(d), case)
    dt, dv = d.iloc[tr], d.iloc[va]
    yt, _ = M.targets(dt)
    yv, _ = M.targets(dv)
    key = "c_" + form if part == "c" else part
    reg = M.ScaledTarget(M.make_regressor(mtype, hp)).fit(M.features(dt), yt[key])
    p = reg.predict(M.features(dv, vds_floor=M.VDS_FLOOR))   # same input clip as at prediction time
    r = np.abs(10.0 ** (p - yv[key]) - 1.0)          # relative error of the physical quantity
    return dict(case=case, model_type=mtype, part=part, form=form, hp=json.dumps(hp),
                val_mean=float(r.mean()), val_median=float(np.median(r)),
                val_p95=float(np.percentile(r, 95)), seconds=round(time.time() - t0, 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for mtype in ["mlp", "xgboost", "random_forest", "poly_ridge"]:      # slowest first
        for case, part, hp in itertools.product(M.CASES, M.PARTS, GRIDS[mtype]):
            for form in (("plain", "conductance") if part == "c" else ("-",)):
                jobs.append((case, mtype, part, form, hp))
    rows = []
    for mtype in ["mlp", "xgboost", "random_forest", "poly_ridge"]:
        js = [j for j in jobs if j[1] == mtype]
        nj = M.n_jobs(GB_PER_FIT[mtype])
        print(f"{mtype}: {len(js)} fits, {nj} in parallel ({os.cpu_count()} cores; "
              f"capped by free memory, ~{GB_PER_FIT[mtype]} GB per fit)", flush=True)
        rows += Parallel(n_jobs=nj, verbose=5)(delayed(_job)(*j) for j in js)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "tuning_results.csv", index=False)

    best = {}
    for (case, mtype, part), g in res.groupby(["case", "model_type", "part"]):
        r = g.sort_values(["val_mean", "val_p95"]).iloc[0]
        hp = json.loads(r.hp)
        if part == "c":
            hp["form"] = r.form
        best.setdefault(case, {}).setdefault(mtype, {})[part] = hp
    (OUT / "best_hyperparameters.json").write_text(json.dumps(best, indent=2))
    print("written:", OUT / "tuning_results.csv", OUT / "best_hyperparameters.json")


if __name__ == "__main__":
    main()
