"""Step 2b - train the 16 transistor models (4 cases x 4 model types) and compare them on the
validation set.

Run from the project root (after 03_tune.py):   python scripts/04_train.py
Reads   results/step2/best_hyperparameters.json
Outputs:
  models/all/<case>__<model_type>.joblib    the trained models (all types except random forest,
                                            whose files are ~186 MB each; random forests are still
                                            trained and evaluated for the comparison table)
  results/step2/model_comparison_validation.csv
        per case x model type x current: R2 (on log10|I|), median and p95 relative error
  results/step2/model_comparison_summary.csv
        per case x model type, the four currents combined (inputs for choosing the winners)
The test set is not touched here; it is used once, for the chosen models, in the next script.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml_lib as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step2"
MODELS = ROOT / "models" / "all"


def _train(case, mtype, hps):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    d = M.load_case(ROOT, case)
    tr, va, _ = M.split_indices(len(d), case)
    model = M.TransistorModel(case, mtype, hps).fit(d.iloc[tr])
    if mtype != "random_forest":
        joblib.dump(model, MODELS / f"{case}__{mtype}.joblib", compress=3)
    met = M.metrics(model.predict(d.iloc[va]), d.iloc[va][M.CURRENTS].to_numpy())
    rows = [dict(case=case, model_type=mtype, current=c, r2_log=v["r2_log"], median=v["median"], p95=v["p95"])
            for c, v in met.items()]
    return rows, time.time() - t0


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    best = json.loads((OUT / "best_hyperparameters.json").read_text())
    jobs = [(c, t, best[c][t]) for t in ["mlp", "xgboost", "random_forest", "poly_ridge"] for c in M.CASES]
    res = Parallel(n_jobs=M.n_jobs(1.0), verbose=5)(delayed(_train)(*j) for j in jobs)
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    df["model_type"] = pd.Categorical(df.model_type, M.MODEL_TYPES)
    df["case"] = pd.Categorical(df.case, M.CASES)
    df = df.sort_values(["case", "model_type", "current"])
    df.to_csv(OUT / "model_comparison_validation.csv", index=False)

    s = df.groupby(["case", "model_type"], observed=True).agg(
        r2_log_avg=("r2_log", "mean"), median_avg=("median", "mean"),
        p95_avg=("p95", "mean"), p95_worst=("p95", "max")).reset_index()
    s.to_csv(OUT / "model_comparison_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n", s.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
