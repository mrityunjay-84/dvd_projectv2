"""Step 2c - choose one model per case on the validation set, then evaluate it ONCE on the test set.

Run from the project root (after 04_train.py):   python scripts/05_select_and_test.py
Selection: lowest median relative error on validation, averaged over Id, Ig, Is, Ib. The script
also reports the winner under the other candidate rules (average p95, worst-current p95, R2) so any
disagreement is visible.
For the cases in ml_lib.N_AVERAGE (NMOS OFF, PMOS OFF) the chosen model is the average of 3
networks with the chosen settings and random starts 0, 1, 2 (start 0 is the model from 04_train.py).
Outputs:
  models/<case>.joblib                        the four chosen models (used by the gate scripts)
  results/step2/chosen_models.csv             winner per case and the rule check
  results/step2/test_results_chosen.csv       test set: R2 (log10|I|), median and p95 relative error
  results/step2/plot_pred_vs_spice_<case>.png predicted vs SPICE, test set, one figure per case
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml_lib as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step2"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOR = {"nmos": "#2a78d6", "pmos": "#eb6834"}
LABEL = {"id": "Id", "ig": "Ig", "is": "Is", "ib": "Ib"}


def choose():
    s = pd.read_csv(OUT / "model_comparison_summary.csv")
    rows = []
    for case, g in s.groupby("case", sort=False):
        r = dict(case=case)
        for rule, col in [("median_avg", "median_avg"), ("p95_avg", "p95_avg"), ("p95_worst", "p95_worst")]:
            r[f"winner_by_{rule}"] = g.loc[g[col].idxmin(), "model_type"]
        r["winner_by_r2_log_avg"] = g.loc[g["r2_log_avg"].idxmax(), "model_type"]
        r["chosen"] = r["winner_by_median_avg"]
        r["all_rules_agree"] = len({r[k] for k in r if k.startswith("winner_by_")}) == 1
        rows.append(r)
    return pd.DataFrame(rows)


def _extra_member(case, mtype, hps, seed):
    warnings.filterwarnings("ignore")
    d = M.load_case(ROOT, case)
    tr, _, _ = M.split_indices(len(d), case)
    return case, seed, M.TransistorModel(case, mtype, hps, seed=seed).fit(d.iloc[tr])


def final_models(ch):
    """Chosen model per case; averaged over several random starts where configured."""
    best = json.loads((OUT / "best_hyperparameters.json").read_text())
    jobs = [(r.case, r.chosen, best[r.case][r.chosen], s) for r in ch.itertuples()
            for s in range(1, M.N_AVERAGE.get(r.case, 1))]
    extra = Parallel(n_jobs=M.n_jobs(0.3), verbose=5)(delayed(_extra_member)(*j) for j in jobs) if jobs else []
    models = {}
    for r in ch.itertuples():
        base = joblib.load(ROOT / "models" / "all" / f"{r.case}__{r.chosen}.joblib")
        members = [base] + [m for c, s, m in sorted(extra, key=lambda x: x[1]) if c == r.case]
        models[r.case] = M.AveragedModel(members) if len(members) > 1 else base
        joblib.dump(models[r.case], ROOT / "models" / f"{r.case}.joblib", compress=3)
    return models


def plot_case(case, pred, true):
    kind = case.split("_")[0]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4))
    fig.patch.set_facecolor(SURF)
    for ax, j, c in zip(axes.ravel(), range(4), M.CURRENTS):
        t, p = np.abs(true[:, j]), np.abs(pred[:, j])
        wrong_sign = np.sign(pred[:, j]) != np.sign(true[:, j])
        lo, hi = min(t.min(), p.min()) / 2, max(t.max(), p.max()) * 2
        ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1, linestyle="--", label="y = x")
        ax.scatter(t[~wrong_sign], p[~wrong_sign], s=6, color=COLOR[kind], alpha=0.5, linewidths=0,
                   label="test samples")
        if wrong_sign.any():
            ax.scatter(t[wrong_sign], p[wrong_sign], s=14, facecolors="none", edgecolors=INK,
                       linewidths=0.8, label=f"wrong sign ({wrong_sign.sum()})")
        r = M.rel_err(pred[:, j], true[:, j])
        ax.text(0.03, 0.97, f"median {100*np.median(r):.2f}%\np95 {100*np.percentile(r, 95):.2f}%",
                transform=ax.transAxes, va="top",
                fontsize=8, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_facecolor(SURF); ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK2); ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_title(LABEL[c], fontsize=10, color=INK, loc="left")
        ax.set_xlabel(f"SPICE |{LABEL[c]}| (A)", fontsize=9, color=INK2)
        ax.set_ylabel(f"ML |{LABEL[c]}| (A)", fontsize=9, color=INK2)
        ax.legend(fontsize=7, frameon=False, loc="lower right", markerscale=2)
    fig.suptitle(f"{case.replace('_', ' ').upper()}: ML vs SPICE, test set (n={len(true)})",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / f"plot_pred_vs_spice_{case}.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    ch = choose()
    ch.to_csv(OUT / "chosen_models.csv", index=False)
    print(ch.to_string(index=False))
    models = final_models(ch)
    rows = []
    for _, r in ch.iterrows():
        case, mtype = r["case"], r["chosen"]
        model = models[case]
        n_avg = len(model.members) if isinstance(model, M.AveragedModel) else 1
        d = M.load_case(ROOT, case)
        _, _, te = M.split_indices(len(d), case)
        dt = d.iloc[te]
        pred, true = model.predict(dt), dt[M.CURRENTS].to_numpy()
        for c, v in M.metrics(pred, true).items():
            rows.append(dict(case=case, model_type=mtype, networks_averaged=n_avg, current=c,
                             r2_log=v["r2_log"], median=v["median"], p95=v["p95"],
                             wrong_sign=int((np.sign(pred[:, M.CURRENTS.index(c)])
                                             != np.sign(true[:, M.CURRENTS.index(c)])).sum())))
        plot_case(case, pred, true)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "test_results_chosen.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nTest set (chosen models)\n", t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
