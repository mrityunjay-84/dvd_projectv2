"""Step 1b - data check for the four datasets.

Run from the project root:   python scripts/02_data_check.py
Outputs (results/step1/) - deliverable 3 of the brief:
  kirchhoff_check.csv                 per case: |Id+Ig+Is+Ib|, absolute and relative
  plot_off_current_vs_temperature.png OFF-state |Id| vs temperature
  plot_off_current_vs_vth0.png        OFF-state |Id| vs vth0
  plot_gate_current_vs_toxe.png       |Ig| vs toxe, all four cases
OFF-current plots: all samples in grey; samples with |Vds| >= VDD/2 (the usual OFF-current
condition) in colour with their binned median. Console only (no files): sanity checks
(NaN, exact zeros, resolution-floor pattern) and a range check of every input (Section 4).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spice_lib as L  # noqa: E402

OUT = L.ROOT / "results" / "step1"
CUR = ["id", "ig", "is", "ib"]
QSTEP = 5.5511151231257827e-17          # ngspice current resolution seen with rgatemod/rbodymod=1

# reference palette (dataviz skill): entity -> fixed colour
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOR = {"nmos": "#2a78d6", "pmos": "#eb6834"}


def load():
    return {c: pd.read_csv(L.ROOT / "data" / f"{c}.csv") for c in L.CASES}


def kirchhoff(dfs):
    rows = []
    for c, d in dfs.items():
        I = d[CUR].to_numpy()
        s = np.abs(I.sum(1))
        rel = s / np.abs(I).max(1)
        rows.append(dict(case=c, n=len(d), max_abs_sum_A=s.max(), median_abs_sum_A=np.median(s),
                         max_rel_to_largest_terminal=rel.max(), median_rel=np.median(rel)))
    return pd.DataFrame(rows)


def sanity(dfs):
    rows = []
    for c, d in dfs.items():
        for k in CUR:
            v = d[k].to_numpy(); a = np.abs(v); small = a < 1e-14
            on_grid = (np.isclose(a[small] / QSTEP, np.round(a[small] / QSTEP), atol=1e-3).mean()
                       if small.any() else 0.0)
            rows.append(dict(case=c, current=k, nan=int(np.isnan(v).sum()), exact_zeros=int((v == 0).sum()),
                             frac_below_1em15=(a < 1e-15).mean(), frac_below_1em14=small.mean(),
                             frac_of_those_on_floor_grid=on_grid,
                             min_abs=a.min(), median_abs=np.median(a), max_abs=a.max(),
                             frac_positive=(v > 0).mean()))
    return pd.DataFrame(rows)


def ranges(dfs):
    rows = []
    for c, d in dfs.items():
        kind, vg_f, vb_f = L.CASES[c]
        nom = L.nominal(kind)
        v = d["vdd"]
        checks = {
            "vdd in [0.81, 0.99]": v.between(0.9 * L.VDD_NOM, 1.1 * L.VDD_NOM),
            "temp in [-40, 125]": d["temp"].between(-40, 125),
            "w in [22n, 176n]": d["w"].between(L.LMIN * (1 - 1e-12), 8 * L.LMIN * (1 + 1e-12)),
            "vd in [0, vdd]": (d["vd"] >= 0) & (d["vd"] <= v),
            "vs in [0, vdd]": (d["vs"] >= 0) & (d["vs"] <= v),
            "vg as per case": np.isclose(d["vg"], vg_f(v)),
            "vb as per case": np.isclose(d["vb"], vb_f(v)),
        }
        for p in L.PROC_PARAMS:
            lo, hi = sorted([0.9 * nom[p], 1.1 * nom[p]])
            checks[f"{p} in nominal +/-10%"] = d[p].between(lo, hi)
        for name, ok in checks.items():
            rows.append(dict(case=c, check=name, passed=bool(ok.all()), n_fail=int((~ok).sum())))
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8)


def _binned_median(x, y, nb=20):
    e = np.linspace(x.min(), x.max(), nb + 1)
    idx = np.clip(np.digitize(x, e) - 1, 0, nb - 1)
    xm = 0.5 * (e[1:] + e[:-1])
    ym = np.array([np.median(y[idx == i]) if (idx == i).any() else np.nan for i in range(nb)])
    return xm, ym


def trend_plot(dfs, cases, xcol, ycol, xlabel, ylabel, xscale, title, fname, highlight_high_vds=False):
    if len(cases) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    else:
        fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 3.8), squeeze=False)
    axes = axes.ravel()
    fig.patch.set_facecolor(SURF)
    for ax, c in zip(axes, cases):
        d = dfs[c]; kind = L.CASES[c][0]
        x = d[xcol].to_numpy() * xscale; y = np.abs(d[ycol].to_numpy())
        ok = y > 0
        if highlight_high_vds:
            hi = ok & (np.abs(d["vd"] - d["vs"]).to_numpy() >= 0.5 * d["vdd"].to_numpy())
            ax.scatter(x[ok & ~hi], y[ok & ~hi], s=3, color="#b4b2a9", alpha=0.35, linewidths=0,
                       label="all other samples")
            ax.scatter(x[hi], y[hi], s=4, color=COLOR[kind], alpha=0.55, linewidths=0,
                       label="|Vds| \u2265 VDD/2")
            xm, ym = _binned_median(x[hi], y[hi])
            ax.plot(xm, ym, color=INK, linewidth=2, label="median, |Vds| \u2265 VDD/2")
        else:
            ax.scatter(x[ok], y[ok], s=3, color=COLOR[kind], alpha=0.35, linewidths=0, label="samples")
            xm, ym = _binned_median(x[ok], y[ok])
            ax.plot(xm, ym, color=INK, linewidth=2, label="binned median")
        ax.set_yscale("log")
        _style(ax)
        ax.set_title(c.replace("_", " ").upper(), fontsize=10, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=9, color=INK2); ax.set_ylabel(ylabel, fontsize=9, color=INK2)
        if (~ok).any():
            ax.text(0.99, 0.02, f"{(~ok).sum()} exact zeros not shown", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, loc="best", markerscale=4)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dfs = load()
    k = kirchhoff(dfs); s = sanity(dfs); r = ranges(dfs)
    k.to_csv(OUT / "kirchhoff_check.csv", index=False)
    pd.set_option("display.width", 200)
    print("Kirchhoff check\n", k.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
    print("\nData sanity\n", s.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nRange check: {int(r.passed.sum())}/{len(r)} checks passed;",
          "failures:" if (~r.passed).any() else "no failures", r[~r.passed].to_string(index=False) if (~r.passed).any() else "")

    trend_plot(dfs, ["nmos_off", "pmos_off"], "temp", "id", "Temperature (°C)", "|Id| (A)", 1,
               "OFF-state drain current vs temperature", "plot_off_current_vs_temperature.png",
               highlight_high_vds=True)
    trend_plot(dfs, ["nmos_off", "pmos_off"], "vth0", "id", "vth0 (V)", "|Id| (A)", 1,
               "OFF-state drain current vs vth0", "plot_off_current_vs_vth0.png",
               highlight_high_vds=True)
    trend_plot(dfs, list(L.CASES), "toxe", "ig", "toxe (nm)", "|Ig| (A)", 1e9,
               "Gate current vs toxe", "plot_gate_current_vs_toxe.png")
    print("\nplots written to", OUT)


if __name__ == "__main__":
    main()
