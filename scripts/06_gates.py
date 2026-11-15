"""Step 3 - gate-level validation: INV, NAND2, NOR2, AND2, OR2, every input combination, 200 corners.

Run from the project root (after 05_select_and_test.py):   python scripts/06_gates.py
SPICE (reference, evaluation only): each gate is simulated as a transistor-level netlist; the
internal node voltages and the output voltage are recorded together with the leakage.
ML: every transistor's terminal voltages are taken from those recorded SPICE node voltages and fed
to the four saved transistor models (models/<case>.joblib); the gate leakage is the sum of the
predicted currents into terminals on the VDD-side group (see gate_lib.py).
Outputs:
  netlists/gates/<gate>.cir             one SPICE netlist per gate (corner 0, inputs all 0)
  results/step3/corners.csv             the 200 corners (reused by the test circuit)
  results/step3/gate_simulations.csv    every gate x input x corner: SPICE and ML leakage, node voltages
  results/step3/gate_results_table.csv  per gate and input: SPICE median, ML median, bias (median of
                                        the signed error), median error, 95th-percentile error
  results/step3/plot_gate_parity.png    ML vs SPICE leakage, all gate simulations, log axes
  results/step3/plot_gate_parity_by_input.png  one panel per gate, coloured by input combination
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_lib as G  # noqa: E402
import ml_lib as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step3"
MODELS = Path(os.environ.get("DVD_MODELS", ROOT / "models"))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
INPUT_COLOR = {"00": "#2a78d6", "01": "#eb6834", "10": "#1baf7a", "11": "#eda100", "0": "#2a78d6", "1": "#eda100"}
GATE_COLOR = {"INV": "#2a78d6", "NAND2": "#eb6834", "NOR2": "#1baf7a", "AND2": "#eda100", "OR2": "#e87ba4"}


def write_example_netlists(corners):
    d = ROOT / "netlists" / "gates"
    d.mkdir(parents=True, exist_ok=True)
    c = corners.iloc[0]
    for g, (inputs, devs, internal) in G.GATES.items():
        bits = (0,) * len(inputs)
        nl = G.netlist(devs, inputs, bits, c, title=f"{g}, corner 0, inputs {''.join(map(str, bits))}")
        prints = " ".join([f"v({n})" for n in internal + ["out"]] + ["i(vdd)"] + [f"i(v{n})" for n in inputs])
        nl = nl.replace(".end\n", f".control\nset numdgt=15\nop\nprint {prints}\n.endc\n.end\n")
        head = ("* Leakage = -i(vdd) - i(v<input>) for every input held at VDD.\n"
                "* Change the input sources (0 or VDD) for the other input combinations.\n")
        (d / f"{g}.cir").write_text(head + nl)


def _style(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2); ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_by_input(sim):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    fig.patch.set_facecolor(SURF)
    axes = axes.ravel()
    for ax, g in zip(axes, G.GATES):
        d = sim[sim.gate == g]
        lo = min(d.spice_leakage.min(), d.ml_leakage.min()) / 1.5
        hi = max(d.spice_leakage.max(), d.ml_leakage.max()) * 1.5
        ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1, linestyle="--", label="y = x")
        for inp, di in d.groupby("inputs", sort=True):
            r = di.rel_error.abs()
            ax.scatter(di.spice_leakage, di.ml_leakage, s=8, color=INPUT_COLOR[inp], alpha=0.6, linewidths=0,
                       label=f"{inp}: median {100*r.median():.2f}%, p95 {100*r.quantile(0.95):.2f}%")
        _style(ax); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_title(g, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("SPICE leakage (A)", fontsize=9, color=INK2)
        ax.set_ylabel("ML leakage (A)", fontsize=9, color=INK2)
        ax.legend(fontsize=7, frameon=False, loc="upper left", markerscale=2)
    axes[-1].axis("off")
    fig.suptitle("Gate leakage: ML vs SPICE by input combination (200 corners each)", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "plot_gate_parity_by_input.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    corners = G.draw_corners()
    corners.to_csv(OUT / "corners.csv", index=False)
    write_example_netlists(corners)

    jobs, nls, prints = [], [], []
    for g, (inputs, devs, internal) in G.GATES.items():
        for bits in G.input_vectors(len(inputs)):
            for _, c in corners.iterrows():
                jobs.append((g, bits, c))
                nls.append(G.netlist(devs, inputs, bits, c))
                prints.append(["i(vdd)"] + [f"i(v{n})" for n in inputs] + [f"v({n})" for n in internal + ["out"]])
    vals = G.run_spice(nls, prints)
    print(f"SPICE: {len(nls)} simulations in {time.time() - t0:.0f} s")

    models = {case: joblib.load(MODELS / f"{case}.joblib") for case in M.CASES}
    rows, all_dev, spans = [], [], []
    for (g, bits, c), v in zip(jobs, vals):
        inputs, devs, internal = G.GATES[g]
        node_v = {n: v[f"v({n})"] for n in internal + ["out"]}
        dev = G.device_rows(devs, inputs, bits, c, node_v)
        spans.append((len(all_dev), len(all_dev) + len(dev)))
        all_dev += dev
        r = dict(gate=g, inputs="".join(map(str, bits)), corner=int(c["corner"]), vdd=c["vdd"], temp=c["temp"],
                 spice_leakage=G.spice_leakage(v, inputs, bits), v_out_spice=node_v["out"])
        for n in internal:
            r[f"v_{n}_spice"] = node_v[n]
            r[f"v_{n}_ml"] = node_v[n]          # ML uses the recorded SPICE node voltages (brief, step 4)
        rows.append(r)
    cur = G.predict_devices(models, all_dev)
    for r, (a, b) in zip(rows, spans):
        r["ml_leakage"] = G.group_leakage(all_dev[a:b], cur[a:b])
    sim = pd.DataFrame(rows)
    sim["rel_error"] = (sim.ml_leakage - sim.spice_leakage) / sim.spice_leakage
    sim.to_csv(OUT / "gate_simulations.csv", index=False)

    tab = []
    for (g, i), d in sim.groupby(["gate", "inputs"], sort=False):
        sm, mm = d.spice_leakage.median(), d.ml_leakage.median()
        tab.append(dict(gate=g, inputs=i, spice_median_A=sm, ml_median_A=mm,
                        bias=d.rel_error.median(),
                        median_error=d.rel_error.abs().median(),
                        p95_error=d.rel_error.abs().quantile(0.95)))
    tab = pd.DataFrame(tab)
    tab.to_csv(OUT / "gate_results_table.csv", index=False)
    show = tab.copy()
    for col in ["bias", "median_error", "p95_error"]:
        show[col] = (100 * show[col]).map(lambda x: f"{x:+.2f}%" if "bias" in col else f"{x:.2f}%")
    pd.set_option("display.width", 200)
    print(show.to_string(index=False, formatters={"spice_median_A": "{:.3e}".format, "ml_median_A": "{:.3e}".format}))

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    fig.patch.set_facecolor(SURF)
    lo = min(sim.spice_leakage.min(), sim.ml_leakage.min()) / 1.5
    hi = max(sim.spice_leakage.max(), sim.ml_leakage.max()) * 1.5
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1, linestyle="--", label="y = x")
    for g in G.GATES:
        d = sim[sim.gate == g]
        ax.scatter(d.spice_leakage, d.ml_leakage, s=7, color=GATE_COLOR[g], alpha=0.55, linewidths=0,
                   label=f"{g} ({len(d)})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2); ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel("SPICE leakage (A)", fontsize=9, color=INK2)
    ax.set_ylabel("ML leakage (A)", fontsize=9, color=INK2)
    r = sim.rel_error.abs()
    ax.text(0.03, 0.97, f"all {len(sim)} simulations\nmedian error {100*r.median():.2f}%\n"
            f"p95 error {100*r.quantile(0.95):.2f}%", transform=ax.transAxes, va="top", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right", markerscale=2.5)
    ax.set_title("Gate leakage: ML vs SPICE (5 gates, all inputs, 200 corners)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "plot_gate_parity.png", dpi=160, facecolor=SURF)
    plt.close(fig)
    plot_by_input(sim)
    print(f"done in {time.time() - t0:.0f} s; outputs in {OUT}")


if __name__ == "__main__":
    main()
