"""Step 4 - test circuit: 8-bit parity checker, every input vector (256) at the same 200 corners.

Run from the project root (after 06_gates.py):   python scripts/07_test_circuit.py
SPICE (reference): the complete transistor-level netlist (7 XORs = 28 NAND2, 112 transistors).
Leakage = current delivered by VDD plus by every input held at VDD (same definition as the gates).
ML (hierarchical, as in the brief): for each input vector the logic value at the inputs of every
NAND2 is found, the NAND2 gate model of step 3 is called for that input combination and corner
(ML leakage from results/step3/gate_simulations.csv) and the 28 gate leakages are summed. This
assumes every gate input sits at exactly 0 or VDD and ignores loading between gates.
Diagnostic: the same sum using the SPICE gate leakages separates ML error from composition error.
Outputs:
  netlists/circuit/parity8.cir                 complete netlist (corner 0, all inputs 0)
  results/step4/circuit_simulations.csv        every vector x corner: SPICE, ML, sum of SPICE gate leakages,
                                               output voltage, largest gate-output deviation from its rail
  results/step4/circuit_results_table.csv      SPICE median, ML median, bias (median signed error),
                                               median error, 95th-percentile error
  results/step4/plot_circuit_parity.png        ML vs SPICE leakage, log axes
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_lib as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step4"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOR = "#2a78d6"


def vectors():
    return [tuple((v >> k) & 1 for k in range(8)) for v in range(256)]   # bit k drives input i<k>


def write_netlist(ins, devs, c):
    d = ROOT / "netlists" / "circuit"
    d.mkdir(parents=True, exist_ok=True)
    nl = G.netlist(devs, ins, (0,) * 8, c, title="8-bit parity checker, corner 0, inputs all 0")
    nl = nl.replace(".end\n", ".control\nset numdgt=15\nop\nprint v(out) i(vdd) "
                    + " ".join(f"i(v{n})" for n in ins) + "\n.endc\n.end\n")
    (d / "parity8.cir").write_text("* Leakage = -i(vdd) - i(v<input>) for every input held at VDD.\n" + nl)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ins, devs, nands = G.parity8()
    corners = G.draw_corners()
    write_netlist(ins, devs, corners.iloc[0])

    gate = pd.read_csv(ROOT / "results" / "step3" / "gate_simulations.csv", dtype={"inputs": str})
    gate = gate[gate.gate == "NAND2"].set_index(["inputs", "corner"])
    nand_out = [o for _, _, _, o, _ in nands]

    logic_of = {bits: G.switch_level(devs, ins, bits)[0] for bits in vectors()}
    prints = ["i(vdd)"] + [f"i(v{n})" for n in ins] + [f"v({n})" for n in nand_out]
    rows = []
    for start in range(0, len(corners), 10):                  # 10 corners (2,560 runs) at a time
        block = corners.iloc[start:start + 10]
        jobs, nls = [], []
        for bits in vectors():
            for _, c in block.iterrows():
                jobs.append((bits, int(c["corner"])))
                nls.append(G.netlist(devs, ins, bits, c))
        vals = G.run_spice(nls, [prints] * len(nls), batch=100)
        for (bits, k), v in zip(jobs, vals):
            logic = logic_of[bits]
            combos = [f"{logic[A]}{logic[B]}" for _, A, B, _, _ in nands]   # A then B, as in step 3
            vdd = corners.loc[k, "vdd"]
            dev = max(abs(v[f"v({o})"] - logic[o] * vdd) for o in nand_out)
            g = gate.loc[[(cb, k) for cb in combos]]
            rows.append(dict(vector="".join(str(b) for b in reversed(bits)), ones=sum(bits), corner=k,
                             vdd=vdd, temp=corners.loc[k, "temp"],
                             spice_leakage=G.spice_leakage(v, ins, bits), ml_leakage=g.ml_leakage.sum(),
                             spice_gate_sum=g.spice_leakage.sum(), v_out_spice=v["v(out)"],
                             max_gate_output_deviation_V=dev))
        print(f"corners {start}-{start + len(block) - 1} done, {time.time() - t0:.0f} s", flush=True)
    sim = pd.DataFrame(rows)
    sim["rel_error"] = (sim.ml_leakage - sim.spice_leakage) / sim.spice_leakage
    sim["composition_error"] = (sim.spice_gate_sum - sim.spice_leakage) / sim.spice_leakage
    sim.to_csv(OUT / "circuit_simulations.csv", index=False)

    sm, mm = sim.spice_leakage.median(), sim.ml_leakage.median()
    tab = pd.DataFrame([dict(circuit="8-bit parity checker", simulations=len(sim),
                             spice_median_A=sm, ml_median_A=mm,
                             bias=sim.rel_error.median(),
                             median_error=sim.rel_error.abs().median(),
                             p95_error=sim.rel_error.abs().quantile(0.95))])
    tab.to_csv(OUT / "circuit_results_table.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.to_string(index=False))
    print(f"composition only (sum of SPICE gate leakages vs circuit SPICE): median "
          f"{100*sim.composition_error.abs().median():.2f}%, p95 {100*sim.composition_error.abs().quantile(0.95):.2f}%, "
          f"bias {100*sim.composition_error.median():+.2f}%")

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    fig.patch.set_facecolor(SURF)
    lo = min(sim.spice_leakage.min(), sim.ml_leakage.min()) / 1.2
    hi = max(sim.spice_leakage.max(), sim.ml_leakage.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1, linestyle="--", label="y = x")
    ax.scatter(sim.spice_leakage, sim.ml_leakage, s=3, color=COLOR, alpha=0.35, linewidths=0,
               label=f"{len(sim)} simulations (256 vectors x 200 corners)")
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
    ax.text(0.03, 0.97, f"median error {100*r.median():.2f}%\np95 error {100*r.quantile(0.95):.2f}%",
            transform=ax.transAxes, va="top", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right", markerscale=3)
    ax.set_title("8-bit parity checker leakage: hierarchical ML vs SPICE", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "plot_circuit_parity.png", dpi=160, facecolor=SURF)
    plt.close(fig)
    print(f"done in {time.time() - t0:.0f} s; outputs in {OUT}")


if __name__ == "__main__":
    main()
