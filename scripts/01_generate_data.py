"""Step 1 - generate the four single-transistor datasets (NMOS/PMOS x ON/OFF).

Run from the project root:   python scripts/01_generate_data.py
Every parameter is drawn as in Section 4 of the brief. PMOS OFF only (TA-approved): 60% of its
samples get rail-type drain/source voltages (spice_lib.rail_place), so that the model learns the
full-rail bias an OFF PMOS holds inside gates; all other parameters are drawn as usual.
Outputs:
  data/<case>.csv                       10,000 rows each; columns as required by the brief
  netlists/transistor/<case>_example.cir  first sample of each case, runnable on its own:
                                          ngspice -b netlists/transistor/nmos_on_example.cir
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spice_lib as L  # noqa: E402

SEED = 20261114          # fixed master seed; each case gets its own independent stream
N_SAMPLES = 10_000
COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
           "vth0", "toxe", "ndep", "u0", "xj", "id", "ig", "is", "ib"]


def main():
    data_dir = L.ROOT / "data"
    nl_dir = L.ROOT / "netlists" / "transistor"
    data_dir.mkdir(exist_ok=True)
    nl_dir.mkdir(parents=True, exist_ok=True)
    print(f"ngspice: {L.NGSPICE}")

    for k, (case, (kind, _, _)) in enumerate(L.CASES.items()):
        rng = np.random.default_rng([SEED, k])
        samples = L.draw_samples(case, N_SAMPLES, rng)
        if case in L.RAIL_FRACTION:
            rr = np.random.default_rng([SEED, k, 1])            # separate stream for the rail placement
            samples = [L.rail_place(s, rr) if rr.random() < L.RAIL_FRACTION[case] else s for s in samples]
        t0 = time.time()
        cur = L.simulate_fets(kind, samples)
        n_bad = int(np.isnan(cur).any(axis=1).sum())
        if n_bad:
            raise RuntimeError(f"{case}: {n_bad} samples failed to simulate")

        df = pd.DataFrame(samples)
        df[["id", "ig", "is", "ib"]] = cur
        df = df[COLUMNS]
        df.to_csv(data_dir / f"{case}.csv", index=False, float_format="%.15g")

        ex = L.fet_netlist(kind, samples[0]).replace(
            ".end\n",
            ".control\nset numdgt=15\nop\nprint i(vd) i(vg) i(vs) i(vb)\n.endc\n.end\n")
        header = (f"* Example netlist, case {case} (sample 0 of data/{case}.csv)\n"
                  "* Printed i(Vx) is the source branch current; current INTO the device\n"
                  "* terminal = -i(Vx), which is what the CSV stores.\n")
        (nl_dir / f"{case}_example.cir").write_text(header + ex)
        print(f"{case:9s}: {N_SAMPLES} samples in {time.time() - t0:5.1f} s -> data/{case}.csv")


if __name__ == "__main__":
    main()
