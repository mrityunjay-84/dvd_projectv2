"""Shared code for gate-level (step 3) and circuit-level (step 4) leakage: topologies, corners,
SPICE netlists and runs, switch-level logic, and ML leakage via VDD-side grouping.

Conventions (approved):
  * static CMOS; NMOS W = 66 nm, PMOS W = 132 nm, L = 22 nm; every device gets the same instance
    suffix as the transistor netlists (rgatemod=0 rbodymod=0, AD/AS/PD/PS with Ldiff = 44 nm)
  * NAND2 pull-down: B next to the output, A next to ground.  NOR2 pull-up: A next to VDD, B next
    to the output (course slides).  AND2 = NAND2 + INV, OR2 = NOR2 + INV, simulated as one netlist.
  * a corner = one random draw of VDD, temperature, NMOS process values, PMOS process values
    (Section 4 distributions); all NMOS share one draw, all PMOS share another
  * SPICE leakage = current delivered by VDD + current delivered by every input held at VDD
  * ML leakage   = sum of predicted currents INTO transistor terminals that sit on the VDD-side
    group: VDD, inputs held high, and every node connected to VDD through ON transistors
    (switch level). Channel currents of ON transistors inside the group cancel exactly, because
    each transistor model satisfies Id + Ig + Is + Ib = 0.
"""
import itertools
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import spice_lib as L

WN, WP = 3 * L.LMIN, 6 * L.LMIN          # 66 nm, 132 nm
CORNER_SEED = 20261115
N_CORNERS = 200


# ----------------------------------------------------------------------------- topologies
def _inv(inp, out, tag):
    return [(f"MP{tag}", "p", out, inp, "vdd", "vdd"), (f"MN{tag}", "n", out, inp, "0", "0")]


def _nand2(a, b, out, x, tag):
    return [(f"MPA{tag}", "p", out, a, "vdd", "vdd"), (f"MPB{tag}", "p", out, b, "vdd", "vdd"),
            (f"MNB{tag}", "n", out, b, x, "0"), (f"MNA{tag}", "n", x, a, "0", "0")]


def _nor2(a, b, out, y, tag):
    return [(f"MPA{tag}", "p", y, a, "vdd", "vdd"), (f"MPB{tag}", "p", out, b, y, "vdd"),
            (f"MNA{tag}", "n", out, a, "0", "0"), (f"MNB{tag}", "n", out, b, "0", "0")]


GATES = {   # name -> (inputs, devices, internal nodes recorded)
    "INV": (["a"], _inv("a", "out", "1"), []),
    "NAND2": (["a", "b"], _nand2("a", "b", "out", "x", "1"), ["x"]),
    "NOR2": (["a", "b"], _nor2("a", "b", "out", "y", "1"), ["y"]),
    "AND2": (["a", "b"], _nand2("a", "b", "n1", "x", "1") + _inv("n1", "out", "2"), ["x", "n1"]),
    "OR2": (["a", "b"], _nor2("a", "b", "n1", "y", "1") + _inv("n1", "out", "2"), ["y", "n1"]),
}


# ----------------------------------------------------------------------------- test circuit
def _xor_nand4(a, b, out, tag):
    """XOR from four NAND2: n1 = NAND(a,b), n2 = NAND(a,n1), n3 = NAND(b,n1), out = NAND(n2,n3).
    Returns (devices, list of NAND instances as (name, A, B, out, x))."""
    n1, n2, n3 = f"{tag}n1", f"{tag}n2", f"{tag}n3"
    inst = [(f"{tag}g1", a, b, n1), (f"{tag}g2", a, n1, n2), (f"{tag}g3", b, n1, n3), (f"{tag}g4", n2, n3, out)]
    devs, nands = [], []
    for name, A, B, o in inst:
        x = f"{name}x"
        devs += _nand2(A, B, o, x, name)
        nands.append((name, A, B, o, x))
    return devs, nands


def parity8():
    """8-bit parity checker: balanced tree of 7 XORs (28 NAND2). Output = 1 for an odd number of 1s."""
    ins = [f"i{k}" for k in range(8)]
    plan = [("x0", ins[0], ins[1], "p0"), ("x1", ins[2], ins[3], "p1"), ("x2", ins[4], ins[5], "p2"),
            ("x3", ins[6], ins[7], "p3"), ("x4", "p0", "p1", "q0"), ("x5", "p2", "p3", "q1"), ("x6", "q0", "q1", "out")]
    devs, nands = [], []
    for tag, a, b, o in plan:
        d, n = _xor_nand4(a, b, o, tag)
        devs += d; nands += n
    return ins, devs, nands


# ----------------------------------------------------------------------------- corners
def draw_corners(n=N_CORNERS, seed=CORNER_SEED):
    rng = np.random.default_rng(seed)
    nom = {k: L.nominal(k) for k in ("nmos", "pmos")}
    rows = []
    for i in range(n):
        r = dict(corner=i, vdd=rng.uniform(0.9 * L.VDD_NOM, 1.1 * L.VDD_NOM), temp=rng.uniform(-40, 125))
        for k, pre in (("nmos", "n_"), ("pmos", "p_")):
            for p in L.PROC_PARAMS:
                r[pre + p] = nom[k][p] * rng.uniform(0.9, 1.1)
        rows.append(r)
    return pd.DataFrame(rows)


def corner_params(c, kind):
    pre = "n_" if kind == "nmos" else "p_"
    return {p: float(c[pre + p]) for p in L.PROC_PARAMS}


# ----------------------------------------------------------------------------- netlists
def netlist(devices, inputs, bits, c, ammeters=False, title="gate"):
    """SPICE netlist of a circuit at one corner and one input vector.
    ammeters=True inserts a 0 V source in series with every device terminal (check use only)."""
    t = [f"* {title}", L.OPTIONS, f".temp {c['temp']:.10g}",
         L.model_card("nmos", "nch", corner_params(c, "nmos")),
         L.model_card("pmos", "pch", corner_params(c, "pmos"))]
    for name, typ, d, g, s, b in devices:
        w = WN if typ == "n" else WP
        nodes = [d, g, s, b]
        if ammeters:
            nodes = []
            for term, node in zip("dgsb", (d, g, s, b)):
                nodes.append(f"{name}_{term}")
                t.append(f"VA{name}{term} {node} {name}_{term} 0")
        t.append(f"{name} {' '.join(nodes)} {'nch' if typ == 'n' else 'pch'} W={w:.10g} L={L.LMIN:.10g}"
                 + L.device_suffix(w))
    t.append(f"Vdd vdd 0 {c['vdd']:.10g}")
    for n, bit in zip(inputs, bits):
        t.append(f"V{n} {n} 0 {bit * c['vdd']:.10g}")
    return "\n".join(t) + "\n.end\n"


def run_spice(netlists, prints, batch=200, workers=None):
    """Run netlists (operating point) and return, per netlist, {vector name: value}."""
    def _batch(chunk):
        start, nls = chunk
        with tempfile.TemporaryDirectory() as td:
            ctl = ["* batch", ".control", "set numdgt=15"]
            for i, nl in enumerate(nls):
                f = Path(td) / f"s{i}.cir"
                f.write_text(nl)
                ctl += [f"source {f}", "op", f"echo RES {i}", "print " + " ".join(prints[start + i]), "remcirc"]
            ctl += ["quit", ".endc", ".end"]
            (Path(td) / "run.cir").write_text("\n".join(ctl) + "\n")
            r = subprocess.run([L.NGSPICE, "-b", str(Path(td) / "run.cir")], capture_output=True, text=True)
        out = [dict() for _ in nls]
        cur = None
        for line in r.stdout.splitlines():
            if line.startswith("RES "):
                cur = int(line.split()[1])
            elif cur is not None and " = " in line:
                k, v = [x.strip() for x in line.split("=", 1)]
                try:
                    out[cur][k.lower()] = float(v)
                except ValueError:
                    pass
        return out
    chunks = [(i, netlists[i:i + batch]) for i in range(0, len(netlists), batch)]
    with ThreadPoolExecutor(workers or os.cpu_count()) as ex:
        res = list(ex.map(_batch, chunks))
    return [r for part in res for r in part]


def spice_leakage(vals, inputs, bits):
    """Current delivered by VDD plus by every input held at VDD (A)."""
    return -vals["i(vdd)"] + sum(-vals[f"i(v{n})"] for n, bit in zip(inputs, bits) if bit)


# ----------------------------------------------------------------------------- switch level
def switch_level(devices, inputs, bits):
    """Logic value of every node (1, 0 or None) and ON/OFF of every device, by iterating
    'gate logic known -> device state known -> channel-connected groups -> node logic'."""
    logic = {"vdd": 1, "0": 0, **{n: int(bit) for n, bit in zip(inputs, bits)}}
    state = {}
    for _ in range(len(devices) + 2):
        for name, typ, d, g, s, b in devices:
            if g in logic and logic[g] is not None:
                state[name] = (logic[g] == 1) if typ == "n" else (logic[g] == 0)
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for name, typ, d, g, s, b in devices:
            find(d), find(s)
            if state.get(name):
                parent[find(d)] = find(s)
        hi, lo = find("vdd"), find("0")
        for node in list(parent):
            if node in inputs:
                continue
            root = find(node)
            logic[node] = 1 if root == hi else 0 if root == lo else None
    high = {n for n, v in logic.items() if v == 1}
    return logic, state, high


def device_rows(devices, inputs, bits, c, node_v):
    """One row per transistor with the model inputs (dataset column names) and its case."""
    logic, state, high = switch_level(devices, inputs, bits)
    v = {"vdd": c["vdd"], "0": 0.0, **{n: bit * c["vdd"] for n, bit in zip(inputs, bits)}, **node_v}
    rows = []
    for name, typ, d, g, s, b in devices:
        kind = "nmos" if typ == "n" else "pmos"
        r = dict(device=name, case=f"{kind}_{'on' if state[name] else 'off'}",
                 vdd=c["vdd"], temp=c["temp"], w=WN if typ == "n" else WP,
                 vd=v[d], vg=v[g], vs=v[s], vb=v[b],
                 in_high_d=d in high, in_high_g=g in high, in_high_s=s in high, in_high_b=b in high)
        r.update(corner_params(c, kind))
        rows.append(r)
    return rows


def group_leakage(rows, currents):
    """Sum of currents INTO device terminals on the VDD-side group. currents: (N, 4) Id, Ig, Is, Ib."""
    mask = np.array([[r["in_high_d"], r["in_high_g"], r["in_high_s"], r["in_high_b"]] for r in rows])
    return float((currents * mask).sum())


def predict_devices(models, rows):
    """Predict Id, Ig, Is, Ib for a list of device rows, using the case model of each row."""
    df = pd.DataFrame(rows)
    out = np.zeros((len(df), 4))
    for case, idx in df.groupby("case").groups.items():
        out[np.asarray(idx)] = models[case].predict(df.loc[idx])
    return out


def input_vectors(n_inputs):
    return [tuple(b) for b in itertools.product([0, 1], repeat=n_inputs)]
