"""Shared helpers: model-card variation, single-FET netlists, batched ngspice runs.

All netlists include the options required by the project brief:
    .options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9
Currents are reported as flowing INTO each device terminal (positive = into device).

Approved netlist conventions (see decisions log):
  * model file 22nm_HP.pm is used unmodified; per-sample process values are written
    into a copy of the model card (vth0, toxe, ndep, u0, xj only; toxp/toxm/toxref fixed)
  * every transistor line sets the RF-only switches rgatemod=0 rbodymod=0 (BSIM4 instance
    parameters, manual Ch. 9) to remove the ~1e-16 A measurement floor; DC unchanged
  * junctions: AD = AS = W*Ldiff, PD = PS = 2*(W + Ldiff), Ldiff = 44 nm (permod=1)
"""
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "model" / "22nm_HP.pm"
NGSPICE = os.environ.get("NGSPICE", shutil.which("ngspice") or "/opt/ngspice-45.2/bin/ngspice")

VDD_NOM = 0.9
LMIN = 22e-9
LDIFF = 2 * LMIN  # 44 nm, used only if junction areas are enabled
OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
PROC_PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]


def _split_cards(text):
    """Return {'nmos': card_text, 'pmos': card_text} from the PTM file."""
    parts = re.split(r"(?im)^(?=\.model)", text)
    cards = {}
    for p in parts:
        m = re.match(r"(?i)\.model\s+(\w+)\s+(nmos|pmos)", p)
        if m:
            cards[m.group(2).lower()] = p.strip()
    return cards


CARDS = _split_cards(MODEL_FILE.read_text())


def nominal(kind):
    card = CARDS[kind]
    return {p: float(re.search(rf"(?i)\b{p}\s*=\s*([-+0-9.eE]+)", card).group(1)) for p in PROC_PARAMS}


def model_card(kind, name, params):
    """Copy of the nmos/pmos card with a new model name and substituted process params."""
    card = re.sub(rf"(?i)^\.model\s+\w+\s+{kind}", f".model {name} {kind}", CARDS[kind], count=1)
    for p, v in params.items():
        card, n = re.subn(rf"(?i)\b({p}\s*=\s*)[-+0-9.eE]+", lambda m: f"{m.group(1)}{v:.10g}", card, count=1)
        assert n == 1, p
    return card


def device_suffix(w):
    """Instance parameters appended to every transistor line (single FET, gates, circuit)."""
    a, pr = w * LDIFF, 2 * (w + LDIFF)
    return f" rgatemod=0 rbodymod=0 AD={a:.10g} AS={a:.10g} PD={pr:.10g} PS={pr:.10g}"


def fet_netlist(kind, s, junctions=True):
    """Single-transistor netlist; every terminal has its own voltage source.
    s: dict with vd, vg, vs, vb, w, temp, and process params."""
    params = {p: s[p] for p in PROC_PARAMS}
    assert junctions, "approved setup always uses junction areas/perimeters"
    geo = device_suffix(s["w"])
    return f"""* single {kind.upper()}
{OPTIONS}
.temp {s['temp']:.10g}
{model_card(kind, 'mdev', params)}
M1 d g s b mdev W={s['w']:.10g} L={LMIN:.10g}{geo}
Vd d 0 {s['vd']:.10g}
Vg g 0 {s['vg']:.10g}
Vs s 0 {s['vs']:.10g}
Vb b 0 {s['vb']:.10g}
.end
"""


def _run_batch(args):
    """Run many single-FET netlists in ONE ngspice process (source / op / remcirc)."""
    netlists, = args
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ctl = ["* batch", ".control", "set numdgt=15"]
        for i, nl in enumerate(netlists):
            f = td / f"s{i}.cir"
            f.write_text(nl)
            ctl += [f"source {f}", "op",
                    f"echo RES {i}", "print i(vd) i(vg) i(vs) i(vb)", "remcirc"]
        ctl += ["quit", ".endc", ".end"]
        (td / "run.cir").write_text("\n".join(ctl) + "\n")
        r = subprocess.run([NGSPICE, "-b", str(td / "run.cir")], capture_output=True, text=True)
    out = np.full((len(netlists), 4), np.nan)
    idx = {"i(vd)": 0, "i(vg)": 1, "i(vs)": 2, "i(vb)": 3}
    cur = None
    for line in r.stdout.splitlines():
        if line.startswith("RES "):
            cur = int(line.split()[1])
        elif cur is not None and "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            if k in idx:
                # ngspice reports source branch current flowing from the circuit into the
                # source's + node; the current INTO the device terminal is its negative.
                out[cur, idx[k]] = -float(v)
    return out


def simulate_fets(kind, samples, junctions=True, batch=250, workers=None):
    """samples: list of dicts. Returns array (N,4) of [Id, Ig, Is, Ib] into the device."""
    nls = [fet_netlist(kind, s, junctions) for s in samples]
    chunks = [(nls[i:i + batch],) for i in range(0, len(nls), batch)]
    workers = workers or os.cpu_count()
    with ThreadPoolExecutor(workers) as ex:  # ngspice runs as subprocesses; threads suffice
        res = list(ex.map(_run_batch, chunks))
    return np.vstack(res)


CASES = {  # gate / body voltage as a function of VDD
    "nmos_on": ("nmos", lambda v: v, lambda v: 0.0),
    "nmos_off": ("nmos", lambda v: 0.0, lambda v: 0.0),
    "pmos_on": ("pmos", lambda v: 0.0, lambda v: v),
    "pmos_off": ("pmos", lambda v: v, lambda v: v),
}


RAIL_FRACTION = {"pmos_off": 0.60}    # share of rail-type samples per case (TA-approved; others uniform)


def rail_place(s, rng):
    """Rail-type bias for an OFF transistor (overwrites vd, vs):
    75% full rail - one end within 50 mV of 0 V, the other within 50 mV of VDD;
    25% stack     - one end within 50 mV of a rail, the other uniform in [0, VDD]."""
    v = s["vdd"]
    if rng.random() < 0.75:
        a, b = rng.uniform(0, 0.05), v - rng.uniform(0, 0.05)
    else:
        a = rng.uniform(0, 0.05) if rng.random() < 0.5 else v - rng.uniform(0, 0.05)
        b = rng.uniform(0, v)
    if rng.random() < 0.5:
        a, b = b, a
    s["vd"], s["vs"] = float(np.clip(a, 0, v)), float(np.clip(b, 0, v))
    return s


def draw_samples(case, n, rng):
    """Draw n samples per the brief, Section 4 (VDD drawn first)."""
    kind, vg_f, vb_f = CASES[case]
    nom = nominal(kind)
    out = []
    for _ in range(n):
        vdd = rng.uniform(0.9 * VDD_NOM, 1.1 * VDD_NOM)
        s = dict(vdd=vdd, temp=rng.uniform(-40, 125), w=rng.uniform(1 * LMIN, 8 * LMIN),
                 vd=rng.uniform(0, vdd), vs=rng.uniform(0, vdd), vg=vg_f(vdd), vb=vb_f(vdd))
        for p in PROC_PARAMS:
            s[p] = nom[p] * rng.uniform(0.9, 1.1)
        out.append(s)
    return out
