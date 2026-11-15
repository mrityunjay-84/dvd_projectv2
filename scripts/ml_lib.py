"""Shared ML code: data split, features, physics-structured targets, model wrappers, metrics.

Physics-structured transistor model (one per case and model type)
-----------------------------------------------------------------
Inputs  : the dataset columns only (vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj).
Outputs : Id, Ig, Is, Ib (A, positive = into the device terminal).
Inside  : three regressors of the same model type
            r_g : log10|Ig|                      (Ig never changes sign within a case)
            r_b : log10|Ib|                      (Ib never changes sign within a case)
            r_c : log10 of the channel part  Ich = (I_hi - I_lo)/2 > 0, either as is ("plain")
                  or divided by |Vds| ("conductance", smooth as Vds -> 0); the form is a
                  tuning choice made on the validation set
          where hi/lo are the source/drain terminals at the higher/lower voltage (the device
          is drain/source symmetric, verified in SPICE to 1e-14).
          Prediction only: the log|Vds| input is held at VDS_FLOOR (4 mV, the 1st percentile of
          training |Vds|) when |Vds| is smaller. Inside gates, ON transistors sit at uV-nV, far below
          the training range; gate and body currents are flat in Vds there. Training is unclipped.
          Rebuild (Kirchhoff: Id + Is = -(Ig + Ib)):
            I_hi =  Ich - (Ig + Ib)/2 ,  I_lo = -Ich - (Ig + Ib)/2 ,  then map hi/lo back to d/s.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import xgboost as xgb

SPLIT_SEED = 20261114
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
INPUTS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
CURRENTS = ["id", "ig", "is", "ib"]
PARTS = ["g", "b", "c"]
MODEL_TYPES = ["poly_ridge", "random_forest", "xgboost", "mlp"]
VDS_FLOOR = 4e-3        # V, floor of the log|Vds| input at prediction time
N_AVERAGE = {"nmos_off": 3, "pmos_off": 3}   # final chosen model = average of this many networks


# ----------------------------------------------------------------------------- data
def split_indices(n, case):
    """Fixed random 80/10/10 split (train/validation/test) per case."""
    k = CASES.index(case)
    p = np.random.default_rng([SPLIT_SEED, 100 + k]).permutation(n)
    a, b = int(0.8 * n), int(0.9 * n)
    return p[:a], p[a:b], p[b:]


def orient(df):
    """hi/lo terminal voltages and the boolean 'drain is the hi terminal'."""
    up = (df["vd"] >= df["vs"]).to_numpy()
    vhi = np.where(up, df["vd"], df["vs"])
    vlo = np.where(up, df["vs"], df["vd"])
    return up, vhi, vlo


def features(df, vds_floor=None):
    """Model inputs, built from the dataset columns only (symmetric hi/lo frame).
    vds_floor: if given, the log|Vds| input uses max(|Vds|, vds_floor) (prediction only)."""
    up, vhi, vlo = orient(df)
    vg, vb = df["vg"].to_numpy(), df["vb"].to_numpy()
    T = df["temp"].to_numpy() + 273.15
    return np.column_stack([
        df["vdd"], T, 1.0 / T, np.log(df["w"]),
        vhi, vlo, vhi - vlo, np.log(np.maximum(vhi - vlo, vds_floor or 1e-12)),
        vg - vlo, vg - vhi, vb - vlo, vb - vhi,
        df["vth0"], df["toxe"] * 1e9, df["ndep"] / 1e18, df["u0"], df["xj"] * 1e9,
    ])


def targets(df):
    """Training targets for the three regressors, plus the constant signs of Ig and Ib."""
    up, vhi, vlo = orient(df)
    i_d, i_s = df["id"].to_numpy(), df["is"].to_numpy()
    i_hi, i_lo = np.where(up, i_d, i_s), np.where(up, i_s, i_d)
    ich = (i_hi - i_lo) / 2.0
    if not (ich > 0).all():
        raise ValueError("channel part must be positive in the hi/lo frame")
    sg, sb = np.sign(df["ig"]).unique(), np.sign(df["ib"]).unique()
    if len(sg) != 1 or len(sb) != 1:
        raise ValueError("Ig and Ib must keep one sign within a case")
    y = {"g": np.log10(np.abs(df["ig"].to_numpy())),
         "b": np.log10(np.abs(df["ib"].to_numpy())),
         "c_plain": np.log10(ich),
         "c_conductance": np.log10(ich / (vhi - vlo))}
    return y, {"g": float(sg[0]), "b": float(sb[0])}


def channel_target(y, form):
    return y["c_" + form]


def channel_current(df, pred_c, form):
    """Undo the channel-part transform: returns Ich (A)."""
    up, vhi, vlo = orient(df)
    return 10.0 ** pred_c * ((vhi - vlo) if form == "conductance" else 1.0)


def rebuild(df, pred, signs, c_form):
    """Turn the three regressor outputs into Id, Ig, Is, Ib (A)."""
    up, vhi, vlo = orient(df)
    ig = signs["g"] * 10.0 ** pred["g"]
    ib = signs["b"] * 10.0 ** pred["b"]
    ich = channel_current(df, pred["c"], c_form)
    com = -(ig + ib) / 2.0
    i_hi, i_lo = ich + com, -ich + com
    return np.column_stack([np.where(up, i_hi, i_lo), ig, np.where(up, i_lo, i_hi), ib])


# ----------------------------------------------------------------------------- regressors
def make_regressor(mtype, hp, seed=0):
    if mtype == "poly_ridge":
        return make_pipeline(StandardScaler(), PolynomialFeatures(hp["degree"]), StandardScaler(),
                             Ridge(alpha=hp["alpha"]))
    if mtype == "random_forest":
        return RandomForestRegressor(n_estimators=hp["n_estimators"], max_features=hp["max_features"],
                                     min_samples_leaf=hp["min_samples_leaf"], n_jobs=1, random_state=seed)
    if mtype == "xgboost":
        return xgb.XGBRegressor(n_estimators=hp["n_estimators"], learning_rate=hp["learning_rate"],
                                max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
                                subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
                                tree_method="hist", n_jobs=1, random_state=seed)
    if mtype == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=tuple(hp["hidden"]), activation="tanh", alpha=hp["alpha"],
            learning_rate_init=hp["lr"], max_iter=hp["max_iter"], tol=1e-10,
            n_iter_no_change=hp["patience"], random_state=seed))
    raise ValueError(mtype)


class ScaledTarget:
    """Standardises y before fitting (helps the MLP and ridge; harmless for trees)."""

    def __init__(self, reg):
        self.reg = reg

    def fit(self, X, y):
        self.mu, self.sd = float(np.mean(y)), float(np.std(y))
        self.reg.fit(X, (y - self.mu) / self.sd)
        return self

    def predict(self, X):
        return self.reg.predict(X) * self.sd + self.mu


class TransistorModel:
    """One saved model per case: predicts Id, Ig, Is, Ib from the dataset input columns."""

    def __init__(self, case, mtype, hps, seed=0):
        self.case, self.mtype, self.hps, self.seed = case, mtype, hps, seed

    def fit(self, df):
        X = features(df)
        y, self.signs = targets(df)
        self.c_form = self.hps["c"]["form"]
        y["c"] = channel_target(y, self.c_form)
        self.regs = {p: ScaledTarget(make_regressor(self.mtype, self.hps[p], seed=self.seed)).fit(X, y[p])
                     for p in PARTS}
        return self

    def predict_parts(self, df):
        X = features(df, vds_floor=VDS_FLOOR)
        return {p: self.regs[p].predict(X) for p in PARTS}

    def predict(self, df):
        """Returns array (N, 4): Id, Ig, Is, Ib in A."""
        return rebuild(df, self.predict_parts(df), self.signs, self.c_form)


class AveragedModel:
    """Average of several TransistorModels (same settings, different random starts).
    The three parts are averaged in their log/transformed space, then Id, Ig, Is, Ib are rebuilt."""

    def __init__(self, members):
        self.members = members
        self.case, self.mtype = members[0].case, members[0].mtype
        self.signs, self.c_form = members[0].signs, members[0].c_form

    def predict_parts(self, df):
        parts = [m.predict_parts(df) for m in self.members]
        return {p: np.mean([q[p] for q in parts], axis=0) for p in PARTS}

    def predict(self, df):
        return rebuild(df, self.predict_parts(df), self.signs, self.c_form)


# ----------------------------------------------------------------------------- metrics
def rel_err(pred, true):
    return np.abs(pred - true) / np.abs(true)


def metrics(pred, true):
    """pred/true (N, 4) currents -> dict per current: R2 on log10|I|, median/mean/p95 rel. error."""
    out = {}
    for j, c in enumerate(CURRENTS):
        r = rel_err(pred[:, j], true[:, j])
        lt, lp = np.log10(np.abs(true[:, j])), np.log10(np.maximum(np.abs(pred[:, j]), 1e-300))
        r2 = 1 - np.sum((lp - lt) ** 2) / np.sum((lt - lt.mean()) ** 2)
        out[c] = dict(r2_log=r2, median=np.median(r), mean=np.mean(r), p95=np.percentile(r, 95))
    return out


def load_case(root, case):
    return pd.read_csv(root / "data" / f"{case}.csv")


def n_jobs(gb_per_job=2.0):
    """Parallel workers: all cores, but no more than free memory allows (about gb_per_job each).
    Override with the environment variable N_JOBS, e.g.  N_JOBS=4 python scripts/03_tune.py"""
    if os.environ.get("N_JOBS"):
        return max(1, int(os.environ["N_JOBS"]))
    cores = os.cpu_count() or 1
    try:
        with open("/proc/meminfo") as f:
            avail_kb = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable:"))
        by_mem = max(1, int(avail_kb / 1024 ** 2 / gb_per_job))
    except (OSError, StopIteration):
        by_mem = cores
    return min(cores, by_mem)
