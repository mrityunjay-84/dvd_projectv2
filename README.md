# ML-Driven PVT-Variation-Aware Leakage Estimation (22nm HP, VDD 0.9 V, 8-bit parity checker)

Digital VLSI Design course project. Machine-learning models of single NMOS/PMOS transistors
(ON and OFF) predict the four terminal currents (Id, Ig, Is, Ib) under process, voltage and
temperature variation. The models are then used to predict the leakage of INV, NAND2, NOR2, AND2
and OR2 for every input combination and, hierarchically, of an 8-bit parity checker. Every
prediction is checked against ngspice.

| Item | Value |
|---|---|
| Model file | `model/22nm_HP.pm` (PTM-style BSIM4, `level=54`) |
| Nominal VDD | 0.9 V (sampled 0.81-0.99 V) |
| Channel length | L = 22 nm; widths W = 22-176 nm (1-8 x Lmin) |
| Test circuit | 8-bit parity checker (7 XOR, each 4 NAND2: 28 NAND2, 112 transistors) |
| Simulator | ngspice 45.2 |

## Results at a glance

| Level | Median error | 95th-percentile error | Bias (median signed error) |
|---|---|---|---|
| Transistor models, test set (16 currents) | 0.30-1.29 % | 1.2-4.9 % | - |
| Gates, 18 gate/input cases x 200 corners | 0.74-1.32 % | 1.8-4.4 % | -0.6 to -1.3 % |
| 8-bit parity checker, 256 vectors x 200 corners | 1.29 % | 4.36 % | -1.27 % |

Relative error = |ML - SPICE| / SPICE. The chosen model is an MLP for all four cases.

## Requirements

* Linux or WSL Ubuntu
* ngspice 45.2 on `PATH` (or set `NGSPICE=/path/to/ngspice`)
* Python 3.14 with the packages below (the versions used for the reported results):

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy==2.5.4 pandas==3.1.1 scipy==1.18.1 scikit-learn==1.9.1 \
            xgboost==3.4.1 matplotlib==3.11.2 joblib==1.6.0
```

Check the installation:

```bash
ngspice -v | head -2
python -c "import sklearn, xgboost; print(sklearn.__version__, xgboost.__version__)"
```

## How to run

Run every script from the project root, in this order. Each script reads the outputs of the
previous ones.

| # | Command | What it does |
|---|---|---|
| 1 | `python scripts/01_generate_data.py` | 4 x 10,000 single-transistor SPICE samples, example netlists |
| 2 | `python scripts/02_data_check.py` | Kirchhoff check, three trend plots, range checks |
| 3 | `python scripts/03_tune.py` | hyperparameter search, 4 model types x 4 cases (train -> validation); the longest step |
| 4 | `python scripts/04_train.py` | trains the 16 models, validation comparison table |
| 5 | `python scripts/05_select_and_test.py` | picks the winner per case, saves the 4 models, test-set results |
| 6 | `python scripts/06_gates.py` | 18 gate/input cases x 200 corners in SPICE and ML |
| 7 | `python scripts/07_test_circuit.py` | parity checker: 256 vectors x 200 corners in SPICE and ML |

Parallel jobs are capped by free memory; set `N_JOBS=<n>` to override
(e.g. `N_JOBS=4 python scripts/03_tune.py`).

All random choices use fixed seeds (data 20261114, split 20261114, corners 20261115, network
starts 0/1/2), and every fit runs single-threaded, so a rerun reproduces the results digit for
digit.

## Repository layout

```
model/22nm_HP.pm                 assigned model file (unchanged)
scripts/
  spice_lib.py                   model-card variation, single-FET netlists, batched ngspice runs
  ml_lib.py                      data split, features, physics-structured model, metrics
  gate_lib.py                    gate/circuit topologies, corners, switch-level logic, ML grouping
  01_generate_data.py ... 07_test_circuit.py
data/<case>.csv                  datasets (deliverable 1)
netlists/transistor|gates|circuit/   example netlists (deliverable 2)
results/step1/                   Kirchhoff check, trend plots (deliverable 3)
results/step2/                   model comparison, chosen models, test results, plots (4, 5)
models/<case>.joblib             the four chosen models (5)
results/step3/                   gate results table, gate parity plots, raw simulations (6, 7)
results/step4/                   parity-checker results table and parity plot (8)
```

`models/all/` (the 16 comparison models) is created by `04_train.py` and is not committed.

## Deliverables map

| # | Deliverable | File(s) |
|---|---|---|
| 1 | Datasets | `data/nmos_on.csv`, `nmos_off.csv`, `pmos_on.csv`, `pmos_off.csv` |
| 2 | Example netlists | `netlists/transistor/*_example.cir`, `netlists/gates/*.cir`, `netlists/circuit/parity8.cir` |
| 3 | Data check | `results/step1/kirchhoff_check.csv`, `plot_off_current_vs_temperature.png`, `plot_off_current_vs_vth0.png`, `plot_gate_current_vs_toxe.png` |
| 4 | Model comparison | `results/step2/model_comparison_validation.csv` (summary: `model_comparison_summary.csv`) |
| 5 | Chosen models | `results/step2/test_results_chosen.csv`, `plot_pred_vs_spice_<case>.png`, `models/<case>.joblib` |
| 6 | Gate results table | `results/step3/gate_results_table.csv` |
| 7 | Gate parity plot | `results/step3/plot_gate_parity.png` (per input: `plot_gate_parity_by_input.png`) |
| 8 | Test-circuit results | `results/step4/circuit_results_table.csv`, `plot_circuit_parity.png` |
| 9 | Code | `scripts/` |

## Method notes

* **Netlists.** Every netlist has `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`;
  single-transistor netlists drive every terminal with its own source. Every transistor sets
  `rgatemod=0 rbodymod=0` (RF-only BSIM4 switches; removes a ~5.6e-17 A measurement floor without
  changing DC) and junction geometry AD = AS = W x 44 nm, PD = PS = 2(W + 44 nm).
* **Sampling.** Section 4 of the brief: VDD first, then Vd, Vs, W, T and vth0, toxe, ndep, u0, xj
  (+/-10 %), all uniform. For PMOS OFF only, 60 % of the samples place Vd/Vs at rail-type values
  (TA-approved), so the model sees the full-supply bias an OFF PMOS holds inside gates.
* **Models.** Each transistor model predicts log|Ig|, log|Ib| and the channel part
  (I_hi - I_lo)/2 in a drain/source-symmetric frame, then rebuilds Id and Is so that
  Id + Ig + Is + Ib = 0 exactly. Four model types were compared (polynomial ridge, random forest,
  XGBoost, MLP); the MLP wins every case. The OFF models average three networks.
* **Gates.** Terminal voltages come from the recorded SPICE node voltages. Gate leakage = the
  predicted currents into the terminals connected to VDD (through ON transistors) plus inputs
  held high, which matches the SPICE definition (current from VDD and from high inputs).
* **Parity checker.** For each input vector, switch-level logic gives each NAND2's inputs; the
  step-3 NAND2 ML leakage for that input pair and corner is summed over the 28 NAND2s.
