# Machine Learning

> **Status:** Phase 1 in progress — block 1.1 (data foundation and exploratory analysis) done. Sections 4–10 are written in blocks 1.2–1.5.

## 1. Dataset

**Source.** NASA C-MAPSS turbofan engine degradation simulation, subset FD001.
Citation: A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. Downloaded from the PHM Society mirror of the NASA repository with `python scripts/download_data.py`, which also writes `data/raw/cmapss/manifest.json` with a SHA-256 checksum and line count per file, so any silent change in the data can be detected.

**Files.**

| File | Content |
|---|---|
| `train_FD001.txt` | Run-to-failure trajectories: every unit is observed until it fails |
| `test_FD001.txt` | Truncated trajectories: every unit stops some time before failure |
| `RUL_FD001.txt` | True remaining useful life after the last observed cycle of each test unit |

**Columns.** `unit`, `cycle`, three operational settings and 21 sensors. Sensor meanings follow Saxena et al. (2008), Table 2, in the column order commonly assumed in the literature (the dataset readme does not name the sensors). In the project narrative, the high-pressure compressor (HPC) and turbines correspond to the gas-generator core of the fictitious aeroderivative units.

**Validation at load time** (`industrial_ai.ml.data`): expected number of columns, no missing or infinite values, positive unit ids, and contiguous cycles starting at 1 in every unit. Any violation raises `DataValidationError` instead of silently producing a wrong dataset.

**Exploratory findings.** *To be completed after running `notebooks/01_data_exploration.ipynb`: size and lifetime range, constant columns, operating conditions, sensors with a degradation trend, label balance and the leakage experiment.*

## 2. Labels

**Remaining useful life (RUL)** is the number of operating cycles left after the current one.

- *Training set:* `RUL = last cycle of the unit − current cycle`, so the last recorded cycle has RUL = 0.
- *Test set:* the RUL file only gives the remaining life after the last observed cycle, so every earlier cycle is reconstructed as `RUL = RUL_last + (last observed cycle − current cycle)`. This allows evaluating on every test cycle, not only the last one.

**Classification target.** `failure_within_horizon = 1` if `RUL ≤ H`, with `H = 30` cycles by default (configurable). The horizon is a business choice: how much warning maintenance needs to plan an intervention. The EDA compares H = 15, 30 and 50 to show its effect on class balance.

## 3. Leakage risks and split strategy

Risks identified before modelling:

- **Random row splits.** Consecutive cycles of the same unit are very similar and share that unit's baseline, so a random split evaluates on near-copies of training rows and produces optimistic metrics. The EDA quantifies this with a simple experiment.
- **Overlapping rolling windows.** Rolling features make neighbouring rows share even more information, which amplifies the previous risk.
- **Preprocessing fitted on all data.** Scalers or imputers fitted before splitting leak statistics of the evaluation data; they are fitted inside a scikit-learn `Pipeline` on training data only.
- **Features computed with future information.** Rolling statistics are computed per unit using past cycles only.
- **Model selection on the test set.** The official test set is used once, for the final evaluation.
- **The `cycle` feature.** The age of a unit is known at prediction time, so it is not leakage, but it can let the model learn the lifetime distribution of this fleet instead of its health state. It is evaluated with and without in block 1.2.

Split strategy: see [D-003 in decisions.md](decisions.md#d-003--split-strategy-by-engine-unit-with-past-only-features) — grouped cross-validation by unit for model selection and the official test set for the final evaluation.

## 4. Feature engineering

*Block 1.2.*

## 5. Baselines and model comparison

*Block 1.3.*

## 6. Metrics and decision-threshold selection

*Block 1.3.*

## 7. Anomaly detection and lead-time evaluation

*Block 1.4.*

## 8. Explainability

*Block 1.5.*

## 9. Model persistence and versioning

*Block 1.5.*

## 10. Classical ML vs deep learning for this problem

*Block 1.5.*
