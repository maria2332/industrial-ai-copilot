# Machine Learning

> **Status:** Phase 1 in progress — blocks 1.1 (data and EDA) and 1.2 (features and validation) done. Sections 5–10 are written in blocks 1.3–1.5.

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

### Exploratory findings (notebook 01)

- **Size.** Train: 20,631 rows, 100 units. Test: 13,096 rows, 100 units. Both match the published sizes of FD001 (checked by `tests/integration`).
- **Lifetimes.** Between 128 and 362 cycles (median 199, mean 206, standard deviation 46), right-skewed with a few long-lived units.
- **Quality.** No missing values. `setting_3` and sensors 1, 5, 10, 16, 18 and 19 have a single distinct value (the non-zero standard deviations of sensors 5 and 16 are floating-point noise). Sensor 6 only takes two values (21.60 / 21.61): quantisation noise with a weak correlation with RUL (−0.13).
- **Operating conditions.** `setting_3` is constant and `setting_1` / `setting_2` only show small symmetric noise around 0, which confirms a single operating regime. The settings are therefore not used as features in FD001; they would be essential in FD002 and FD004.
- **Selected sensors (14).** 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21. The EDA arrived independently at the same subset commonly used for FD001 in the literature.
- **Degradation patterns.** As failure approaches, sensors 2, 3, 4, 8, 11, 13, 15 and 17 increase and sensors 7, 12, 20 and 21 decrease. Sensors 9 (Nc) and 14 (NRc) change clearly within each unit, but in opposite directions depending on the unit, which explains their weak pooled correlation (−0.32 and −0.20): a low pooled correlation does not mean a useless sensor.
- **Noise and offsets.** Cycle-to-cycle noise is comparable to the degradation range until roughly the last 50–100 cycles, and units start at visibly different levels.
- **Quantised and redundant sensors.** Sensors 8 and 13 (about 50 distinct values) and 17 (13 integer values) are coarse but still show a trend. Physical and corrected speeds (8 / 13 and 9 / 14) are almost redundant under a single operating condition, so feature attributions will be split between each pair.
- **Strongest monotonic association with RUL (Spearman).** Sensor 11 (−0.72), 4 (−0.70), 12 (+0.69), 7 (+0.68), 15 (−0.67).
- **Label balance.** Positive rate in train: 7.8% (H = 15), 15.0% (H = 30), 24.7% (H = 50). In test, over all cycles: 2.5% at H = 30, because test trajectories are truncated and most units end far from failure (median RUL after the last cycle: 86).
- **Leakage experiment with raw sensors.** PR-AUC 0.956 with a random row split versus 0.961 with a split by unit (single split, random forest): no measurable gap. See section 3.

## 2. Labels

**Remaining useful life (RUL)** is the number of operating cycles left after the current one.

- *Training set:* `RUL = last cycle of the unit − current cycle`, so the last recorded cycle has RUL = 0.
- *Test set:* the RUL file only gives the remaining life after the last observed cycle, so every earlier cycle is reconstructed as `RUL = RUL_last + (last observed cycle − current cycle)`. This allows evaluating on every test cycle, not only the last one.

**Classification target.** `failure_within_horizon = 1` if `RUL ≤ H`, with `H = 30` cycles by default (configurable). The horizon is a business choice: how much warning maintenance needs to plan an intervention. The EDA compares H = 15, 30 and 50 to show its effect on class balance.

**Prevalence shift between train and test.** 15% of training cycles are positive but only 2.5% of test cycles. This is realistic (a monitored fleet is mostly healthy) and has consequences for evaluation: recall does not depend on prevalence, but precision and PR-AUC do (the PR-AUC of a random model equals the prevalence). Cross-validation and test precision are therefore not directly comparable, and the test set is evaluated in two ways in block 1.3: every cycle (continuous monitoring) and the last observed cycle of each unit (the usual benchmark protocol).

## 3. Leakage risks and split strategy

Risks identified before modelling:

- **Random row splits.** Consecutive cycles of the same unit are very similar and share that unit's baseline, so a random split evaluates on near-copies of training rows and produces optimistic metrics. The EDA quantifies this with a simple experiment.
- **Overlapping rolling windows.** Rolling features make neighbouring rows share even more information, which amplifies the previous risk.
- **Preprocessing fitted on all data.** Scalers or imputers fitted before splitting leak statistics of the evaluation data; they are fitted inside a scikit-learn `Pipeline` on training data only.
- **Features computed with future information.** Rolling statistics are computed per unit using past cycles only.
- **Model selection on the test set.** The official test set is used once, for the final evaluation.
- **The `cycle` feature.** The age of a unit is known at prediction time, so it is not leakage, but it can let the model learn the lifetime distribution of this fleet instead of its health state. It is evaluated with and without in block 1.2.

**What the experiments showed.** With raw sensors (notebook 01), the random row split was not more optimistic than the split by unit (0.956 vs 0.961, single split). The most likely reason: each raw reading carries independent noise, and the degradation signature is shared by the whole fleet (one fault mode), so the model learns a fleet-wide pattern rather than memorising individual units. The experiment is repeated with rolling features in notebook 02, where overlapping windows make neighbouring rows much more similar. Either way, the split by unit remains the protocol because it reproduces deployment: the model is always evaluated on engines it has never seen.

Split strategy: see [D-003 in decisions.md](decisions.md#d-003--split-strategy-by-engine-unit-with-past-only-features) — grouped cross-validation by unit for model selection and the official test set for the final evaluation.

## 4. Feature engineering

**Principle.** Features are *stateless* and *past-only*: each row is computed from its own unit's current and previous cycles, never from other units or from statistics of the training set. That is why computing them before a grouped split cannot leak, and why the same function serves training and inference. Anything *stateful* (scaling, imputation) is fitted inside the model `Pipeline` on training folds only. The feature step itself is wrapped as a scikit-learn transformer (`UnitHistoryFeatures`), so the saved model goes from raw sensor history to prediction (see D-017).

**Accepted features** (for each of the 14 selected sensors, 42 in total):

| Feature | Definition | Why (evidence from the EDA) |
|---|---|---|
| `mean10` | Rolling mean over the last 10 cycles | Noise dominates cycle-to-cycle variation; smoothing exposes the level |
| `delta_baseline` | `mean10` minus the unit's mean over its first 20 cycles | Units start at different levels; the deviation from each unit's own healthy reference isolates degradation from manufacturing differences |
| `trend10` | `mean10` now minus `mean10` ten cycles earlier (0 until available) | Degradation accelerates near failure, so the rate of change should grow |

**Window sizes.** A 10-cycle window is about 5% of the median lifetime and a third of the 30-cycle horizon: long enough to smooth noise, short enough to react within the horizon. Longer windows smooth more but delay detection, which is the central trade-off. The 20-cycle baseline fits within the shortest observed test history (31 cycles). All three values are configurable in `FeatureConfig`.

**Rejected candidates** (not added "just in case"):

| Candidate | Reason |
|---|---|
| Rolling standard deviation | The EDA shows no visible change in variability near failure; can be revisited with an ablation |
| Lag features | Redundant with the rolling mean and the trend |
| Exponentially weighted mean | Redundant with the rolling mean, which is easier to explain |
| Ratios between sensors | No physically justified combination we can defend; sensor 12 (phi) is already a ratio and tree models capture interactions |
| Operational settings | Pure noise in FD001 (single regime) |
| `cycle` (age) | Not leakage, but it may teach "old units fail" instead of health; decided with a pre-registered rule in notebook 02 (kept only if it improves grouped PR-AUC by more than one standard deviation) |

**Input contract.** Features need each unit's complete history (contiguous cycles from 1) because the baseline comes from its first cycles. `build_features` rejects incomplete histories instead of computing misleading values. In production this corresponds to keeping the history since commissioning or the last overhaul.

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
