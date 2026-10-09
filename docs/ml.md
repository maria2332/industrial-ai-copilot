# Machine Learning

> **Status:** Phase 1 in progress — blocks 1.1 (data and EDA) and 1.2 (features and validation) done, with results. Block 1.3 (sections 5–6): method implemented, results pending. Sections 7–10 are written in blocks 1.4–1.5.

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
- **The `cycle` feature.** The age of a unit is known at prediction time, so it is not leakage, but it can let the model learn the lifetime distribution of this fleet instead of its health state. It is evaluated with and without in block 1.2, and included because it passes the pre-registered rule (D-018).

**What the experiments showed.** With raw sensors (notebook 01), the random row split was not more optimistic than the split by unit (0.956 vs 0.961, single split). The most likely reason: each raw reading carries independent noise, and the degradation signature is shared by the whole fleet (one fault mode), so the model learns a fleet-wide pattern rather than memorising individual units. With rolling features (notebook 02, 5-fold cross-validation, same diagnostic random forest) the gap appears, as anticipated: **0.994 ± 0.002 with random rows vs 0.972 ± 0.008 with whole units**. Overlapping windows make each validation row nearly identical to its neighbours in training. The random split hides most of the real errors (1 − PR-AUC of 0.006 vs 0.028, between 4 and 5 times smaller) and is suspiciously stable across folds. Grouped cross-validation by unit is therefore confirmed as the protocol; it also reproduces deployment, where the model always faces engines it has never seen. Folds are balanced without stratification (positive rate 0.15 in every fold) because every unit contributes exactly H + 1 positive cycles.

Split strategy: see [D-003 in decisions.md](decisions.md#d-003--split-strategy-by-engine-unit-with-past-only-features) — grouped cross-validation by unit for model selection and the official test set for the final evaluation.

## 4. Feature engineering

**Principle.** Features are *stateless* and *past-only*: each row is computed from its own unit's current and previous cycles, never from other units or from statistics of the training set. That is why computing them before a grouped split cannot leak, and why the same function serves training and inference. Anything *stateful* (scaling, imputation) is fitted inside the model `Pipeline` on training folds only. The feature step itself is wrapped as a scikit-learn transformer (`UnitHistoryFeatures`), so the saved model goes from raw sensor history to prediction (see D-017).

**Accepted features** (for each of the 14 selected sensors, 42 in total, plus the unit's age: 43):

| Feature | Definition | Why (evidence from the EDA) |
|---|---|---|
| `mean10` | Rolling mean over the last 10 cycles | Noise dominates cycle-to-cycle variation; smoothing exposes the level |
| `delta_baseline` | `mean10` minus the unit's mean over its first 20 cycles | Units start at different levels; the deviation from each unit's own healthy reference isolates degradation from manufacturing differences |
| `trend10` | `mean10` now minus `mean10` ten cycles earlier, once both windows are complete (0 before cycle 20) | Degradation accelerates near failure, so the rate of change should grow |
| `cycle` | Age of the unit in operating cycles (once per row, not per sensor) | Passes the pre-registered rule of D-018: +0.0088 grouped PR-AUC, more than one standard deviation |

**Unit age (`cycle`).** Not leakage (the age is known at prediction time), but it may teach the model the lifetimes of this fleet instead of reading its health. Pre-registered rule: include it only if it improves grouped PR-AUC by more than one standard deviation across folds. With the final feature definition (after the warm-up fix): **0.9812 ± 0.0058 with vs 0.9724 ± 0.0076 without**, a gain of +0.0088 that exceeds both standard deviations, so `cycle` is included (D-018). Before the warm-up fix the same comparison gave a gain equal to one standard deviation and `cycle` had been excluded; the rule was applied again to the corrected features. The generalisation risk is accepted and recorded as a limitation. The feature exploration in notebook 02 (sections 1–5) uses the 42 sensor features; the model pipelines use the selected set with `cycle` (`SELECTED_FEATURES` in `industrial_ai.ml.models`).

**Window sizes.** A 10-cycle window is about 5% of the median lifetime and a third of the 30-cycle horizon: long enough to smooth noise, short enough to react within the horizon. Longer windows smooth more but delay detection, which is the central trade-off. The 20-cycle baseline fits within the shortest observed test history (31 cycles). All three values are configurable in `FeatureConfig`.

**Rejected candidates** (not added "just in case"):

| Candidate | Reason |
|---|---|
| Rolling standard deviation | The EDA shows no visible change in variability near failure; can be revisited with an ablation |
| Lag features | Redundant with the rolling mean and the trend |
| Exponentially weighted mean | Redundant with the rolling mean, which is easier to explain |
| Ratios between sensors | No physically justified combination we can defend; sensor 12 (phi) is already a ratio and tree models capture interactions |
| Operational settings | Pure noise in FD001 (single regime) |

**Results (notebook 02).**

- `delta_baseline` has the strongest correlation with RUL for all 14 sensors (|ρ| up to 0.865 for sensor 11, vs 0.718 raw). The two steps add separate gains: smoothing (raw → rolling mean) adds up to +0.14, mostly for noisy or quantised sensors (3, 17, 2, 20, 21); removing each unit's offset (rolling mean → `delta_baseline`) adds a further +0.03 to +0.14, most for the fan speeds 13 and 8 and for sensors 11, 12 and 7.
- Sensors 9 and 14 stay far below the rest (−0.37 and −0.21 after the baseline): removing an offset cannot fix trends that go in opposite directions across units.
- `trend` is the weakest feature on its own (|ρ| 0.20–0.53): the rate of change is noisy and close to zero for most of a unit's life, so any value it has is concentrated near failure. Its contribution is checked with model attributions in block 1.5.
- **Warm-up artefact found and fixed.** The plot of unit 1 showed a spike in `trend` around cycle 11: during the first cycles the rolling mean is computed over fewer than 10 values, so comparing against it is noisy. `trend` is now 0 until both windows are complete (cycle ≥ window + lag), with a dedicated test. After the fix its correlation with RUL rose slightly for every sensor (sensor 11: −0.511 → −0.532), and the decision on `cycle` changed (D-018). The level (`mean10`) keeps a partial window during warm-up; this only affects cycles far from failure (the shortest lifetime is 128 cycles).
- Raw sensors and engineered features compared under the same protocol (block 1.3): a random forest on the current-cycle raw sensors reaches 0.951 ± 0.008 grouped PR-AUC vs 0.972 ± 0.008 with the engineered sensor features (without `cycle`), almost halving the remaining error (0.049 → 0.028).

**Input contract.** Features need each unit's complete history (contiguous cycles from 1) because the baseline comes from its first cycles. `build_features` rejects incomplete histories instead of computing misleading values. In production this corresponds to keeping the history since commissioning or the last overhaul.

## 5. Baselines and model comparison

**Protocol.** Every candidate is a complete Pipeline (`industrial_ai.ml.models.build_pipeline`): raw sensor history → `UnitHistoryFeatures` with the selected feature set (42 sensor features + `cycle`) → (scaling) → classifier. All candidates use the **same five unit folds** as the leakage experiment, and `cross_validate_by_unit` (`industrial_ai.ml.selection`) refits a fresh copy of each pipeline per fold and stores an **out-of-fold score** for every training row: each row is scored by a model that never saw its unit. A spy-model test verifies this property. Model inputs go through `model_inputs`, which keeps only unit, cycle and sensors, so the labels can never enter the model.

**Candidates** (from simplest to most complex):

| Candidate | Why it is included |
|---|---|
| Logistic regression (with standard scaling) | Linear baseline: if it is close to the others, the problem is mostly linear in the features |
| Random forest | Robust, few sensitive hyperparameters, captures interactions and non-linearities |
| Gradient boosting (`HistGradientBoostingClassifier`) | Usually the strongest family on tabular data |
| XGBoost | Industry-standard gradient boosting; included to check whether it adds anything over scikit-learn's implementation |
| *Reference:* random forest on the current-cycle values of the same inputs (raw sensors and age) | Not a candidate: measures what the engineered features add under the same protocol |

Only logistic regression is scaled: tree models split on thresholds and are insensitive to monotonic rescaling, while a regularised linear model is not.

**Hyperparameters** are fixed, reasonable defaults and are not tuned (D-020). Gradient boosting runs with **early stopping disabled**: scikit-learn would hold out a random 10% of the training rows to decide when to stop, and random rows are exactly the leaky split this project avoids.

**Primary metric: PR-AUC** (average precision), threshold-free and focused on the positive class (section 6).

**Selection rule** (pre-registered, D-015): the simplest candidate whose mean PR-AUC is within one standard deviation of the best candidate's mean (that candidate's standard deviation across folds). A difference smaller than the fold-to-fold noise is not evidence for a more complex model.

**Results.** *To be completed after running notebook 02 (block 1.3).*

## 6. Metrics and decision-threshold selection

**Why not accuracy.** With 15% positive cycles, a model that never predicts failure is 85% accurate in cross-validation (97.5% on the test set) while being useless. Accuracy is reported only to make this point.

**Threshold-free metrics.**

- **PR-AUC** (primary): summarises precision over all recall levels. Its value for a random model equals the prevalence, so it must always be read next to it (0.15 in cross-validation, 0.025 on all test cycles).
- **ROC-AUC** (secondary): insensitive to class imbalance, which makes it look excellent even when precision is poor; reported for completeness.

**Metrics at the decision threshold** (`industrial_ai.evaluation.metrics.threshold_metrics`): precision, recall, F1, false positive rate, false negative rate and the confusion matrix.

**Which error matters more.** In this scenario a missed warning window (false negative) is assumed to cost more than an unnecessary inspection (false positive), so recall is prioritised. This is a business assumption, not a universal rule: too many false alarms cause *alarm fatigue*, and operators stop trusting the system. With real costs, the threshold would minimise expected cost instead.

**Threshold rule** (pre-registered, D-019): the threshold with the highest precision among those reaching **recall ≥ 0.90**, chosen on the **out-of-fold** scores of the selected model. It is never chosen on training predictions (optimistic) or on the test set (which would turn the test set into a validation set).

**Unit-level view: first alarm.** Row metrics count every cycle separately, but maintenance reacts to the first alarm of a unit. `first_alarms` returns, for each unit, the first cycle at which the score stays above the threshold for `k` consecutive cycles, and its RUL at that moment. On out-of-fold scores every training unit fails, so each first alarm is classified as missed, in window (RUL ≤ H), early (H < RUL ≤ 2H, still useful warning) or premature (RUL > 2H, probably a false alarm). Requiring 3 consecutive cycles filters isolated spikes at the cost of a short delay.

**Label boundary.** A cycle with RUL 31 is labelled negative and one with RUL 30 positive, although the engine is in practically the same state. Many "false positives" are therefore early warnings just outside the horizon, which the first-alarm view and the scatter of test units make visible.

**Final test evaluation.** The selected pipeline is refitted on all 100 training units and evaluated **once** on the test set with the threshold from cross-validation, in two views: every cycle (continuous monitoring) and the last observed cycle per unit (the usual benchmark protocol). Recall should be comparable with cross-validation; precision and PR-AUC are expected to be lower on all test cycles because of the prevalence shift (section 2).

**Results.** *To be completed after running notebook 02 (block 1.3).*

## 7. Anomaly detection and lead-time evaluation

*Block 1.4.*

## 8. Explainability

*Block 1.5.*

## 9. Model persistence and versioning

*Block 1.5.*

## 10. Classical ML vs deep learning for this problem

*Block 1.5.*
