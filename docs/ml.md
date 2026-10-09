# Machine Learning

> **Status:** Phase 1 complete. Blocks 1.1–1.5 (data, features, classifier, anomaly detector, explanations, calibration and persistence) done, with results. Open issues and next steps in section 11.

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

**Results** (notebook 02, section 6; selected feature set with `cycle`; grouped 5-fold cross-validation):

| Candidate | PR-AUC | ROC-AUC | Training time, 5 folds (s) |
|---|---|---|---|
| *Reference:* random forest, current-cycle raw sensors and age | 0.952 ± 0.008 | 0.990 ± 0.002 | 6.5 |
| Logistic regression | **0.991 ± 0.001** | 0.998 ± 0.000 | 1.3 |
| Random forest | 0.982 ± 0.006 | 0.997 ± 0.001 | 9.2 |
| Gradient boosting | 0.992 ± 0.004 | 0.998 ± 0.001 | 5.2 |
| XGBoost | 0.992 ± 0.004 | 0.998 ± 0.001 | 5.5 |

- **Selected: logistic regression.** The best mean is 0.992 ± 0.004 (gradient boosting), so the band starts at 0.988; logistic regression (0.991) is inside it and is the simplest candidate. It is also the most stable across folds (± 0.001) and the fastest.
- **The gain comes from the features, not from model complexity.** With the same random forest, engineered features reach 0.982 vs 0.952 with the current-cycle raw values: the remaining error falls from 0.048 to 0.018. The reference is the lowest model in every fold. A linear model is enough because `delta_baseline` is almost monotonic with RUL.
- **XGBoost adds nothing** over scikit-learn's gradient boosting: both give identical scores and almost identical curves per fold.
- **Fold variability.** Logistic regression is nearly flat across folds (0.991–0.993), while the tree models vary more; the ranking between the top models changes from fold to fold, which is what the selection rule protects against.
- **Effect of `cycle`.** Logistic regression improved from 0.973 ± 0.005 without `cycle` (earlier run, same protocol) to 0.991 ± 0.001 with it, a larger gain than the random forest's (+0.009 in the ablation of block 1.2). Because a linear model cannot build interactions from the sensor features alone, the age may be compensating for that, or the model may lean on this fleet's lifetimes. How much the model relies on `cycle` is checked in block 1.5 (attributions and an age-only baseline).

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

**Results** (notebook 02, sections 8–10; logistic regression with the selected feature set).

*Threshold.* The rule selects **0.748**. At 0.5 the out-of-fold recall is already 0.943, so the rule raises the threshold to buy precision while keeping recall ≥ 0.90: precision 0.947 → 0.979 and false alarms 165 → 59 (false positive rate 0.9% → 0.3%), at the cost of 126 detected positive cycles. A different rule (for example, maximum recall with precision ≥ 0.95, or an expected-cost rule) would choose differently; the choice reflects the assumption of D-019.

| View | Prevalence | PR-AUC | Precision | Recall | False positive rate | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| Cross-validation (out-of-fold) | 0.150 | 0.991 | 0.979 | 0.902 | 0.003 | 2797 / 59 / 303 / 17472 |
| Test, every cycle | 0.025 | 0.955 | 0.943 | 0.798 | 0.001 | 265 / 16 / 67 / 12748 |
| Test, last cycle per unit | 0.250 | 0.997 | 1.000 | 0.920 | 0.000 | 23 / 0 / 2 / 75 |

*Accuracy trap.* A model that never predicts failure is 85.0% accurate on the training cycles; the selected model's 98.2% hides that it still misses 10% of the positive cycles.

*Alarms per unit (out-of-fold).* No unit fails without an alarm. With 1 cycle: 80 units alarm in the window (RUL ≤ 30), 20 early (31–60 cycles before failure) and none prematurely; median warning 28 cycles. Requiring 3 consecutive cycles moves 9 units from early to in window, with a median warning of 26 cycles. The worst case (unit 67) still gets 18 cycles of warning (16 with 3 consecutive cycles). The premature alarm seen without `cycle` disappears.

*Test set and prevalence.* Precision holds on all test cycles (0.943) despite the drop in prevalence (0.150 → 0.025) because the false positive rate also drops: most test cycles are far from failure, where the model is very confident. Recall drops (0.902 → 0.798), and the reason is the mix of cycles, not a weaker model:

| Recall by distance to failure | RUL 0–10 | RUL 11–20 | RUL 21–30 |
|---|---|---|---|
| Cross-validation (positive cycles) | 1.000 (1100) | 0.997 (1000) | 0.700 (1000) |
| Test, every cycle (positive cycles) | 1.000 (17) | 1.000 (106) | 0.679 (209) |

Within each band recall is practically the same. The hard band (RUL 21–30) holds 32% of the positives in cross-validation but 63% on the test set, because truncated test units contribute few positive cycles, mostly close to the horizon. Applying the cross-validation recall of each band to the test mix gives 0.810, so the mix explains almost all of the drop (0.902 → 0.810); the remaining 0.012 comes from the hard band.

*Where the errors are.* At the last observed cycle there are no false positives: test units at RUL 34, 37 and 38 score 0.41, 0.23 and 0.06, and every unit beyond RUL 40 scores close to 0. Of the two misses, one is a near miss (RUL ≈ 26, score 0.69) and the other is a **confident miss** (RUL ≈ 28, score 0.20), the only clear error of the model; it is investigated with local attributions in block 1.5.

## 7. Anomaly detection and lead-time evaluation

**Why a second model.** The classifier needs failure labels and only knows the fault mode it was trained on. A real plant rarely has many recorded failures, and a new fault mode would not be among them. An anomaly detector answers a different question, *is this unit behaving differently from healthy operation?*, without ever seeing a failure label. Here labels are used only to evaluate it.

**Design** (`industrial_ai.ml.anomaly.AnomalyDetector`, D-021):

- **Healthy reference:** cycles 20–60 of each training unit. Cycle 20 is the first with complete features (baseline and trend windows filled); cycle 60 is still at least 68 cycles before the shortest observed failure. Assumption: degradation is negligible that early.
- **Features:** deviation features only, `delta_baseline` and `trend` of the 14 sensors (28). Levels are excluded because the unit-to-unit offsets widen the healthy region; `cycle` is excluded because every old cycle would look anomalous, the reference containing only young units.
- **Methods**, sharing features, reference and threshold rule:
  - **max |z-score|** (baseline): the largest deviation of any feature from its healthy mean, in healthy standard deviations, the logic of a control chart;
  - **Isolation Forest** (scikit-learn), fitted on the healthy reference.
- **Label-free threshold:** the 99th percentile of the scores on the healthy reference, i.e. a **1% false-alarm budget** per cycle. An alarm needs **3 consecutive cycles** above the threshold.
- **Explanation:** for any cycle, the features with the largest |z-score| with respect to the healthy reference (`top_deviations`). It describes how the unit differs from healthy operation, not why: a deviation is a symptom, not a diagnosed cause.

**Evaluation** (notebook 02, sections 12–17), with the same five unit folds as the classifier: each detector is fitted on the healthy cycles of 80 units and scores the 20 held-out units. Because the threshold is computed on the training reference itself, the out-of-fold false-alarm rate on the held-out units' healthy cycles shows whether the 1% budget holds on unseen units.

| What | Metric |
|---|---|
| False alarms | Share of alarms on held-out healthy cycles and on cycles with RUL > 60; units alarmed (3 consecutive cycles) while healthy |
| Detection | Missed units; RUL at the first alarm (warning time): median and worst case; compared with the classifier under the same alarm rule |
| Failure window | Precision, recall and PR-AUC against the RUL ≤ 30 label (not what the detector is optimised for) |
| Test set | False-alarm rate on test cycles with RUL > 60; every-cycle and last-cycle metrics next to the classifier |

**Which metric matters most.** For an unsupervised detector the binding constraint is usually the **false-alarm rate during healthy operation**: an alarm with no visible cause erodes trust quickly. Given an acceptable false-alarm rate, the benefit is **warning time** with no missed units. As with the classifier, this ranking is an assumption about costs, not a universal rule.

**Selection rule** (pre-registered, D-021): fewer missed units wins; with the same number, Isolation Forest is chosen only if its median warning is at least 5 cycles longer and its healthy false-alarm rate is not higher; otherwise the simpler max |z-score|.

**Ablation:** the selected method is also run on the levels (`mean10`) instead of the deviation features, to check that removing unit offsets improves detection.

**A caveat found while testing.** On a synthetic check in which only one sensor drifted (2 of the 28 features), Isolation Forest barely reacted while max |z-score| flagged the drift clearly. Isolation Forest draws its splits on randomly chosen features and within the range seen in training, so a large deviation in a few features is diluted and its score saturates once a point leaves the training range. When several sensors drift together, as in FD001 degradation, both methods detect it; the comparison on the real data decides.

**Results** (notebook 02, sections 12–17; out-of-fold unless stated):

| | max \|z-score\| | Isolation Forest | Classifier (section 8) |
|---|---|---|---|
| False alarms on held-out healthy cycles (budget 1%) | 1.4% | 1.1% | — |
| Units alarmed while healthy (3 consecutive cycles, cycles 20–60) | 7 | 5 | — |
| Alarms on cycles with RUL > 60 | 35.2% | 34.8% | — |
| Missed units | 0 | 0 | 0 |
| Median warning (RUL at first alarm) | 109 | 109 | 26 |
| Least warning | 62 | 62 | 16 |
| PR-AUC against the RUL ≤ 30 label | 0.759 | 0.850 | 0.991 |

- **The budget holds on unseen units:** 1.4% and 1.1% of held-out healthy cycles raise an alarm, close to the 1% set on the training reference. With 3 consecutive cycles, 7 (z-score) and 5 (Isolation Forest) of 100 units still raise a spurious alarm early in life.
- **Selected: max |z-score**, by the pre-registered rule: no missed units for either method and the same median warning, so the simpler method wins. Isolation Forest is marginally better on healthy false alarms (1.1% vs 1.4%, 5 vs 7 units), a difference the rule deliberately does not reward.
- **The detector sees degradation about 100 cycles before failure.** Every unit's first alarm comes more than 60 cycles before failure (median 109, minimum 62), and the margin trajectories cross the threshold for good around RUL 100–130. That is why 35% of the cycles with RUL > 60 are flagged: most of them are genuine early degradation, not false alarms. "RUL > 60" is not "healthy", and only the healthy-window rate measures false alarms.
- **Two models, two questions.** The detector answers *has degradation started?* (about 109 cycles of warning, no labels needed); the classifier answers *will it fail within 30 cycles?* (precise timing, PR-AUC 0.991 against that label vs 0.759 for the detector). They are complementary: the detector builds a watch-list early, the classifier says when it becomes urgent.
- **Deviations beat levels** (ablation): with the levels (`mean10`), the median warning drops from 109 to 66 cycles and the worst case from 62 to 21, with a similar healthy false-alarm rate (1.8% vs 1.4%). Levels look better on alarms with RUL > 60 (11%) only because they detect later. Removing the unit offsets is what makes early detection possible.
- **Explanation of the most anomalous test unit** (unit 100 at cycle 198, true RUL 20): the largest deviations are the core speeds, sensor 14 (NRc, +61σ) and sensor 9 (Nc, +50σ), followed by sensor 4 (T50, +12σ). These are the sensors whose direction differs between units (notebook 01), which a linear classifier with fixed signs cannot fully use; an absolute deviation from each unit's own healthy state can. This is a second reason why the two models complement each other.
- **Test set:** the detector flags every positive cycle (recall 1.000) but also 28.7% of the negative ones, and 47 of the 75 units with RUL > 30 at their last cycle; most test units are already past the onset of degradation (median RUL at the last cycle: 86). For the question *failure within 30 cycles* the classifier is far better (last cycle: precision 1.000 vs 0.347); the detector is not meant to answer it.

## 8. Explainability and calibration

**Exact explanations for the selected model** (`industrial_ai.ml.explain`, D-022). The classifier is a logistic regression on standardised features, so its log-odds are exactly

    log-odds(x) = intercept + Σ_j coef_j · z_j,   with z_j = (x_j − training mean_j) / training std_j

and `coef_j · z_j` is the contribution of feature j relative to an average training cycle. The contributions add up exactly to the model's output (verified by a test). For a linear model with independent features this is what SHAP computes, so the `shap` dependency is not needed.

- **Global:** the coefficients, i.e. the change in log-odds per standard deviation of each feature.
- **Local:** the contributions of each feature for one cycle (`top_contributions`), which the API returns with each prediction.

Two cautions. Correlated features (the level and `delta_baseline` of the same sensor, and the redundant pairs 8/13 and 9/14) share or offset each other's coefficients, and each coefficient is a *conditional* effect: the change in risk when that feature moves and all others stay fixed. Coefficients describe the model, not the physics of the engine.

**Importance by sensor** (`grouped_permutation_importance`): for each of the five unit folds, the model is fitted on the training units; on the validation units, all features of one sensor (or `cycle`) are shuffled together and the drop in PR-AUC is recorded. Grouping by sensor avoids splitting the credit between correlated features; measuring on held-out units avoids rewarding features the model merely memorised.

**Age baseline.** The same protocol is applied to a model that only knows the age of the unit, to one that only knows the sensor features and to the selected one (both), to measure how much of the performance comes from `cycle` (D-018).

**Local case study.** The confident miss of section 6 (a test unit near failure scored low) is explained with its contributions and compared with what the anomaly detector saw at the same cycle.

**Calibration** (D-024). The API reports a *probability* of failure within the horizon, so it should mean what it says. The reliability table groups the scores into ten bins and compares the mean predicted probability with the observed rate; the **expected calibration error** (ECE) is their average gap weighted by bin size, and the **Brier score** is the mean squared error of the probabilities. The test set has a lower prevalence (2.5% vs 15%), but that comes from *which* cycles are observed (truncated units far from failure), not from a different relationship between sensors and failure (covariate shift, not label shift); if so, out-of-fold calibration should carry over to the test set. Rule fixed before looking: recalibrate only if the out-of-fold ECE exceeds 0.05.

**Results** (notebook 02, sections 19–23):

- **Coefficients.** The largest coefficient is `cycle`, and it is **negative** (−6.4 log-odds per standard deviation): *at the same sensor deviations*, an older unit is less likely to fail within 30 cycles. The deviations grow with age, so reaching a given deviation early in life signals fast degradation, while reaching it late signals a slow degrader. The model uses age to turn "how far from healthy" into "how fast", not to say that old units fail. Among sensor features the largest are the fan speed deviation (`sensor_8_delta_baseline`, +3.1) and several levels; correlated pairs share or offset each other's weight (for example `sensor_17_mean10` −1.3 and `sensor_17_delta_baseline` +0.9), so single coefficients should not be read in isolation.
- **Importance by sensor** (drop in out-of-fold PR-AUC when shuffled): sensor 8 (Nf) 0.108 ± 0.044, `cycle` 0.077 ± 0.013, sensor 9 (Nc) 0.074 ± 0.021, sensor 12 (phi) 0.042, sensor 11 (Ps30) 0.023, sensor 14 (NRc) 0.018; the rest below 0.015. Importance is not correlation: sensors 4, 11 and 7 had the strongest correlations with RUL (section 1) but low importance, because most sensors carry the same degradation signal and the model can do without any one of them. Grouping by sensor handles correlation *within* a sensor but not *across* sensors: the fitted model relies on fan speed 8 rather than its near-twin 13, so a low importance for 13 does not mean that 13 is uninformative. Refitting without each group (drop-column importance) would measure uniqueness; listed as future work.
- **Age baseline** (grouped cross-validation): age only PR-AUC 0.526 ± 0.029 (random model 0.15), sensors only 0.973 ± 0.005, both 0.991 ± 0.001. Age alone is a weak predictor; the model is driven by the sensors, and age adds +0.018 by interacting with them. The reliance on `cycle` is real (second in importance) and specific to this fleet's degradation rates, which confirms the limitation recorded in D-018.
- **The confident miss** (test unit 18, cycle 133, true RUL 28, risk 0.199 for a threshold of 0.748). Its fan speed had moved far from its own baseline (`sensor_8_delta_baseline` contributes +7.1), but its absolute levels were not yet in the range typical of failure, and its **core speed had fallen** instead of rising: `sensor_9_delta_baseline` contributes −1.7, because the model learned a positive sign for it. This is the behaviour seen in notebook 01, where sensors 9 and 14 moved in opposite directions in different units; a linear model with one sign per feature cannot follow both. **The anomaly detector flagged the same cycle strongly** (margin +10.0, sustained alarm): fan speed +13.5σ, corrected core speed (sensor 14) −10.0σ and sensor 20 −10.0σ from its healthy reference, because it uses absolute deviations. For comparison, test unit 40 at the same cycle and RUL (risk 0.931) was caught through its levels (sensors 11, 12, 15 and 17). The two models together would not have missed unit 18; the agent of Phase 4 will report both.
- **Calibration.** Out-of-fold: ECE 0.0031, Brier score 0.0117. Test, every cycle: ECE 0.0020, Brier score 0.0047. Both are far below the 0.05 rule, so **no recalibration** (D-024). The reliability diagram follows the diagonal, with a slight overestimation between 0.5 and 0.7 where only about 90 cycles per bin fall. Calibration carries over to the test set despite the prevalence shift, which supports the covariate-shift explanation; the Brier score is lower on test only because most test cycles are far from failure and easy.
- **Saved artifacts.** `scripts/train.py` reproduces the notebook exactly (PR-AUC 0.991 ± 0.001, threshold 0.748, test last-cycle recall 0.920; detector healthy false alarms 1.4%, median warning 109 cycles), and the loaded `Predictor` returns the same probability as the notebook for unit 18 (0.199).

## 9. Model persistence and versioning

**Separate training from inference.** `scripts/train.py` repeats the decisions of notebook 02 without the exploration (same features, folds and selection rules: D-015, D-019, D-021), evaluates once on the test set and saves two artifacts. The API only loads them (`industrial_ai.ml.inference.Predictor`); it never trains.

**Layout** (`industrial_ai.ml.registry`, D-023):

    models/failure_classifier/1.0.0/{model.joblib, metadata.json}
    models/anomaly_detector/1.0.0/{model.joblib, metadata.json}

`metadata.json` records what is needed to trust and reproduce an artifact: name, version, creation time (UTC), Python and library versions, SHA-256 of the training and test files, the feature list, the decision policy (classifier threshold and horizon; detector threshold, healthy window and consecutive-cycle rule) and the cross-validation and test metrics.

- **Immutable versions** (MAJOR.MINOR.PATCH, sorted numerically so 1.10.0 follows 1.9.0): saving over an existing version fails unless `--overwrite` is given. `MODEL_VERSION` sets the version; the latest version is loaded by default.
- **joblib, with care:** joblib stores scikit-learn objects efficiently, but it is pickle underneath. Loading a file executes code, so only artifacts created by this project are loaded; and pickles are tied to library versions, so a mismatch between the saved and the current versions raises a warning. Portable formats (ONNX, skops) are future work.
- **Artifacts are not committed** (`models/` is git-ignored): they are reproducible from the script and the data, whose checksums are stored in the metadata.

**Inference contract.** Both models need the complete history of one unit from cycle 1 (the features compare each cycle with the unit's own past) and describe its last cycle:

| `predict_failure` | `detect_anomaly` |
|---|---|
| probability, threshold, `high_risk`, horizon, top 3 feature contributions, model version | margin, `anomalous_now`, `sustained_alarm` (3 consecutive cycles), top 3 deviations, model version |

## 10. Classical ML vs deep learning for this problem

Deep learning (LSTMs or 1D CNNs on raw sequences) is popular on C-MAPSS, mostly for regressing the remaining useful life. It was not used here, deliberately:

- **Data size.** 100 training units and about 20,000 strongly autocorrelated rows: the effective sample is closer to 100 trajectories than to 20,000 examples. Sequence models need more data, tuning and compute to be reliable, and are harder to validate without leakage.
- **Little headroom.** A logistic regression on engineered features reaches PR-AUC 0.991 ± 0.001 with grouped cross-validation, and boosting does not beat it (section 5). The gain came from the features, not from model capacity.
- **Explainability and operation.** The linear model is explained exactly, trains in about a second and is a few kilobytes on disk.

Deep learning would make sense with many more units, high-frequency raw signals (vibration, acoustic spectra) where hand-crafted features are hard to design, several operating regimes and fault modes (FD002–FD004), or multimodal inputs such as inspection images. A sequence autoencoder for anomaly detection is listed as future work, to be adopted only if it beats the current detector under the same protocol.

## 11. Open issues and next steps

- **The test set is no longer untouched.** Unit 18 was examined in detail, so any change motivated by it (for example, absolute deviations for sensors 9 and 14, or combining both models into one score) must be validated with grouped cross-validation and reported as a new model version, not as a test-set result.
- **Age as a rate.** The negative coefficient of `cycle` suggests an explicit degradation-rate feature (deviation per cycle of age) could replace the raw age and generalise better to fleets with other lifetimes.
- **Direction-free sensor effects.** Absolute deviations of the core speeds, or the detector's margin as an input, would let the classifier use sensors whose direction differs between units.
- **Drop-column importance**, to separate what a sensor uniquely contributes from what the fitted model happens to rely on.
- **Tuning with nested cross-validation**, the other C-MAPSS subsets (FD002–FD004, several operating regimes and fault modes) and a sequence autoencoder for anomaly detection, under the same protocol.
- **Combining both models** in the API and the agent: the detector as an early watch-list, the classifier for urgency (Phases 4–5).
