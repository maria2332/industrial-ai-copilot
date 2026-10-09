# Architecture Decision Records

Each decision follows: **Problem → Options → Decision → Reason → Trade-offs**.
Status is *Accepted*, *Proposed* (to be validated in a later phase) or *Pending*.

---

## D-001 · Sensor dataset: NASA C-MAPSS FD001

**Status:** Accepted (Phase 0)

- **Problem:** we need public sensor data with temporal structure that supports both supervised failure prediction and anomaly detection.
- **Options:** NASA C-MAPSS; MetroPT-3 (real compressor data); Microsoft Azure Predictive Maintenance (synthetic); AI4I 2020 (synthetic, tabular); SKAB (pump test bench).
- **Decision:** C-MAPSS, subset FD001.
- **Reason:** 100 run-to-failure training trajectories give enough failure events to evaluate a classifier meaningfully; no missing values; one operating condition and one fault mode make it the right starting point; it is a widely cited benchmark. MetroPT-3 has very few documented failures, AI4I 2020 has no real temporal structure and SKAB only supports anomaly detection.
- **Trade-offs:** simulated rather than real plant data; time is measured in operating cycles; widely used, so the project must stand out through engineering and evaluation rather than dataset novelty.

## D-002 · Equipment narrative: fictitious aeroderivative gas turbines

**Status:** Accepted (Phase 0)

- **Problem:** the synthetic documentation must be consistent with the sensor data; an initial idea based on pumps did not match a turbofan simulation.
- **Options:** (A) aeroderivative gas turbines GT-101/GT-102; (B) pumps P-101/P-102 with a documented mismatch.
- **Decision:** A.
- **Reason:** aeroderivative gas turbines are derived from aircraft engines and are used in power generation and oil & gas, so documents, sensors and agent answers tell one coherent story.
- **Trade-offs:** C-MAPSS models a turbofan, including a fan and bypass flow that an industrial unit does not have; sensor semantics are an approximation. Asset tags are mapped to C-MAPSS engine units in the equipment registry, and the mapping is documented.

## D-003 · Split strategy: by engine unit, with past-only features

**Status:** Accepted (Phase 0)

- **Problem:** consecutive cycles of the same engine are almost identical, so a random row split leaks information and produces optimistic metrics.
- **Options:** random row split; global time cutoff; hold-out by unit; grouped cross-validation by unit.
- **Decision:** grouped cross-validation by unit for model selection and the official test set for the final evaluation. Rolling features are computed per unit using past cycles only, and all preprocessing is fitted inside a scikit-learn `Pipeline` on training data only.
- **Reason:** each C-MAPSS unit is an independent trajectory starting at cycle 1, so there is no shared calendar to cut on. Splitting by unit simulates the real question: *how does the model behave on an engine it has never seen?*
- **Trade-offs:** fewer units per fold and higher variance across folds, so metrics are reported as mean ± standard deviation.
- **Evidence (Phase 1):** with raw sensors, a random row split was not more optimistic than a split by unit (PR-AUC 0.956 vs 0.961, single split). With rolling features the leakage appears: 0.994 ± 0.002 with random rows vs 0.972 ± 0.008 with whole units (5-fold), so the random split hides most of the real errors.

## D-004 · RAG instead of fine-tuning

**Status:** Accepted (Phase 0)

- **Problem:** answers must be based on a small, changing set of technical documents, with exact values and traceable sources.
- **Options:** fine-tuning an LLM on the documents; RAG; putting all documents in the prompt (long context).
- **Decision:** RAG.
- **Reason:** updating knowledge means re-indexing a document instead of retraining; every answer can cite document, page and section; exact technical values are copied from retrieved text instead of being recalled from weights; low cost. Fine-tuning is better suited to teaching style, format or behaviour than to reliably injecting exact facts. Long context works for six documents but does not scale, is expensive per query and makes precise citation harder.
- **Trade-offs:** quality depends on chunking and retrieval; adds retrieval latency and an extra component (vector store).

## D-005 · LLM provider: local-first (Ollama) behind an abstraction

**Status:** Accepted (Phase 0), model choice Proposed (Phase 4)

- **Problem:** the system must not depend on a single vendor, and API keys must never be hard-coded.
- **Options:** OpenAI only; local models only; provider interface with several implementations.
- **Decision:** an `LLMProvider` interface with Ollama (default) and OpenAI implementations, selected through `.env`.
- **Reason:** local models are free, need no key and keep data on the machine, which matters for industrial data; OpenAI remains available when quality or tool-calling reliability requires it.
- **Trade-offs:** small local models are weaker at tool calling and need adequate RAM; two implementations to test.

## D-006 · Vector store: ChromaDB for the MVP

**Status:** Accepted (Phase 0)

- **Problem:** store chunk embeddings with metadata and filter by equipment and document type.
- **Options:** FAISS; ChromaDB; Qdrant; PostgreSQL + pgvector.
- **Decision:** ChromaDB with local persistence.
- **Reason:** native metadata filtering (essential so a question about GT-101 does not retrieve GT-102 chunks), persistence and zero infrastructure. FAISS is an index only, so metadata would have to be managed separately.
- **Trade-offs:** not the choice for large-scale production; pgvector would unify relational and vector data and is listed as future work.

## D-007 · Embedding model: multilingual sentence-transformers

**Status:** Proposed — validated with a retrieval benchmark in Phase 3

- **Problem:** questions may be in Spanish while documents are in English.
- **Options:** English-only sentence-transformers; multilingual models (e.g. `intfloat/multilingual-e5-small`, `BAAI/bge-m3`); OpenAI embeddings.
- **Decision (initial):** `intfloat/multilingual-e5-small`, behind an `EmbeddingProvider` interface.
- **Reason:** cross-lingual retrieval, small size, runs locally.
- **Trade-offs:** dense embeddings handle equipment tags and numbers poorly, which is mitigated with metadata filters (and hybrid BM25 search as an extension).

## D-008 · Router before the agent

**Status:** Accepted (Phase 0)

- **Problem:** should every question go through the agent?
- **Options:** agent for everything; deterministic router plus agent only for multi-step questions.
- **Decision:** router first. Documentation-only questions go directly to RAG; prediction endpoints never call an LLM.
- **Reason:** lower latency and cost, deterministic behaviour and easier testing. An agent adds value only when a question requires combining several tools.
- **Trade-offs:** routing rules can misclassify ambiguous questions; this is measured in the evaluation.

## D-009 · Custom agent loop, no framework in the MVP

**Status:** Accepted (Phase 0)

- **Problem:** choose how to implement tool calling.
- **Options:** LangChain / LangGraph; LlamaIndex; a small custom loop over the provider's native tool calling.
- **Decision:** custom loop.
- **Reason:** a few dozen lines that are fully understood, testable and explainable; no hidden behaviour.
- **Trade-offs:** no built-in features such as persistence or graph orchestration; LangGraph would be reconsidered if workflows became more complex.

## D-010 · FastAPI + Pydantic for the API

**Status:** Accepted (Phase 0)

- **Problem:** expose ML, RAG and the agent through a validated, documented API.
- **Options:** Flask; Django REST Framework; FastAPI.
- **Decision:** FastAPI with Pydantic models.
- **Reason:** request/response validation from type hints, automatic OpenAPI/Swagger documentation, async support for LLM calls.
- **Trade-offs:** fewer batteries included than Django (no admin, ORM or auth out of the box), which this PoC does not need.

## D-011 · Separation of training, inference and services

**Status:** Accepted (Phase 0)

- **Problem:** avoid a monolith where the API retrains models or recomputes embeddings.
- **Decision:** separate packages for `ml`, `rag`, `agents` and `api`; training and ingestion run as scripts; the API only loads artifacts.
- **Reason:** fast startup, reproducible artifacts, independent testing of each layer, and a clear path to production.
- **Trade-offs:** more files and explicit interfaces to maintain.

## D-012 · Packaging: src layout, pyproject as source of truth

**Status:** Accepted (Phase 0)

- **Problem:** reproducible installation and imports that behave the same in tests and in deployment.
- **Decision:** `src/industrial_ai/` package installed with `pip install -e .`; dependencies declared in `pyproject.toml` as optional groups per phase; a pinned `requirements.txt` generated in Phase 7 from the tested environment for Docker.
- **Reason:** the src layout prevents accidental imports from the working directory; optional groups keep early phases light; a lock file pins exactly what was tested.
- **Trade-offs:** two dependency files with different roles that must be kept in sync.

## D-013 · Relational store: SQLite via SQLAlchemy

**Status:** Accepted (Phase 0)

- **Problem:** store equipment master data and maintenance records.
- **Options:** JSON/YAML files; SQLite; PostgreSQL.
- **Decision:** SQLite through SQLAlchemy.
- **Reason:** real SQL with zero setup; moving to PostgreSQL only requires changing `DATABASE_URL`.
- **Trade-offs:** no concurrent writers; acceptable for a single-user PoC.

## D-014 · Deployment: Docker Compose, no Kubernetes

**Status:** Accepted (Phase 0)

- **Problem:** start the whole system reproducibly with one command.
- **Options:** manual setup; Docker Compose; Kubernetes.
- **Decision:** Docker Compose.
- **Reason:** reproducibility and simplicity for two or three services.
- **Trade-offs:** no scaling or self-healing, which a PoC does not need.

## D-015 · Final failure-prediction model

**Status:** Accepted (Phase 1, block 1.3) — logistic regression

- **Problem:** choose one model among several candidates whose scores may differ by less than the noise between folds.
- **Options:** the highest mean PR-AUC; the highest score on the test set; the simplest candidate that performs as well as the best.
- **Decision:** the simplest candidate (logistic regression < random forest < gradient boosting < XGBoost) whose mean PR-AUC under grouped 5-fold cross-validation is within one standard deviation of the best candidate's mean. Rule fixed before seeing the results.
- **Reason:** with 100 units, small differences in mean PR-AUC are within fold-to-fold variation; choosing by the maximum would reward noise. Choosing on the test set would leave no unbiased estimate of performance. Accuracy is not used (85% for a model that never predicts failure).
- **Trade-offs:** a slightly better model may be passed over when its advantage is smaller than the noise; the complexity order is a judgement call.
- **Result:** logistic regression (0.991 ± 0.001) is within the band of the best candidates (gradient boosting and XGBoost, 0.992 ± 0.004; band from 0.988), and is also the most stable and the fastest. Random forest (0.982 ± 0.006) falls outside the band. A linear model also makes block 1.5 simpler: its coefficients are a first, global explanation.

## D-016 · Sensor selection: explicit list from the EDA

**Status:** Accepted (Phase 1)

- **Problem:** several FD001 columns carry no information.
- **Options:** keep everything; a fitted variance threshold; an explicit list justified by the EDA.
- **Decision:** explicit list of 14 sensors (`SELECTED_SENSORS`).
- **Reason:** six sensors and `setting_3` are constant, sensor 6 only takes two values, and the operational settings are noise in a single regime. An explicit list is transparent and reviewable; a variance threshold would keep sensor 6 unless tuned for it.
- **Trade-offs:** the list is specific to FD001 and must be revisited for other subsets.

## D-017 · Features as a stateless transformer inside the model Pipeline

**Status:** Accepted (Phase 1)

- **Problem:** features must be computed identically in training and in the API (no training/serving skew), without leaking information.
- **Options:** compute features in a notebook and train on the result; a separate feature script; a scikit-learn transformer inside the Pipeline.
- **Decision:** `UnitHistoryFeatures` transformer wrapping `build_features`, placed first in the saved Pipeline.
- **Reason:** the saved artifact goes from raw sensor history to prediction, so the API cannot compute features differently. Features are stateless and past-only, so they behave the same inside or outside cross-validation folds.
- **Trade-offs:** the Pipeline expects each unit's complete history as input, not a single row; the API must pass the history of the unit.

## D-018 · Unit age (`cycle`) is a feature

**Status:** Accepted (Phase 1) — revised after the warm-up fix of the trend feature

- **Problem:** the age of a unit is known at prediction time and may improve scores, but it can also teach the model the lifetime distribution of this fleet instead of its health.
- **Options:** include `cycle`; exclude it; decide with a rule fixed before seeing the result.
- **Decision:** included, by applying the pre-registered rule: include only if grouped PR-AUC improves by more than one standard deviation across folds.
- **Evidence:** with the final feature definition, 0.9812 ± 0.0058 with `cycle` vs 0.9724 ± 0.0076 without (random forest, same five unit folds). The gain (+0.0088) exceeds both standard deviations, so the result does not depend on which one the rule refers to.
- **History:** the first evaluation, run with the trend warm-up artefact, gave a gain equal to one standard deviation (0.981 ± 0.006 vs 0.973 ± 0.008), so `cycle` was excluded. After fixing the artefact the rule was applied again, unchanged, to the corrected features, and the result changed. The rule decides, not the preference expressed earlier.
- **Reason:** following a pre-registered rule in both directions keeps the decision honest; a measurable gain above the fold-to-fold noise is evidence, not chance.
- **Trade-offs:** the model may partly rely on the age distribution of this simulated fleet (lifetimes of 128–362 cycles). Units with much longer or shorter lives, or another fleet, could be misjudged by their age. Recorded as a limitation; feature attributions in block 1.5 will show how much the model relies on `cycle`.

## D-019 · Decision threshold: recall ≥ 0.90 on out-of-fold scores

**Status:** Accepted (Phase 1, block 1.3)

- **Problem:** turning a risk score into an alarm needs a threshold, and 0.5 has no justification.
- **Options:** 0.5; maximise F1; a minimum recall with the best precision; minimise expected cost.
- **Decision:** the threshold with the highest precision among those reaching recall ≥ 0.90, chosen on the out-of-fold scores of the selected model.
- **Reason:** in this scenario a missed warning window is assumed to cost more than an unnecessary inspection. Out-of-fold scores come from models that never saw the unit, so the threshold is not tuned on optimistic training predictions or on the test set.
- **Trade-offs:** the 0.90 target is an assumption, not a cost analysis; real maintenance costs would lead to an expected-cost threshold. A recall target accepts more false alarms, which can cause alarm fatigue.

## D-020 · Fixed hyperparameters, no tuning in Phase 1

**Status:** Accepted (Phase 1, block 1.3)

- **Problem:** hyperparameters can be tuned, but the same 100 units are used to compare models and choose the threshold.
- **Options:** grid or random search on the same folds; nested cross-validation; fixed, reasonable defaults.
- **Decision:** fixed defaults (e.g. random forest with 300 trees and at least 5 samples per leaf; gradient boosting with learning rate 0.05 and 300 iterations), and early stopping disabled in `HistGradientBoostingClassifier`.
- **Reason:** tuning on the same folds used for comparison makes the comparison optimistic, and nested cross-validation adds cost and complexity that a portfolio PoC does not need. Scikit-learn's early stopping would hold out a random 10% of training rows, which is the leaky random split this project avoids.
- **Trade-offs:** candidates may be slightly below their best possible performance; nested cross-validation is listed as an improvement.

## D-021 · Anomaly detection against a healthy reference

**Status:** Accepted (Phase 1, block 1.4) for the design; the selected method is recorded after running notebook 02

- **Problem:** detect departures from healthy operation without failure labels, with a controlled false-alarm rate, as a complement to the supervised classifier.
- **Options:** Isolation Forest; one-class SVM; an autoencoder (PyTorch); statistical control charts (max |z-score|, Mahalanobis distance / Hotelling's T²).
- **Decision:** a detector fitted only on a healthy reference (cycles 20–60 of each training unit) using the deviation features (`delta_baseline`, `trend`, no `cycle`), with two candidate methods: max |z-score| as the simple baseline and Isolation Forest. Label-free threshold at the 99th percentile of the reference scores (1% false-alarm budget), alarms after 3 consecutive cycles. Selection rule fixed before the results: fewer missed units wins; with the same number, Isolation Forest only if its median warning is at least 5 cycles longer and its healthy false-alarm rate is not higher; otherwise max |z-score|.
- **Reason:** a control-chart baseline is transparent and familiar to plant engineers, so a more complex method has to earn its place. Isolation Forest needs no distributional assumption and captures combinations of features. A label-free threshold keeps the detector honestly unsupervised; labels are used only to evaluate it. Deviation features remove unit offsets, and excluding `cycle` avoids flagging age instead of behaviour.
- **Trade-offs:** the healthy-reference assumption may label early degradation as normal; a threshold computed on the reference itself is optimistic (measured out-of-fold on unseen units); max |z-score| ignores correlations between features (a Mahalanobis distance would use them, at the cost of estimating a 28 × 28 covariance); Isolation Forest scores saturate outside the training range and dilute deviations confined to a few features. An autoencoder is listed as future work and would only be adopted if it beats both under the same protocol.
