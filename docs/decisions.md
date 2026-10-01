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

**Status:** Pending — decided in Phase 1

Chosen after comparing logistic regression, random forest, gradient boosting and XGBoost with grouped cross-validation, primarily on PR-AUC and on precision at the recall required by the operating threshold, not on accuracy.
