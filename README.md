# Industrial AI Copilot

![CI](https://github.com/maria2332/industrial-ai-copilot/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**A decision-support prototype that combines retrieval-augmented generation (RAG) over technical documentation with predictive maintenance on equipment sensor data.**

> [!IMPORTANT]
> All industrial documentation used in this project is synthetic and created exclusively for demonstration purposes. This is an independent, educational proof of concept: it is not affiliated with or built for any company, it uses no private or confidential data, and it must not be used for real operational decisions.

**Status:** 🚧 Phase 0 — architecture and scaffolding. See the [roadmap](#roadmap).

## Overview

Industrial AI Copilot helps plant engineers answer two kinds of questions:

- **"What does the documentation say?"** — e.g. *What is the maximum exhaust gas temperature of GT-101?* Answered with RAG, always citing document, page and section.
- **"What is the equipment doing?"** — e.g. *Does GT-101 show a high risk of failure?* Answered with machine-learning models trained on sensor data, together with the variables that most influenced the prediction.

A read-only agent combines both when a question needs it, e.g. *Is GT-101 behaving anomalously, and which maintenance procedure applies?*

The guiding principle is **no evidence → no confident answer**: when the system cannot find supporting evidence, it says so and indicates which document or data would be needed.

## Problem

ACME Process Energy (a **fictitious** company) operates process plants whose critical rotating equipment includes two aeroderivative gas turbine units, **GT-101** and **GT-102**. Technical knowledge is spread across long manuals and procedures, and maintenance is mostly calendar-based or reactive. Engineers lose time searching for exact values and get little early warning of degradation.

Typical questions:

- What are the operating conditions of GT-101?
- What is the maximum allowed exhaust gas temperature?
- What procedure must be followed after a high-temperature alarm?
- What is the difference between GT-101 and GT-102?
- Is there a recent anomaly in the behaviour of GT-101?
- Which documents support this answer?

## Architecture

```mermaid
flowchart LR
    U["Engineer"] --> UI["Web UI"] --> API["FastAPI"]
    API --> R{"Router"}
    R -->|docs only| RAG["RAG pipeline"]
    R -->|multi-step| AG["Agent"]
    AG --> RAG
    AG --> ML["ML models"]
    AG --> EQ["Equipment registry"]
    API --> ML
    RAG --> VDB[("Vector DB")]
    RAG --> LLM["LLM"]
    AG --> LLM
```

Full design in [docs/architecture.md](docs/architecture.md) · design decisions and trade-offs in [docs/decisions.md](docs/decisions.md).

## Features

- [ ] Failure-risk prediction with explanation of the main contributing variables (Phase 1)
- [ ] Unsupervised anomaly detection on sensor data (Phase 1)
- [ ] Synthetic, internally consistent technical documentation (Phase 2)
- [ ] RAG with citations, evidence gate and numeric grounding check (Phase 3)
- [ ] Read-only agent with real tools (Phase 4)
- [ ] REST API with validated schemas and Swagger docs (Phase 5)
- [ ] Simple web UI (Phase 6)
- [ ] One-command deployment with Docker Compose (Phase 7)

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| Data & ML | pandas, NumPy, SciPy, scikit-learn, XGBoost, SHAP |
| LLM | Ollama (local, default) or OpenAI, behind a provider interface |
| Embeddings | sentence-transformers (Hugging Face) |
| Vector store | ChromaDB |
| Relational store | SQLite via SQLAlchemy (PostgreSQL-ready) |
| API | FastAPI + Pydantic |
| UI | Streamlit |
| Quality | pytest, Ruff, pre-commit, GitHub Actions |
| Deployment | Docker, Docker Compose |

## Dataset

Sensor data comes from the **NASA C-MAPSS** turbofan engine degradation simulation, subset **FD001**: run-to-failure trajectories of 100 training engines with 3 operational settings and 21 sensor channels, under one operating condition and one fault mode.

C-MAPSS simulates an aircraft turbofan. It is used here as a **proxy** for the degradation of an aeroderivative gas turbine, which is derived from aircraft engines and shares its gas-generator core. This is an approximation, not a model of a real industrial unit (see [Limitations](#limitations)). The fictitious assets GT-101 and GT-102 are mapped to C-MAPSS engine units in the equipment registry.

The raw data is not redistributed in this repository; it is downloaded by a script (Phase 1).

> Reference: A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", *International Conference on Prognostics and Health Management (PHM)*, 2008.

## ML Pipeline

*Implemented in Phase 1.* Leakage-safe pipeline: validation → label definition (failure within the next N cycles) → past-only rolling features per engine → split by engine unit → baseline comparison → threshold selection → explainability → versioned model artifacts. Details in [docs/ml.md](docs/ml.md).

## RAG Pipeline

*Implemented in Phase 3.* Ingestion (PDF → text → metadata → chunks → embeddings → vector store) is separated from retrieval and generation. Answers include document, page, section and the supporting fragments. Details in [docs/rag.md](docs/rag.md).

## Agent

*Implemented in Phase 4.* A small, auditable tool-calling loop with read-only tools: `search_documents`, `get_equipment_information`, `predict_failure`, `detect_anomaly`, `compare_equipment` and `retrieve_maintenance_procedure`. Details in [docs/agent.md](docs/agent.md).

## API

*Implemented in Phase 5.* Interactive Swagger documentation will be available at `http://localhost:8000/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/ask` | Question answering (RAG or agent) |
| POST | `/predict` | Failure-risk prediction |
| POST | `/anomaly` | Anomaly score |
| POST | `/documents/ingest` | Ingest a new document |
| GET | `/equipment/{equipment_id}` | Equipment data sheet |
| POST | `/compare` | Compare two units |
| GET | `/health` | Health check |

## Demo

*A 5-minute walkthrough will be added in Phase 10.*

## Installation

Development setup (current phase):

```bash
git clone https://github.com/maria2332/industrial-ai-copilot.git
cd industrial-ai-copilot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
cp .env.example .env               # Windows: copy .env.example .env
```

## Usage

*Available from Phase 5 (API) and Phase 7 (`docker compose up`).*

## Testing

```bash
pytest
ruff check .
ruff format --check .
```

The same checks run automatically on GitHub Actions for every push to `main` and every pull request.

## Evaluation

*Results will be reported in Phase 9.* Planned metrics:

- **ML:** PR-AUC, ROC-AUC, precision, recall, F1 and confusion matrix at the selected threshold; for anomaly detection, false-alarm rate on early life and detection lead time before failure.
- **RAG:** retrieval hit@k and MRR, exact numeric correctness (with units), citation correctness, groundedness and correct abstention on unanswerable questions.
- **Agent:** tool-selection accuracy, task completion and documented failure cases.
- **API:** p50/p95 latency and error-path coverage.

## Limitations

- Sensor data comes from a public **simulation** of an aircraft turbofan, used as a proxy; real plant data would differ in noise, sampling, operating regimes and failure modes.
- All technical documentation is **synthetic**.
- Models are **not validated for production** and the decision threshold reflects assumptions, not a real cost analysis.
- The LLM can make mistakes; mitigations reduce but do not eliminate this risk.
- RAG quality depends on the quality and coverage of the documentation.
- Anomaly detection and feature attributions show **correlations, not causes**.
- The system must not be used for real operational or safety decisions.

## Safety Considerations

- **Human-in-the-loop:** the system informs an engineer; it never acts.
- **No autonomous control:** no tool can modify equipment, parameters or plant state. All tools are read-only.
- **No safety-critical decisions:** answers are decision support only and state their uncertainty.
- **No private company data and no personal data** are used.

## Future Work

Computer vision for inspection, OCR, multimodal RAG, digital twins, real-time streaming (Kafka), experiment tracking (MLflow), PostgreSQL + pgvector, drift monitoring, human feedback loops, systematic LLM evaluation, fine-tuning where justified, multimodal agents and other C-MAPSS subsets (FD002–FD004).

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Architecture and scaffolding | 🚧 In progress |
| 1 | Dataset + ML | ⏳ |
| 2 | Synthetic technical documents | ⏳ |
| 3 | RAG | ⏳ |
| 4 | Agent | ⏳ |
| 5 | FastAPI | ⏳ |
| 6 | Frontend | ⏳ |
| 7 | Docker | ⏳ |
| 8 | Testing | ⏳ |
| 9 | Evaluation | ⏳ |
| 10 | Documentation and demo | ⏳ |
| 11 | Interview preparation | ⏳ |

## Author

**María Arribas Ballesteros** — Mathematical Engineering graduate (Artificial Intelligence specialisation), CEU San Pablo University.

[LinkedIn](<https://www.linkedin.com/in/mar%C3%ADa-arribas-ballesteros>) · [GitHub](https://github.com/maria2332)
