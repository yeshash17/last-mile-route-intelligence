# Last-Mile Route Intelligence Engine
## Session Handoff — Continue in Claude Code

---

## What this project is

A **Decision Intelligence system** for a delivery company that optimises driver routes, predicts delivery failures before dispatch, and learns from every outcome to get smarter over time.

This is not just a route optimiser. It is a full DI pipeline that answers:
> *"In what order should driver X visit their stops today — and which stops are at risk of failing?"*

---

## DI Canvas (the business logic)

| Layer | Definition |
|---|---|
| **Objective** | Maximise first-attempt delivery success rate and minimise cost per drop |
| **Decision** | Optimal stop sequence per driver per shift (pre-shift + real-time re-route) |
| **Input** | Delivery manifest, live traffic API, driver GPS, historical delivery outcomes |
| **Output** | Ordered route with per-stop ETAs, failure risk scores, recommended actions |
| **Action** | Route pushed to driver app; dispatcher alerted for high-risk stops; customer SMS sent |
| **Feedback** | Actual delivery outcomes logged → nightly model retrain → predictions improve |

---

## System Architecture (5 layers)

```
┌─────────────────────────────────────────────────────────┐
│  DATA SOURCES                                           │
│  Manifest CSV │ Traffic API (Google/HERE) │ Driver GPS  │
│  History DB (SQLite → PostgreSQL in prod)               │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  PROCESSING & ML                                        │
│  Feature engineering │ Distance matrix │ ML models      │
│  (data/features.py)    (haversine now,   LightGBM +     │
│                         Maps API later)  XGBoost        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  DECISION ENGINE                                        │
│  VRP Optimizer (OR-Tools) │ Constraint Handler          │
│  Time windows, capacity, shift hours enforced           │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  FASTAPI DECISION SERVER                                │
│  /optimize-route  /re-route  /risk-score  /health       │
└───────────────┬───────────────────────┬─────────────────┘
                │                       │
┌───────────────▼──────┐  ┌────────────▼──────────────────┐
│  Driver app          │  │  Dispatcher dashboard         │
│  Route + stop brief  │  │  Fleet map + SLA alerts       │
└──────────────────────┘  └───────────────────────────────┘
```

---

## Files Created This Session

```
route-intelligence/
├── config.py                    ← env vars, all thresholds
├── requirements.txt             ← all dependencies
├── data/
│   └── features.py              ← engineer ML features per stop
├── models/
│   └── failure_predictor.py     ← LightGBM P(fail) classifier, train + predict
├── optimizer/
│   └── vrp_solver.py            ← OR-Tools VRPTW solver ← START HERE
├── api/
│   ├── main.py                  ← FastAPI app, /optimize-route endpoint
│   └── schemas.py               ← Pydantic request/response models
└── feedback/
    └── collector.py             ← log delivery outcomes, calibration report
```

---

## What Is NOT Built Yet (next session)

### Priority 1 — Must build to have a working system

| File | What it does | Notes |
|---|---|---|
| `data/loader.py` | Load delivery manifest CSV + query history DB | Connect to LaDe dataset |
| `data/distance_matrix.py` | Build travel-time matrix via Google Maps API | Replace haversine approximation in `api/main.py` |
| `models/eta_predictor.py` | XGBoost: predict dwell time per stop | Feeds into VRP `dwell_mins` field |
| `models/trainer.py` | Train + save both models from real data | Run once before starting API |
| `optimizer/constraints.py` | Road restrictions, ULEZ zones, weight limits | Plugs into VRP solver |

### Priority 2 — Makes it a full DI system

| File | What it does |
|---|---|
| `api/routes.py` | `/re-route` endpoint for mid-shift re-optimisation |
| `dashboard/app.py` | Streamlit dispatcher view: fleet map + SLA risk alerts |
| `feedback/retrainer.py` | Nightly scheduled model refresh using new outcomes |

### Priority 3 — Production hardening

- Swap SQLite → PostgreSQL
- Add Redis + Celery for async route generation
- Add JWT auth to FastAPI
- Containerise with Docker
- Add proper logging + Sentry error tracking

---

## Build Order for Next Session

```
1. pip install -r requirements.txt
2. python optimizer/vrp_solver.py          # smoke test — should print a 4-stop demo route
3. python models/failure_predictor.py      # trains on synthetic data, saves model
4. uvicorn api.main:app --reload           # start API
5. curl http://localhost:8000/docs         # open Swagger UI in browser
6. Build data/loader.py                   # connect real data
7. Build data/distance_matrix.py          # Google Maps API call
8. Retrain failure_predictor with real data
9. Build dashboard/app.py                 # streamlit run dashboard/app.py
10. Build feedback/retrainer.py           # close the loop
```

---

## Key Design Decisions (rationale)

**Why OR-Tools for routing?**
Production-grade, Google-maintained, handles VRPTW natively. Guided Local Search finds near-optimal solutions in 30s even for 200 stops.

**Why LightGBM for failure prediction?**
Handles class imbalance well (only ~15% of deliveries fail). Fast inference — scores 200 stops in <50ms. Interpretable feature importances.

**Why haversine now, Google Maps later?**
Haversine gets you a working system in hours without API costs. Swap in `data/distance_matrix.py` once the pipeline is proven.

**Why FastAPI + Pydantic?**
Auto-generates Swagger docs. Pydantic validates all inputs before they reach the solver — prevents silent bugs in production.

**Why SQLite for feedback?**
Zero setup. Swap to PostgreSQL when you hit ~50k records or need concurrent writes.

---

## Dataset

Use **LaDe (Cainiao/Alibaba)** for training the ML models:
```
huggingface-cli download Cainiao-AI/LaDe --repo-type dataset
```
10.6M real packages, 21k couriers, 6 months. Columns include:
- `courier_id`, `lng`, `lat`, `finish_time`, `accept_time`, `task_type`

Map LaDe columns → `failure_predictor.py` FEATURES in `data/loader.py`.

---

## KPIs to track (baseline before going live, measure after)

| Metric | Definition | Target |
|---|---|---|
| On-time rate % | Stops delivered within promised window | > 95% |
| FADR % | First-attempt delivery rate | +5pp vs baseline |
| ETA accuracy | % stops within ±15 min of prediction | > 85% |
| Cost per drop £ | Total ops cost / successful deliveries | −10–15% |
| Model AUC | Failure predictor validation AUC | > 0.80 |

---

## Context for Claude Code prompt

When starting a new Claude Code session, paste this:

> "I am building a last-mile route intelligence engine for a delivery company.
> The project is in `route-intelligence/`. Key files already exist:
> `optimizer/vrp_solver.py` (OR-Tools VRPTW), `models/failure_predictor.py`
> (LightGBM), `api/main.py` (FastAPI), `data/features.py`, `feedback/collector.py`.
> See SESSION_HANDOFF.md for full context.
> Today I want to build: [pick from Priority 1 list above]."
