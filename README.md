# EconoCausal — Dynamic Pricing via Double Machine Learning

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![EconML](https://img.shields.io/badge/EconML-DML-orange)
![DoWhy](https://img.shields.io/badge/DoWhy-Causal%20Graph-lightgrey)
![Status](https://img.shields.io/badge/status-in%20development-yellow)

> Moves beyond predictive ML into **causal & prescriptive AI**: instead of
> predicting who *will* buy, EconoCausal estimates who will buy **because**
> of a discount, and optimally allocates a fixed marketing budget to them.

## 1. What problem this solves

A normal ML churn/propensity model answers *"who is likely to buy / churn?"*.
It does **not** answer the question that actually matters for a marketing
budget: *"would THIS user have bought anyway, or did the discount cause it?"*

EconoCausal answers that second question by estimating the **Individual
Treatment Effect (ITE)** — how much a specific user's purchase probability
*changes* because of a treatment (e.g. a $10 vs $20 discount, or a
promotional email). It then feeds those ITEs into a budget optimizer that
allocates the discount money only to **Persuadables** — people who buy
*because* of the discount, not people who would have bought anyway
("Sure Things") or people who will never buy regardless ("Lost Causes").

## 2. The four user segments (why correlation-based ML fails)

| Segment | Buys if treated | Buys if NOT treated | Should you spend on them? |
|---|---|---|---|
| Sure Things | Yes | Yes | No — wasted discount |
| Lost Causes | No | No | No — discount doesn't help |
| Persuadables | Yes | No | **Yes — this is the ROI** |
| Sleeping Dogs | No | Yes | Never — discount actively hurts |

A standard classifier can only estimate P(buy | treated), which conflates
all four groups. DML explicitly separates the *causal* component.

## 3. Dataset

We use **Kevin Hillstrom's MineThatData E-Mail Analytics dataset** — the
standard public benchmark for uplift/causal marketing models (64,000
customers, randomized 3-arm email experiment: Men's email / Women's email /
No email, with `visit`, `conversion`, `spend` outcomes and 8 pre-treatment
covariates such as recency, history, channel, zip code).

- Official source (direct CSV): `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`
- Same data via a maintained Python package (recommended, handles the
  download + caching for you): `scikit-uplift` → `sklift.datasets.fetch_hillstrom()`
- Background write-up: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

Because it's a true randomized experiment, it's ideal for validating a DML
pipeline: you can check the model's estimated Average Treatment Effect
against the known randomized-experiment ground truth before trusting it on
observational (non-randomized) data.

`data/download_data.py` pulls this automatically, and falls back to a
synthetic generator with a known ground-truth treatment effect (useful for
offline dev, or if the network is unavailable) so the rest of the pipeline
never blocks on connectivity.

## 4. Pipeline (maps to the week-wise plan)

```
data/download_data.py        → raw dataframe (covariates X, treatment T, outcome Y)
        │
src/causal_graph.py          → DoWhy causal model + DAG (confounders/treatment/outcome)
        │
src/dml_model.py              → EconML LinearDML / CausalForestDML → ITE per user
        │
src/propensity_matching.py    → propensity scores + matching, used both as a DML
        │                        nuisance component and as a sanity-check estimator
        │
(mid-project) refutation      → DoWhy refutation tests, inside src/causal_graph.py
        │
src/optimization.py            → SciPy solver: allocate $ budget across users by ITE
        │
src/api.py                     → FastAPI REST wrapper around the whole pipeline
        │
dashboard/index.html           → Plotly dashboard: Qini/uplift curve + allocation table
```

## 5. Key modules → files

| Key module from the spec | File |
|---|---|
| Causal Inference Engine (DoWhy + EconML DML) | `src/causal_graph.py`, `src/dml_model.py` |
| Propensity Score Matching | `src/propensity_matching.py` |
| Optimization Solver (SciPy) | `src/optimization.py` |
| REST API packaging | `src/api.py` |
| Experimentation UI (Plotly) | `dashboard/index.html` |

## 6. How to run

```bash
pip install -r requirements.txt
python data/download_data.py          # writes data/hillstrom.csv
python src/causal_graph.py            # prints DAG + refutation test results
python src/dml_model.py               # trains DML, writes data/ite_scores.csv
python src/optimization.py            # writes data/allocation.csv (final prescription)
uvicorn src.api:app --reload          # serves the API on :8000
open dashboard/index.html             # open directly in a browser, or serve via API
```

## 7. Notes / what's simplified vs. a production build

- `dashboard/index.html` is a single self-contained file (Plotly via CDN) so
  it runs with zero build tooling — swap it for a React app later without
  changing any backend contract; the API already returns JSON the same shape.
- The optimizer uses a simple knapsack-style LP relaxation (`scipy.optimize.linprog`)
  rather than a custom mixed-integer solver — swap in `PuLP`/`OR-Tools` if you
  need strict integer discount tiers at scale.
- Data drift detection (Week 4) is stubbed in `src/api.py` as a
  population-stability-index (PSI) check you can wire to a scheduler.

## 8. Project structure

```
econocausal/
├── data/
│   └── download_data.py       # Hillstrom dataset fetch + synthetic fallback
├── src/
│   ├── causal_graph.py        # DoWhy DAG + refutation tests
│   ├── propensity_matching.py # Propensity Score Matching
│   ├── dml_model.py           # EconML CausalForestDML -> per-user ITE
│   ├── optimization.py        # SciPy budget allocation solver
│   └── api.py                 # FastAPI REST layer
├── dashboard/
│   └── index.html             # Plotly experimentation dashboard
├── requirements.txt
├── LICENSE
└── README.md
```

## 9. Contributing

Issues and PRs are welcome — in particular, swapping the LP-relaxation
optimizer for a proper MILP solver (OR-Tools/PuLP) and adding a
continuous-treatment DML estimator for a true non-linear dose-response
curve are the two most useful next steps.

## 10. License

MIT — see [LICENSE](LICENSE).
