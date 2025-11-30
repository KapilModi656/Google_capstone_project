# Google Capstone Project

## Overview

This repository contains a capstone project that demonstrates an agentic AI pipeline for marketing budget allocation. The app helps businesses decide how to split a fixed marketing budget across channels (TV, Radio, Newspaper, etc.) to maximize sales using a combination of:

- Data-driven machine learning models (trained on historical spend vs sales), and
- A lightweight local reinforcement learning (Q-learning) agent that proposes integer allocations summing to a specified budget.

The project uses Google ADK-style agents to coordinate pipeline steps (data analysis, transformation, training, and prediction) and a Streamlit UI to run the pipeline and show results.

## Quickstart

1. Install dependencies (use a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

3. For a quick experiment, upload the sample dataset at `data/sample.csv` (included) or provide your own CSV with the same column structure.

## Data format

The pipeline expects a CSV with columns representing spent budgets on marketing channels and a sales/profit column named `Sales` (case-sensitive). Example columns:

- `TV Ad Budget ($)`
- `Radio Ad Budget ($)`
- `Newspaper Ad Budget ($)`
- `Sales` (target)

If your CSV has different column names, rename them to match the expected names before uploading.

## API / Programmatic Usage

The repo exposes a callable function you can use from Python (used by Streamlit internally):

- `run_data_analyst(dataset_path, total_budget, episodes)`

This function runs the ADK agent pipeline: it analyzes and transforms the dataset, trains ML models, saves artifacts to `file_build/`, makes a prediction for a test observation, and then runs the local RL routine that produces integer allocations summing to `total_budget`.

If you prefer a class-based entrypoint, see `AI_ML_BUILDER/src/data_analyst_fixed.py` which implements a compact pipeline and returns a dictionary with `adk_result`, `artifacts`, and `rl_result`.

## Where artifacts are saved

Pipeline artifacts (trained models and metrics) are written to `file_build/` by the pipeline. Typical files:

- `file_build/model.pkl` — primary trained model (pickle)
- `file_build/evaluation_metrics.pkl` — model metrics
- `file_build/rf_model.pkl` — random-forest model (if trained)

## How it works (high level)

1. ADK root agent receives a user request (dataset path + goal).
2. Data analyst agent issues small python snippets to a safe `execute_code_tool` that runs them and returns serializable outputs.
3. Data is normalized, models are trained (LinearRegression and RandomForestRegressor by default), and model artifacts are saved.
4. Prediction is made for a sample observation (example: `[[5.1, 3.5, 1.4]]`). The numeric prediction is returned alongside a log of actions.
5. A local RL routine ingests saved artifacts and computes an integer allocation vector that sums to the provided budget.

## Streamlit UI

Open the UI with `streamlit run streamlit_app.py`. The UI calls `run_data_analyst(...)` and displays:

- Pipeline logs and status
- Model training metrics
- Numeric prediction for the test observation
- RL allocation suggestions (integers that sum to the provided total budget)

## Development notes

- Keep registered ADK tool callables simple: tools should accept primitive JSON-serializable parameters (e.g., `code: str`) to avoid provider-side serialization errors.
- The LLM wrapper uses provider-prefixed model ids (e.g., `mistral/codestral-latest`) via the litellm-style integration.

## Files of interest

- `AI_ML_BUILDER/src/data_analyst_fixed.py` — a sanitized fallback pipeline module (good reference for wiring a safe pipeline).
- `AI_ML_BUILDER/src/Data_Analyst.py` — original orchestrator (may need harmonization with the fixed version).
- `llm.py` — LLM wrapper using provider-prefixed model ids.
- `utils.py` — helpers for safe code execution and serialization.
- `rl/run_rl_with_pipeline.py` — local RL runner (now refactored to a class-based interface in this workspace).
- `streamlit_app.py` — Streamlit front-end.

## Troubleshooting

- If you see import errors (e.g., `ModuleNotFoundError: No module named 'llm'`), ensure your Python path includes the project root and you installed the project's dependencies.
- If the ADK agents fail with JSON serialization errors, check that any registered tool callable has only simple typed parameters (strings, numbers, booleans) and no complex Python objects in annotations or defaults.

## Contributing

Contributions welcome. Open an issue or submit a PR explaining the change. Tests and small demo notebooks are appreciated.

## License

This project does not include a license file in the repository; add one if you plan to distribute or open-source the code.

---
Last updated: 2025-11-30
