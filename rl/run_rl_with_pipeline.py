import sys
from pathlib import Path
import joblib
import numpy as np
import logging

# Ensure repo root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("run_rl_with_pipeline")
logging.basicConfig(level=logging.INFO)

# Try a few likely artifact locations/names
MODEL_CANDIDATES = [
    PROJECT_ROOT / "file_build" / "sales_prediction_model.pkl",
    PROJECT_ROOT / "models" / "sales_prediction_model.pkl",
    PROJECT_ROOT / "file_build" / "model.pkl",
]
SCALER_CANDIDATES = [
    PROJECT_ROOT / "file_build" / "scaler.pkl",
    PROJECT_ROOT / "models" / "scaler.pkl",
    PROJECT_ROOT / "file_build" / "preprocessor.pkl",
    PROJECT_ROOT / "file_build" / "preprocessor.joblib",
]

model_path = None
for p in MODEL_CANDIDATES:
    if p.exists():
        model_path = p
        break

scaler_path = None
for p in SCALER_CANDIDATES:
    if p.exists():
        scaler_path = p
        break

if model_path is None:
    raise FileNotFoundError("No trained model found. Looked at: " + ", ".join(str(p) for p in MODEL_CANDIDATES))

model = joblib.load(model_path)
logger.info(f"Loaded model from {model_path}")
scaler = None
if scaler_path is not None:
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
    except Exception:
        scaler = None
        logger.warning(f"Failed to load scaler from {scaler_path}; continuing without scaler")

# Import RL agent
from rl.rl_agent import RL_AGENT

import pandas as pd

def predict_pipeline(observation):
    """Robust wrapper: accepts list, ndarray, pandas Series/DataFrame and returns numeric prediction list.

    Returns a flat Python list of predictions.
    """
    # If user passed a pandas DataFrame or Series
    if isinstance(observation, pd.DataFrame):
        X = observation.values
    elif isinstance(observation, pd.Series):
        X = observation.values.reshape(1, -1)
    else:
        X = np.array(observation)

    # Normalize shape to 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # If the model or scaler expects column names (ColumnTransformer), provide a DataFrame
    X_trans = X
    feature_names = None
    # try to get feature names from scaler or model (pipelines often store feature names)
    for candidate in (scaler, model):
        if candidate is None:
            continue
        if hasattr(candidate, "feature_names_in_"):
            feature_names = list(getattr(candidate, "feature_names_in_"))
            break
        # If it's a pipeline, check named steps
        if hasattr(candidate, "named_steps"):
            for step in candidate.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    feature_names = list(getattr(step, "feature_names_in_"))
                    break
            if feature_names:
                break

    if feature_names is not None:
        try:
            # Only convert to DataFrame when shapes align
            if X.ndim == 2 and X.shape[1] == len(feature_names):
                X_df = pd.DataFrame(X, columns=feature_names)
                X_trans = X_df
            else:
                # If dims mismatch, still try to build DataFrame if possible
                if X.ndim == 1 and len(feature_names) == X.shape[0]:
                    X_df = pd.DataFrame([X], columns=feature_names)
                    X_trans = X_df
        except Exception:
            logger.debug("Could not convert to DataFrame for named columns; continuing with ndarray")

    # If scaler is present and not already a dataframe-aware transformer, try transform
    if scaler is not None:
        try:
            X_trans = scaler.transform(X_trans)
        except Exception:
            try:
                X_trans = scaler.transform(np.asarray(X_trans))
            except Exception:
                logger.exception("Failed to transform input with scaler; passing raw X to model")
                X_trans = X

    try:
        preds = model.predict(X_trans)
    except Exception as exc:
        msg = str(exc)
        # Handle ColumnTransformer expecting DataFrame column names
        if "only supported for dataframes" in msg and feature_names is not None:
            try:
                arr = np.asarray(X_trans)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                n_req = len(feature_names)
                n_have = arr.shape[1]
                if n_have < n_req:
                    pad = np.zeros((arr.shape[0], n_req - n_have))
                    arr2 = np.concatenate([arr, pad], axis=1)
                else:
                    arr2 = arr[:, :n_req]
                X_df = pd.DataFrame(arr2, columns=feature_names)
                preds = model.predict(X_df)
            except Exception:
                logger.exception("Failed to recover from dataframe column-indexing error")
                raise
        else:
            raise
    # Convert numpy types to plain python
    try:
        preds_list = [float(p) for p in np.asarray(preds).ravel()]
    except Exception:
        preds_list = list(np.asarray(preds).ravel())
    return preds_list


def main(total_budget: int = 1000, episodes: int = 20, action_space: int = 10):
    # Determine action/state sizes from the model input shape if possible
    sample_input = np.zeros((1, 3))
    try:
        # Try to infer n_features
        n_features = sample_input.shape[1]
    except Exception:
        n_features = 3

    state_size = n_features

    # Instantiate RL agent and train locally
    agent = RL_AGENT(predict_pipeline, action_space=action_space, epsilon=0.9, epsilon_decay=0.99, min_epsilon=0.01, state_size=state_size)
    logger.info(f"Starting local RL training (short run, episodes={episodes})")
    policy = agent.train(episodes=episodes)

    # Save policy
    out_dir = PROJECT_ROOT / "file_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = out_dir / "rl_policy.pkl"
    joblib.dump(policy, policy_path)
    logger.info(f"Saved RL policy to {policy_path}")

    # Derive best action from policy
    if not policy:
        raise RuntimeError("RL policy is empty")
    best_action = max(policy.items(), key=lambda kv: kv[1])[0]
    # Normalize to tuple
    if not isinstance(best_action, (list, tuple)):
        best_action = (best_action,)

    # Convert discrete action integers into allocations that sum to total_budget
    # Interpret the action integers as relative weights and normalize
    levels = [max(0, int(a)) for a in best_action]
    total_levels = sum(levels)
    if total_levels > 0:
        fracs = [lvl / total_levels for lvl in levels]
    else:
        # fallback to equal split
        fracs = [1.0 / len(levels) for _ in levels]

    raw_allocs = [frac * float(total_budget) for frac in fracs]
    int_allocs = [int(x) for x in raw_allocs]
    # Fix rounding so sum equals total_budget by distributing remainder
    remainder = int(total_budget) - sum(int_allocs)
    i = 0
    while remainder > 0:
        int_allocs[i % len(int_allocs)] += 1
        remainder -= 1
        i += 1
    while remainder < 0:
        # If somehow exceeding, subtract from last positive entries
        for j in range(len(int_allocs)-1, -1, -1):
            if int_allocs[j] > 0 and remainder < 0:
                int_allocs[j] -= 1
                remainder += 1
                if remainder == 0:
                    break

    # Save and return
    result = {
        "policy_path": str(policy_path),
        "policy_sample": list(policy.items())[:10],
        "allocation": int_allocs,
        "total_budget": int(total_budget),
    }
    # Compute expected sales for the allocation using the prediction pipeline
    try:
        # prediction expects a 2D observation; allocation is list per-service budgets
        predicted = predict_pipeline(int_allocs)
        predicted_sales = float(predicted[0]) if predicted else None
    except Exception:
        logger.exception("Failed to compute predicted sales for allocation")
        predicted_sales = None

    # Load historical average sales from the dataset if available
    historical_mean = None
    try:
        sample_csv = PROJECT_ROOT / "data" / "sample.csv"
        if sample_csv.exists():
            df = pd.read_csv(sample_csv)
            if "Sales ($)" in df.columns:
                historical_mean = float(df["Sales ($)"].mean())
            else:
                # fallback: take last column as target
                historical_mean = float(df.iloc[:, -1].mean())
    except Exception:
        logger.exception("Failed to compute historical mean sales")

    result["predicted_sales"] = predicted_sales
    result["historical_mean_sales"] = historical_mean
    if predicted_sales is not None and historical_mean is not None:
        result["delta"] = predicted_sales - historical_mean
        try:
            result["pct_change"] = (result["delta"] / historical_mean) * 100.0 if historical_mean != 0 else None
        except Exception:
            result["pct_change"] = None
    else:
        result["delta"] = None
        result["pct_change"] = None
    print("Saved RL policy to:", policy_path)
    print("Sample policy items:", result["policy_sample"])
    print("Allocation:", result["allocation"], "sum=", sum(result["allocation"]))
    return result


if __name__ == "__main__":
    main()
