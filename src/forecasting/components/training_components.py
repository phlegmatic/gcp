"""Pipeline 2 components: parallel single-model training + ensembling.

The two training components are intentionally identical in signature so they
can run as parallel branches of the DAG; the ensemble component fans them in.
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

PY_IMAGE = "python:3.11-slim"
SKLEARN_PKGS = [
    "scikit-learn==1.5.1",
    "pandas==2.2.2",
    "numpy==1.26.4",
    "pyarrow==16.1.0",
]


@dsl.component(base_image=PY_IMAGE, packages_to_install=SKLEARN_PKGS)
def train_single_model(
    training_data: Input[Dataset],
    model_name: str,
    val_fraction: float,
    trained_model: Output[Model],
    train_metrics: Output[Metrics],
) -> float:
    """Train one model (`ridge` or `random_forest`) and serialize it.

    Returns the validation RMSE so the downstream ensemble step can compute
    inverse-error weights. Runs as a parallel DAG branch.
    """
    import pickle

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    df = pd.read_parquet(training_data.path)
    target = "demand"
    feature_cols = [c for c in df.columns if c not in ("ds", target)]

    split = int(len(df) * (1 - val_fraction))
    x_train, y_train = df[feature_cols].iloc[:split], df[target].iloc[:split]
    x_val, y_val = df[feature_cols].iloc[split:], df[target].iloc[split:]

    if model_name == "ridge":
        est = Ridge(alpha=1.0)
    elif model_name == "random_forest":
        est = RandomForestRegressor(
            n_estimators=100, max_depth=8, n_jobs=1, random_state=42
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    est.fit(x_train, y_train)
    preds = est.predict(x_val)
    rmse = float(np.sqrt(np.mean((y_val.to_numpy() - preds) ** 2)))
    mae = float(np.mean(np.abs(y_val.to_numpy() - preds)))

    train_metrics.log_metric("model_name_hash", float(hash(model_name) % 1000))
    train_metrics.log_metric("val_rmse", rmse)
    train_metrics.log_metric("val_mae", mae)

    trained_model.metadata["model_name"] = model_name
    trained_model.metadata["val_rmse"] = rmse
    with open(trained_model.path, "wb") as fh:
        pickle.dump(
            {"model_name": model_name, "estimator": est, "feature_cols": feature_cols},
            fh,
        )

    return rmse


@dsl.component(
    base_image=PY_IMAGE,
    packages_to_install=SKLEARN_PKGS + ["google-cloud-storage==2.17.0"],
)
def build_ensemble(
    training_data: Input[Dataset],
    model_a: Input[Model],
    model_b: Input[Model],
    rmse_a: float,
    rmse_b: float,
    val_fraction: float,
    model_output_uri: str,
    ensemble_model: Output[Model],
    ensemble_metrics: Output[Metrics],
) -> str:
    """Fan-in step: wait for both models, build inverse-RMSE weighted ensemble.

    Evaluates the ensemble on the validation window, serializes the final
    bundle to `ensemble_model` AND uploads it to `model_output_uri` on GCS.
    Returns the GCS URI of the persisted ensemble.
    """
    import pickle

    import numpy as np
    import pandas as pd
    from google.cloud import storage

    with open(model_a.path, "rb") as fh:
        bundle_a = pickle.load(fh)
    with open(model_b.path, "rb") as fh:
        bundle_b = pickle.load(fh)

    df = pd.read_parquet(training_data.path)
    target = "demand"
    feature_cols = bundle_a["feature_cols"]
    split = int(len(df) * (1 - val_fraction))
    x_val = df[feature_cols].iloc[split:]
    y_val = df[target].iloc[split:].to_numpy()

    eps = 1e-9
    inv_a, inv_b = 1.0 / (rmse_a + eps), 1.0 / (rmse_b + eps)
    total = inv_a + inv_b
    w_a, w_b = inv_a / total, inv_b / total

    pred_a = bundle_a["estimator"].predict(x_val)
    pred_b = bundle_b["estimator"].predict(x_val)
    ensemble_pred = w_a * pred_a + w_b * pred_b

    ens_rmse = float(np.sqrt(np.mean((y_val - ensemble_pred) ** 2)))
    ens_mae = float(np.mean(np.abs(y_val - ensemble_pred)))
    denom = np.abs(y_val) + eps
    ens_mape = float(np.mean(np.abs((y_val - ensemble_pred) / denom)) * 100.0)

    ensemble_metrics.log_metric("ensemble_rmse", ens_rmse)
    ensemble_metrics.log_metric("ensemble_mae", ens_mae)
    ensemble_metrics.log_metric("ensemble_mape", ens_mape)
    ensemble_metrics.log_metric("weight_model_a", w_a)
    ensemble_metrics.log_metric("weight_model_b", w_b)

    final_bundle = {
        "type": "weighted_ensemble",
        "members": [
            {
                "model_name": bundle_a["model_name"],
                "weight": w_a,
                "estimator": bundle_a["estimator"],
            },
            {
                "model_name": bundle_b["model_name"],
                "weight": w_b,
                "estimator": bundle_b["estimator"],
            },
        ],
        "feature_cols": feature_cols,
        "metrics": {"rmse": ens_rmse, "mae": ens_mae, "mape": ens_mape},
    }

    payload = pickle.dumps(final_bundle)
    with open(ensemble_model.path, "wb") as fh:
        fh.write(payload)

    bucket_name = model_output_uri.replace("gs://", "").split("/")[0]
    blob_name = (
        "/".join(model_output_uri.replace("gs://", "").split("/")[1:])
        + "/ensemble_model.pkl"
    )
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(payload)

    return f"gs://{bucket_name}/{blob_name}"
