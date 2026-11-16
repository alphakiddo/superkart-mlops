"""Step 3 - Model building: tune Random Forest and Gradient Boosting, track with MLflow, save best."""
import json
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ARTIFACT_DIR, MODEL_DIR, TARGET = "artifacts", "models", "Product_Store_Sales_Total"
CATEGORICAL = ["Product_Sugar_Content", "Store_Size", "Store_Location_City_Type",
               "Store_Type", "Product_Id_char", "Product_Type_Category"]

CANDIDATES = {
    "Random Forest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"model__n_estimators": [100, 200, 300],
         "model__max_depth": [6, 8, 10, None],
         "model__min_samples_split": [2, 5, 10],
         "model__min_samples_leaf": [1, 2, 4],
         "model__max_features": ["sqrt", None]},
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {"model__n_estimators": [100, 200, 300],
         "model__learning_rate": [0.03, 0.05, 0.1],
         "model__max_depth": [2, 3, 4],
         "model__min_samples_leaf": [1, 2, 4],
         "model__subsample": [0.7, 0.85, 1.0]},
    ),
}


def metrics(model, X, y, prefix):
    p = model.predict(X)
    return {f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y, p))),
            f"{prefix}_mae": float(mean_absolute_error(y, p)),
            f"{prefix}_r2": float(r2_score(y, p))}


def main() -> None:
    train = pd.read_csv(f"{ARTIFACT_DIR}/train.csv")
    test = pd.read_csv(f"{ARTIFACT_DIR}/test.csv")
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("SuperKart_Sales_Forecast")

    summary, best_name, best_model, best_rmse = {}, None, None, float("inf")
    for name, (estimator, grid) in CANDIDATES.items():
        pre = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL)],
            remainder="passthrough")
        pipe = Pipeline([("prep", pre), ("model", estimator)])
        search = RandomizedSearchCV(pipe, grid, n_iter=12, cv=3, random_state=42,
                                    scoring="neg_root_mean_squared_error", n_jobs=-1)
        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            # log every tuned combination as a nested run
            res = search.cv_results_
            for i, params in enumerate(res["params"]):
                with mlflow.start_run(run_name=f"{name}_trial_{i}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_rmse", float(-res["mean_test_score"][i]))
            best = search.best_estimator_
            m = {**metrics(best, X_train, y_train, "train"), **metrics(best, X_test, y_test, "test")}
            mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
            mlflow.log_metrics(m)
        summary[name] = {"best_params": search.best_params_, **m}
        print(f"{name}: best params={search.best_params_}")
        print(f"{name}: test RMSE={m['test_rmse']:.1f}  MAE={m['test_mae']:.1f}  R2={m['test_r2']:.4f}")
        if m["test_rmse"] < best_rmse:
            best_name, best_model, best_rmse = name, best, m["test_rmse"]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.joblib", compress=3)
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": summary}, f, indent=2)
    print(f"BEST MODEL: {best_name} (test RMSE {best_rmse:.1f}) saved to {MODEL_DIR}/best_model.joblib")


if __name__ == "__main__":
    main()
