import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import time
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  LOAD & CLEAN DATA
# ─────────────────────────────────────────────
df = pd.read_csv("phone_addiction_dataset.csv")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.drop_duplicates(inplace=True)
df.drop(columns=["name", "location"], inplace=True, errors="ignore")

# ─────────────────────────────────────────────
# 2.  OUTLIER CAPPING  (IQR – cap, not drop)
# ─────────────────────────────────────────────
num_cols_raw = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if "addiction_level" in num_cols_raw:
    num_cols_raw.remove("addiction_level")

df_cap = df.copy()
for col in num_cols_raw:
    Q1 = df_cap[col].quantile(0.25)
    Q3 = df_cap[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_cap[col] = df_cap[col].clip(lower, upper)

# ─────────────────────────────────────────────
# 3.  FEATURES / TARGET / SPLIT
# ─────────────────────────────────────────────
X = df_cap.drop("addiction_level", axis=1)
y = df_cap["addiction_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 4.  PREPROCESSOR
# ─────────────────────────────────────────────
categorical_cols = ["gender", "phone_usage_purpose"]
numerical_cols   = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

cat_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop=None, handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough"
)

def make_preprocessor(scaler):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop=None, handle_unknown="ignore"), categorical_cols),
            ("num", scaler, numerical_cols),
        ]
    )

def make_pipeline(scaler, model):
    return Pipeline([
        ("preprocessor", make_preprocessor(scaler)),
        ("model", model),
    ])

def get_scaler(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    return StandardScaler() if scaler_type == "standard" else MinMaxScaler()

def cv_r2(pipeline, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X, y, scoring="r2", cv=kf).mean()

# ─────────────────────────────────────────────
# 5.  OPTUNA OBJECTIVES
# ─────────────────────────────────────────────

def objective_lr(trial):
    scaler = get_scaler(trial)
    model = Ridge(alpha=trial.suggest_float("alpha", 1e-4, 10.0, log=True))
    return cv_r2(make_pipeline(scaler, model), X_train, y_train)

def objective_dt(trial):
    scaler = get_scaler(trial)
    model = DecisionTreeRegressor(
        max_depth         = trial.suggest_int("max_depth", 3, 20),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state      = 42,
    )
    return cv_r2(make_pipeline(scaler, model), X_train, y_train)

def objective_rf(trial):
    scaler = get_scaler(trial)
    model = RandomForestRegressor(
        n_estimators      = trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth         = trial.suggest_int("max_depth", 5, 30),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        bootstrap         = trial.suggest_categorical("bootstrap", [True, False]),
        random_state      = 42,
        n_jobs            = -1,
    )
    return cv_r2(make_pipeline(scaler, model), X_train, y_train)

def objective_gb(trial):
    scaler = get_scaler(trial)
    model = GradientBoostingRegressor(
        n_estimators      = trial.suggest_int("n_estimators", 100, 500, step=50),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth         = trial.suggest_int("max_depth", 2, 10),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        random_state      = 42,
    )
    return cv_r2(make_pipeline(scaler, model), X_train, y_train)

def objective_knn(trial):
    scaler = get_scaler(trial)
    model = KNeighborsRegressor(
        n_neighbors = trial.suggest_int("n_neighbors", 3, 21, step=2),
        weights     = trial.suggest_categorical("weights", ["uniform", "distance"]),
        metric      = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
    )
    return cv_r2(make_pipeline(scaler, model), X_train, y_train)

def objective_svr(trial):
    scaler = get_scaler(trial)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    params = {
        "kernel":  kernel,
        "C":       trial.suggest_float("C", 1e-2, 1e2, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.01, 0.5),
    }
    if kernel in ["rbf", "poly", "sigmoid"]:
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    model = SVR(**params)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    return cross_val_score(make_pipeline(scaler, model), X_train, y_train,
                           scoring="r2", cv=kf).mean()

# ─────────────────────────────────────────────
# 6.  REBUILD BEST PIPELINE
# ─────────────────────────────────────────────

def rebuild_best_pipeline(model_name, best_params):
    scaler = StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler()

    if model_name == "LinearRegression":
        model = Ridge(alpha=best_params["alpha"])
    elif model_name == "DecisionTree":
        model = DecisionTreeRegressor(
            max_depth         = best_params["max_depth"],
            min_samples_split = best_params["min_samples_split"],
            min_samples_leaf  = best_params["min_samples_leaf"],
            max_features      = best_params["max_features"],
            random_state      = 42,
        )
    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators      = best_params["n_estimators"],
            max_depth         = best_params["max_depth"],
            min_samples_split = best_params["min_samples_split"],
            min_samples_leaf  = best_params["min_samples_leaf"],
            max_features      = best_params["max_features"],
            bootstrap         = best_params["bootstrap"],
            random_state      = 42,
            n_jobs            = -1,
        )
    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor(
            n_estimators      = best_params["n_estimators"],
            learning_rate     = best_params["learning_rate"],
            max_depth         = best_params["max_depth"],
            min_samples_split = best_params["min_samples_split"],
            min_samples_leaf  = best_params["min_samples_leaf"],
            max_features      = best_params["max_features"],
            subsample         = best_params["subsample"],
            random_state      = 42,
        )
    elif model_name == "KNN":
        model = KNeighborsRegressor(
            n_neighbors = best_params["n_neighbors"],
            weights     = best_params["weights"],
            metric      = best_params["metric"],
        )
    elif model_name == "SVR":
        params = {
            "kernel":  best_params["kernel"],
            "C":       best_params["C"],
            "epsilon": best_params["epsilon"],
        }
        if best_params["kernel"] in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = best_params["gamma"]
        if best_params["kernel"] == "poly":
            params["degree"] = best_params["degree"]
        model = SVR(**params)

    return make_pipeline(scaler, model)

# ─────────────────────────────────────────────
# 7.  MODEL ID & SCALER ID MAPPINGS  ← ADDED
# ─────────────────────────────────────────────
objectives = {
    "LinearRegression": objective_lr,
    "DecisionTree":     objective_dt,
    "RandomForest":     objective_rf,
    "GradientBoosting": objective_gb,
    "KNN":              objective_knn,
    "SVR":              objective_svr,
}

# Map model names and scaler types to integer IDs for MLflow logging
model_dict  = {name: idx for idx, name in enumerate(objectives.keys())}
scaler_dict = {"standard": 0, "minmax": 1}

# model_dict  = { "LinearRegression": 0, "DecisionTree": 1,
#                 "RandomForest": 2,     "GradientBoosting": 3,
#                 "KNN": 4,              "SVR": 5 }
# scaler_dict = { "standard": 0, "minmax": 1 }

N_TRIALS = 20

mlflow.set_experiment("MobileAddiction_HPT_Runs")

results = {}

# ─────────────────────────────────────────────
# 8.  MAIN LOOP
# ─────────────────────────────────────────────
for model_name, obj_fn in objectives.items():
    print(f"\n{'='*55}")
    print(f"  Optimizing: {model_name}")
    print(f"{'='*55}")

    mlflow_cb = MLflowCallback(
        tracking_uri=None,
        metric_name="cv_r2",
        mlflow_kwargs={"nested": True},
    )

    study = optuna.create_study(direction="maximize")

    start_fit = time.time()
    study.optimize(obj_fn, n_trials=N_TRIALS, callbacks=[mlflow_cb])
    fit_time = time.time() - start_fit

    best_params = study.best_params
    best_cv_r2  = study.best_value
    print(f"  Best CV R²  : {best_cv_r2:.4f}")
    print(f"  Best Params : {best_params}")

    # Rebuild & fit on full training set
    best_pipeline = rebuild_best_pipeline(model_name, best_params)
    best_pipeline.fit(X_train, y_train)

    # Evaluate
    start_test   = time.time()
    y_test_pred  = best_pipeline.predict(X_test)
    test_time    = time.time() - start_test
    y_train_pred = best_pipeline.predict(X_train)

    train_r2   = round(r2_score(y_train, y_train_pred),          4)
    test_r2    = round(r2_score(y_test,  y_test_pred),           4)
    train_mae  = round(mean_absolute_error(y_train, y_train_pred), 4)
    test_mae   = round(mean_absolute_error(y_test,  y_test_pred),  4)
    train_rmse = round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4)
    test_rmse  = round(np.sqrt(mean_squared_error(y_test,  y_test_pred)),  4)

    diff   = round(train_r2 - test_r2, 4)
    status = ("Good Fit"         if diff < 0.05
              else "Mild Overfitting" if diff < 0.15
              else "Overfitting")

    print(f"\n  Train R²={train_r2}  Test R²={test_r2}  Δ={diff}  [{status}]")
    print(f"  Test MAE={test_mae}  Test RMSE={test_rmse}")
    print(f"  Fit Time: {fit_time:.1f}s  |  Test Time: {test_time:.4f}s")

    # Save model & log to MLflow
    model_path = f"{model_name}_addiction_model.pkl"
    joblib.dump(best_pipeline, model_path)
    model_size = os.path.getsize(model_path)

    # ── MLflow logging ────────────────────────────────────────
    mlflow.log_metric("model_id",   model_dict[model_name])               # ← integer ID
    mlflow.log_metric("scaler_id",  scaler_dict[best_params["scaler_type"]])  # ← integer ID
    mlflow.log_metric("cv_r2",      best_cv_r2)
    mlflow.log_metric("train_r2",   train_r2)
    mlflow.log_metric("test_r2",    test_r2)
    mlflow.log_metric("train_mae",  train_mae)
    mlflow.log_metric("test_mae",   test_mae)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse",  test_rmse)
    mlflow.log_metric("r2_diff",    diff)
    mlflow.log_metric("train_time", fit_time)
    mlflow.log_metric("test_time",  test_time)
    mlflow.log_metric("model_size", model_size)
    mlflow.log_param("fit_status",  status)
    mlflow.sklearn.log_model(best_pipeline, name=f"{model_name}_addiction_model")
    # os.remove(model_path)   ← commented out so .pkl files are kept

    results[model_name] = {
        "best_params":  best_params,
        "cv_r2":        best_cv_r2,
        "train_r2":     train_r2,
        "test_r2":      test_r2,
        "train_mae":    train_mae,
        "test_mae":     test_mae,
        "train_rmse":   train_rmse,
        "test_rmse":    test_rmse,
        "r2_diff":      diff,
        "status":       status,
        "fit_time":     fit_time,
        "test_time":    test_time,
        "model_size":   model_size,
    }

    mlflow.end_run()

# ─────────────────────────────────────────────
# 9.  SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n\n" + "="*80)
print("  FINAL SUMMARY")
print("="*80)
summary = pd.DataFrame(results).T[
    ["cv_r2", "train_r2", "test_r2", "r2_diff", "test_mae", "test_rmse", "status", "fit_time"]
]
summary.index.name = "Model"
print(summary.to_string())

best_model_name = summary["test_r2"].idxmax()
print(f"\n  Best model by Test R²: {best_model_name}  "
      f"(Test R² = {summary.loc[best_model_name, 'test_r2']:.4f})")

print("\n  Model ID Reference:")
for name, idx in model_dict.items():
    print(f"    {idx} = {name}")

print("\n  Scaler ID Reference:")
for name, idx in scaler_dict.items():
    print(f"    {idx} = {name}")