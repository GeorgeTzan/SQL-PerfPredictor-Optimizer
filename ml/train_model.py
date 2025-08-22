import os
import sys
import warnings

warnings.filterwarnings("ignore")

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_path, "..")
sys.path.append(project_root)

from db.init_db import LOG_DB_PATH

MODEL_PATH = os.path.join(os.path.dirname(__file__), "query_cost_predictor.joblib")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "model_features.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "feature_scaler.joblib")


def load_data(db_path: str):
    conn = duckdb.connect(database=db_path, read_only=True)
    df = conn.execute("SELECT * FROM query_logs WHERE execution_time_ms > 0;").fetchdf()
    conn.close()
    return df


def create_advanced_features(df: pd.DataFrame):
    df_enhanced = df.copy()

    df_enhanced["query_complexity_score"] = (
        df_enhanced["num_joins"] * 2
        + df_enhanced["num_aggregations"] * 1.5
        + df_enhanced["num_predicates"] * 0.5
        + df_enhanced["num_tables"] * 1.2
        + df_enhanced["has_subquery"].astype(int) * 2
        + df_enhanced["has_cte"].astype(int) * 1.5
    )

    df_enhanced["selectivity_estimate"] = np.where(
        df_enhanced["has_where"], 1.0 / (1 + df_enhanced["num_predicates"]), 1.0
    )

    df_enhanced["join_complexity"] = (
        df_enhanced["num_joins"] * df_enhanced["num_tables"]
    )

    df_enhanced["aggregation_complexity"] = (
        df_enhanced["num_aggregations"]
        * (1 + df_enhanced["has_groupby"].astype(int))
        * (1 + df_enhanced["num_tables"])
    )

    df_enhanced["query_length_normalized"] = (
        df_enhanced["query_length"] / df_enhanced["num_lines"]
    )

    df_enhanced["has_complex_operations"] = (
        df_enhanced["has_union"]
        | df_enhanced["has_intersect"]
        | df_enhanced["has_except"]
        | df_enhanced["has_cte"]
    ).astype(int)

    df_enhanced["result_size_log"] = np.log1p(df_enhanced["result_rows"])

    df_enhanced["tables_joins_ratio"] = np.where(
        df_enhanced["num_joins"] > 0,
        df_enhanced["num_tables"] / df_enhanced["num_joins"],
        df_enhanced["num_tables"],
    )

    df_enhanced["predicates_per_table"] = np.where(
        df_enhanced["num_tables"] > 0,
        df_enhanced["num_predicates"] / df_enhanced["num_tables"],
        0,
    )

    return df_enhanced


def preprocess_data(df: pd.DataFrame):
    df_enhanced = create_advanced_features(df)

    df_filtered = df_enhanced[
        (df_enhanced["execution_time_ms"] > 0)
        & (
            df_enhanced["execution_time_ms"]
            < df_enhanced["execution_time_ms"].quantile(0.99)
        )
    ].copy()

    feature_columns = [
        "has_select_all",
        "has_where",
        "has_join",
        "has_groupby",
        "query_length",
        "num_tables",
        "num_lines",
        "has_orderby",
        "has_limit",
        "has_distinct",
        "has_union",
        "has_intersect",
        "has_except",
        "has_cte",
        "has_subquery",
        "is_dml",
        "is_ddl",
        "num_joins",
        "num_predicates",
        "num_aggregations",
        "num_unions",
        "result_rows",
        "query_complexity_score",
        "selectivity_estimate",
        "join_complexity",
        "aggregation_complexity",
        "query_length_normalized",
        "has_complex_operations",
        "result_size_log",
        "tables_joins_ratio",
        "predicates_per_table",
    ]

    available_features = [col for col in feature_columns if col in df_filtered.columns]
    X = df_filtered[available_features].copy()
    y = df_filtered["execution_time_ms"]

    boolean_feature_columns = [
        "has_select_all",
        "has_where",
        "has_join",
        "has_groupby",
        "has_orderby",
        "has_limit",
        "has_distinct",
        "has_union",
        "has_intersect",
        "has_except",
        "has_cte",
        "has_subquery",
        "is_dml",
        "is_ddl",
        "has_complex_operations",
    ]

    for col in boolean_feature_columns:
        if col in X.columns:
            X[col] = X[col].astype(bool).astype(int)

    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    y = np.log1p(y)

    return X, y


def train_multiple_models(X, y):
    if len(X) < 10:
        raise ValueError("Not enough samples to train models.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(score_func=f_regression, k=min(15, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42
        ),
        "Ridge": Ridge(alpha=1.0),
        "SVR": SVR(kernel="rbf", C=100, gamma="scale"),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }

    best_model = None
    best_score = float("inf")
    best_model_name = ""
    results = {}

    print("Training and evaluating multiple models...")

    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")

            if name in ["Ridge", "SVR", "MLP"]:
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            y_pred_original = np.expm1(y_pred)
            y_test_original = np.expm1(y_test)

            mae = mean_absolute_error(y_test_original, y_pred_original)
            mse = mean_squared_error(y_test_original, y_pred_original)
            r2 = r2_score(y_test_original, y_pred_original)

            results[name] = {
                "model": model,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "uses_scaling": name in ["Ridge", "SVR", "MLP"],
            }

            print(f"{name} Results:")
            print(f"  MAE: {mae:.2f} ms")
            print(f"  MSE: {mse:.2f}")
            print(f"  R²: {r2:.4f}")

            if mae < best_score:
                best_score = mae
                best_model = model
                best_model_name = name

        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    print(f"\nBest model: {best_model_name} with MAE: {best_score:.2f} ms")

    return best_model, scaler, selector, results, best_model_name


def hyperparameter_tuning(X, y, model_type="RandomForest"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "RandomForest":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        model = RandomForestRegressor(random_state=42, n_jobs=-1)

    elif model_type == "GradientBoosting":
        param_grid = {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.05, 0.1, 0.15],
            "max_depth": [6, 8, 10],
        }
        model = GradientBoostingRegressor(random_state=42)

    else:
        return None

    print(f"Performing hyperparameter tuning for {model_type}...")
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    y_pred_original = np.expm1(y_pred)
    y_test_original = np.expm1(y_test)

    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Tuned model MAE: {mae:.2f} ms")
    print(f"Tuned model R²: {r2:.4f}")

    return best_model


def create_performance_plots(results, save_path=None):
    if not results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Model Performance Comparison", fontsize=16)

    model_names = list(results.keys())
    maes = [results[name]["mae"] for name in model_names]
    r2s = [results[name]["r2"] for name in model_names]
    mses = [results[name]["mse"] for name in model_names]

    axes[0, 0].bar(model_names, maes, color="skyblue")
    axes[0, 0].set_title("Mean Absolute Error (MAE)")
    axes[0, 0].set_ylabel("MAE (ms)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(model_names, r2s, color="lightgreen")
    axes[0, 1].set_title("R² Score")
    axes[0, 1].set_ylabel("R²")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].bar(model_names, mses, color="salmon")
    axes[1, 0].set_title("Mean Squared Error (MSE)")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].tick_params(axis="x", rotation=45)

    performance_df = pd.DataFrame(
        {"Model": model_names, "MAE": maes, "R²": r2s, "MSE": mses}
    )

    axes[1, 1].axis("tight")
    axes[1, 1].axis("off")
    table = axes[1, 1].table(
        cellText=performance_df.round(3).values,
        colLabels=performance_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Performance plots saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print(f"Loading data from: {LOG_DB_PATH}")
    data_df = load_data(LOG_DB_PATH)

    if data_df.empty:
        print("No data found in query_logs.db. Please run query generation first.")
    else:
        print(f"Loaded {len(data_df)} records.")
        try:
            X, y = preprocess_data(data_df)

            if X.empty or len(X) < 10:
                print("Not enough valid data for training.")
            else:
                print(f"Using {len(X)} records for training after preprocessing.")

                best_model, scaler, selector, results, best_model_name = (
                    train_multiple_models(X, y)
                )

                if best_model is not None:
                    joblib.dump(best_model, MODEL_PATH)
                    joblib.dump(X.columns.tolist(), FEATURES_PATH)
                    joblib.dump(scaler, SCALER_PATH)

                    print(f"\nBest model ({best_model_name}) saved to: {MODEL_PATH}")
                    print(f"Feature columns saved to: {FEATURES_PATH}")
                    print(f"Scaler saved to: {SCALER_PATH}")

                    plots_path = os.path.join(
                        os.path.dirname(__file__), "model_performance.png"
                    )
                    create_performance_plots(results, plots_path)

                    if len(X) > 100:
                        print("\nPerforming hyperparameter tuning...")
                        tuned_model = hyperparameter_tuning(X, y, best_model_name)
                        if tuned_model:
                            tuned_model_path = MODEL_PATH.replace(
                                ".joblib", "_tuned.joblib"
                            )
                            joblib.dump(tuned_model, tuned_model_path)
                            print(f"Tuned model saved to: {tuned_model_path}")

        except Exception as e:
            print(f"Error during training: {e}")
            import traceback

            traceback.print_exc()
