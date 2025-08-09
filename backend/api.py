import glob
import json
import os
import sys
import time
import uuid
import duckdb
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_path, "..")
sys.path.append(project_root)

from db.init_db import LOG_DB_PATH
from utils.sql_parser import extract_query_features

DB_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "..", "db")

app = FastAPI(title="SQL Performance Predictor API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str
    db_name: str
    model_name: str


class PredictionRequest(BaseModel):
    query: str
    db_name: str
    model_name: str


model = None
model_features = []
feature_scaler = None


def load_selected_model(model_name: str):
    global model, model_features, feature_scaler
    try:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "ml",
            model_name,
            "query_cost_predictor.joblib",
        )
        features_path = os.path.join(
            os.path.dirname(__file__), "..", "ml", model_name, "model_features.joblib"
        )
        scaler_path = os.path.join(
            os.path.dirname(__file__), "..", "ml", model_name, "feature_scaler.joblib"
        )
        model = joblib.load(model_path)
        model_features = joblib.load(features_path)
        feature_scaler = joblib.load(scaler_path)
        return True
    except:
        return False


def get_analytical_db_connection(db_name: str):
    full_db_path = os.path.join(DB_FOLDER_PATH, db_name)
    if not os.path.exists(full_db_path):
        raise RuntimeError(f"Database file not found: {db_name}")
    return duckdb.connect(database=full_db_path, read_only=False)


def get_query_logs_db_connection():
    if not os.path.exists(LOG_DB_PATH):
        raise RuntimeError("Query logs database not initialized.")
    return duckdb.connect(database=LOG_DB_PATH, read_only=False)


def get_optimization_suggestions(features: dict) -> list:
    suggestions = []
    if features.get("has_select_all"):
        suggestions.append("Avoid SELECT * — specify only needed columns.")
    if features.get("num_joins", 0) > 3:
        suggestions.append(
            "Query has many JOINs — consider reducing joins or indexing join keys."
        )
    elif features.get("num_joins", 0) > 0:
        suggestions.append("Ensure join keys are indexed for better performance.")
    if not features.get("has_where") and features.get("num_tables", 0) > 0:
        suggestions.append(
            "Add a WHERE clause to filter results and improve performance."
        )
    if not features.get("has_limit") and features.get("num_aggregations", 0) == 0:
        suggestions.append("Add a LIMIT clause to reduce result size.")
    if features.get("query_complexity_score", 0) > 15:
        suggestions.append(
            "Query is complex — consider breaking it into smaller parts or using CTEs."
        )
    if features.get("num_aggregations", 0) > 3:
        suggestions.append(
            "Multiple aggregations detected — consider pre-aggregating data."
        )
    if features.get("has_subquery"):
        suggestions.append(
            "Subquery detected — consider using CTEs for clarity and performance."
        )
    return suggestions


def create_advanced_features(features_dict):
    enhanced_features = features_dict.copy()
    enhanced_features["query_complexity_score"] = (
        enhanced_features.get("num_joins", 0) * 2.5
        + enhanced_features.get("num_aggregations", 0) * 2.0
        + enhanced_features.get("num_predicates", 0) * 0.8
        + enhanced_features.get("num_tables", 0) * 1.5
        + enhanced_features.get("has_subquery", False) * 3
        + enhanced_features.get("has_cte", False) * 2
    )
    enhanced_features["selectivity_estimate"] = (
        1.0 / (1 + enhanced_features.get("num_predicates", 0))
        if enhanced_features.get("has_where", False)
        else 1.0
    )
    enhanced_features["join_complexity"] = enhanced_features.get(
        "num_joins", 0
    ) * enhanced_features.get("num_tables", 0)
    enhanced_features["aggregation_complexity"] = (
        enhanced_features.get("num_aggregations", 0)
        * (1 + int(enhanced_features.get("has_groupby", False)))
        * (1 + enhanced_features.get("num_tables", 0))
    )
    enhanced_features["query_length_normalized"] = enhanced_features.get(
        "query_length", 0
    ) / max(enhanced_features.get("num_lines", 1), 1)
    enhanced_features["has_complex_operations"] = int(
        enhanced_features.get("has_union", False)
        or enhanced_features.get("has_intersect", False)
        or enhanced_features.get("has_except", False)
        or enhanced_features.get("has_cte", False)
    )
    enhanced_features["result_size_log"] = np.log1p(
        enhanced_features.get("result_rows", 0)
    )
    num_joins = enhanced_features.get("num_joins", 0)
    num_tables = enhanced_features.get("num_tables", 0)
    enhanced_features["tables_joins_ratio"] = (
        num_tables / num_joins if num_joins > 0 else num_tables
    )
    num_predicates = enhanced_features.get("num_predicates", 0)
    enhanced_features["predicates_per_table"] = (
        num_predicates / num_tables if num_tables > 0 else 0
    )
    return enhanced_features


def predict_query_cost(features_dict):
    if not model or not model_features or not feature_scaler:
        return -1.0
    try:
        enhanced_features = create_advanced_features(features_dict)
        current_features_df = pd.DataFrame([enhanced_features])
        missing_features = [
            col for col in model_features if col not in current_features_df.columns
        ]
        for feature in missing_features:
            current_features_df[feature] = 0
        current_features_df = current_features_df[model_features]
        boolean_features = [
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
        for col in boolean_features:
            if col in current_features_df.columns:
                current_features_df[col] = (
                    current_features_df[col].astype(bool).astype(int)
                )
        for col in current_features_df.columns:
            if not np.issubdtype(current_features_df[col].dtype, np.number):
                current_features_df[col] = pd.to_numeric(
                    current_features_df[col], errors="coerce"
                )
        current_features_df = current_features_df.fillna(0)
        current_features_df = current_features_df.replace([np.inf, -np.inf], 0)
        if hasattr(feature_scaler, "transform"):
            current_features_scaled = feature_scaler.transform(current_features_df)
        else:
            current_features_scaled = current_features_df.values
        prediction = model.predict(current_features_scaled)
        if len(prediction) > 0:
            estimated_cost = float(np.expm1(prediction[0]))
            return max(0.1, estimated_cost)
        return -1.0
    except:
        return -1.0


@app.get("/list_databases/")
async def list_databases():
    db_files = []
    for db_path in glob.glob(os.path.join(DB_FOLDER_PATH, "*.duckdb")):
        filename = os.path.basename(db_path)
        if filename != "query_logs.duckdb":
            db_files.append(filename)
    return {"databases": db_files}


@app.get("/list_models/")
async def list_models():
    models = []
    ml_folder = os.path.join(os.path.dirname(__file__), "..", "ml")
    for model_dir in os.listdir(ml_folder):
        if os.path.isdir(os.path.join(ml_folder, model_dir)):
            if (
                os.path.exists(
                    os.path.join(ml_folder, model_dir, "query_cost_predictor.joblib")
                )
                and os.path.exists(
                    os.path.join(ml_folder, model_dir, "model_features.joblib")
                )
                and os.path.exists(
                    os.path.join(ml_folder, model_dir, "feature_scaler.joblib")
                )
            ):
                models.append(model_dir)
    return {"models": models}


@app.post("/predict_cost/")
async def predict_cost(request: PredictionRequest):
    if not load_selected_model(request.model_name):
        raise HTTPException(status_code=400, detail="Failed to load selected ML model")
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        features = extract_query_features(query_text)
        features["result_rows"] = 0
        estimated_cost = predict_query_cost(features)
        optimization_suggestions = get_optimization_suggestions(features)
        return {
            "query": query_text,
            "estimated_cost": estimated_cost,
            "features": features,
            "suggestions": optimization_suggestions,
            "message": "Query prediction completed successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/execute_query/")
async def execute_query(request: QueryRequest):
    if not load_selected_model(request.model_name):
        raise HTTPException(status_code=400, detail="Failed to load selected ML model")
    query_text = request.query.strip()
    selected_db_name = request.db_name
    results_df = pd.DataFrame()
    execution_time_ms = 0.0
    result_rows = 0
    processed_results = []
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        start_time = time.perf_counter()
        conn = get_analytical_db_connection(db_name=selected_db_name)
        query_result = conn.execute(query_text)
        try:
            results_df = query_result.fetchdf()
            if not results_df.empty:
                results_df = results_df.fillna("")
                for col in results_df.columns:
                    if results_df[col].dtype == "object":
                        results_df[col] = results_df[col].astype(str)
                json_string = results_df.to_json(
                    orient="records", date_format="iso", indent=None
                )
                processed_results = json.loads(json_string)
                result_rows = len(processed_results)
            else:
                processed_results = []
                result_rows = 0
        except:
            processed_results = []
            results_df = pd.DataFrame()
            result_rows = 0
        execution_time_ms = (time.perf_counter() - start_time) * 1000
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error executing query: {str(e)}")
    finally:
        if "conn" in locals() and conn:
            conn.close()
    features = extract_query_features(query_text)
    features["result_rows"] = result_rows
    estimated_cost = predict_query_cost(features)
    try:
        conn_logs = get_query_logs_db_connection()
        log_id = str(uuid.uuid4())
        conn_logs.execute(
            f"""
             INSERT INTO query_logs (
                log_id, query_text, execution_time_ms, result_rows,
                has_select_all, has_where, has_join, has_groupby, query_length, num_tables, estimated_cost,
                num_lines, has_orderby, has_limit, has_distinct, has_union, has_intersect, has_except,
                has_cte, has_subquery, is_dml, is_ddl, num_joins, num_predicates, num_aggregations, num_unions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
            (
                log_id,
                query_text,
                execution_time_ms,
                result_rows,
                features.get("has_select_all", False),
                features.get("has_where", False),
                features.get("has_join", False),
                features.get("has_groupby", False),
                features.get("query_length", 0),
                features.get("num_tables", 0),
                estimated_cost,
                features.get("num_lines", 0),
                features.get("has_orderby", False),
                features.get("has_limit", False),
                features.get("has_distinct", False),
                features.get("has_union", False),
                features.get("has_intersect", False),
                features.get("has_except", False),
                features.get("has_cte", False),
                features.get("has_subquery", False),
                features.get("is_dml", False),
                features.get("is_ddl", False),
                features.get("num_joins", 0),
                features.get("num_predicates", 0),
                features.get("num_aggregations", 0),
                features.get("num_unions", 0),
            ),
        )
        conn_logs.close()
    except:
        pass
    optimization_suggestions = get_optimization_suggestions(features)
    return {
        "query": query_text,
        "results": processed_results,
        "execution_time_ms": execution_time_ms,
        "result_rows": result_rows,
        "features": features,
        "estimated_cost": estimated_cost,
        "message": "Query executed and logged successfully.",
        "suggestions": optimization_suggestions,
    }


@app.get("/get_logs/")
async def get_logs():
    try:
        conn_logs = get_query_logs_db_connection()
        logs_df = conn_logs.execute(
            "SELECT execution_time_ms, estimated_cost FROM query_logs WHERE execution_time_ms > 0 AND estimated_cost > 0;"
        ).fetchdf()
        conn_logs.close()
        logs_df = logs_df.where(pd.notna(logs_df), None)
        return logs_df.to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching logs: {e}")


@app.get("/get_feature_importance/")
async def get_feature_importance():
    try:
        if model and hasattr(model, "feature_importances_") and model_features:
            importance_dict = dict(zip(model_features, model.feature_importances_))
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            return {"feature_importance": sorted_importance}
        else:
            return {"feature_importance": {}}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting feature importance: {e}"
        )


@app.get("/")
async def root():
    return {
        "message": "SQL Query Performance Predictor API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": [
            "/execute_query/",
            "/predict_cost/",
            "/list_databases/",
            "/list_models/",
            "/get_logs/",
            "/get_feature_importance/",
        ],
    }
