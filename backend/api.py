import glob
import json
import os
import time
import uuid

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from db.init_db import DB_PATH, LOG_DB_PATH
from utils.sql_parser import extract_query_features

DB_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "..", "db")

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    db_name: str = "analytical_db.duckdb"


def get_analytical_db_connection(db_name: str = "analytical_db.duckdb"):
    """Returns a connection to the analytical DuckDB database based on selected name."""
    full_db_path = os.path.join(DB_FOLDER_PATH, db_name)
    if not os.path.exists(full_db_path):
        raise RuntimeError(f"Database file not found: {db_name}")
    return duckdb.connect(database=full_db_path, read_only=False)


def get_query_logs_db_connection():
    if not os.path.exists(LOG_DB_PATH):
        raise RuntimeError("Query logs database not initialized.")
    return duckdb.connect(database=LOG_DB_PATH, read_only=False)


@app.post("/execute_query/")
async def execute_query(request: QueryRequest):
    query_text = request.query.strip()
    selected_db_name = request.db_name
    results_df = pd.DataFrame()
    execution_time_ms = 0.0
    result_rows = 0
    error_message = None

    try:
        start_time = time.perf_counter()
        conn = get_analytical_db_connection(db_name=selected_db_name)

        query_result = conn.execute(query_text)

        try:
            results_df = query_result.fetchdf()

            if not results_df.empty:
                json_string = results_df.to_json(
                    orient="records", date_format="iso", indent=None
                )
                processed_results = json.loads(json_string)
                result_rows = len(processed_results)
            else:
                processed_results = []
                result_rows = 0

        except duckdb.InvalidInputException:
            processed_results = []
            results_df = pd.DataFrame()
            result_rows = 0
        execution_time_ms = (time.perf_counter() - start_time) * 1000

    except Exception as e:
        error_message = str(e)
        raise HTTPException(
            status_code=400, detail=f"Error executing query: {error_message}"
        )
    finally:
        if "conn" in locals() and conn:
            conn.close()

    features = extract_query_features(query_text)

    estimated_cost = -1.0

    try:
        conn_logs = get_query_logs_db_connection()
        log_id = str(uuid.uuid4())
        conn_logs.execute(
            f"""
            INSERT INTO query_logs (
                log_id, query_text, execution_time_ms, result_rows,
                has_select_all, has_where, has_join, has_groupby, query_length, num_tables, estimated_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            ),
        )
        conn_logs.close()
        print(f"Logged query: {query_text[:50]}... Log ID: {log_id}")

    except Exception as log_error:
        print(f"Error logging query: {log_error}")

    return {
        "query": query_text,
        "results": processed_results,
        "execution_time_ms": execution_time_ms,
        "result_rows": result_rows,
        "features": features,
        "estimated_cost": estimated_cost,
        "message": "Query executed and logged successfully.",
    }


@app.get("/list_databases/")
async def list_databases():
    db_files = []
    for db_path in glob.glob(os.path.join(DB_FOLDER_PATH, "*.duckdb")):
        filename = os.path.basename(db_path)
        if filename != "query_logs.duckdb":
            db_files.append(filename)
    return {"databases": db_files}


@app.get("/")
async def root():
    return {"message": "Welcome to the SQL Performance API!"}
