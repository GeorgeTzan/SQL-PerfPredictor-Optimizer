from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import duckdb
import os
import time
import uuid
import pandas as pd

from utils.sql_parser import extract_query_features
from db.init_db import DB_PATH, LOG_DB_PATH


app = FastAPI()

class QueryRequest(BaseModel):
    query: str

def get_analytical_db_connection():
    if not os.path.exists(DB_PATH):
        raise RuntimeError("Analytical database not initialized.")
    return duckdb.connect(database=DB_PATH, read_only=False)

def get_query_logs_db_connection():
    if not os.path.exists(LOG_DB_PATH):
        raise RuntimeError("Query logs database not initialized.")
    return duckdb.connect(database=LOG_DB_PATH, read_only=False)

@app.post("/execute_query/")
async def execute_query(request: QueryRequest):
    query_text = request.query.strip()
    results_df = pd.DataFrame()
    execution_time_ms = 0.0
    result_rows = 0
    error_message = None

    try:
        start_time = time.perf_counter()
        conn = get_analytical_db_connection()
        
        query_result = conn.execute(query_text)
        
        try:
            results_df = query_result.fetchdf()
            result_rows = len(results_df) if results_df is not None else 0
        except duckdb.InvalidInputException:
            results_df = pd.DataFrame()
            result_rows = 0
            
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
    except Exception as e:
        error_message = str(e)
        raise HTTPException(status_code=400, detail=f"Error executing query: {error_message}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    features = extract_query_features(query_text)

    estimated_cost = -1.0

    try:
        conn_logs = get_query_logs_db_connection()
        log_id = str(uuid.uuid4())
        conn_logs.execute(f"""
            INSERT INTO query_logs (
                log_id, query_text, execution_time_ms, result_rows,
                has_select_all, has_where, has_join, has_groupby, query_length, num_tables, estimated_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
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
            estimated_cost
        ))
        conn_logs.close()
        print(f"Logged query: {query_text[:50]}... Log ID: {log_id}")

    except Exception as log_error:
        print(f"Error logging query: {log_error}")

    return {
        "query": query_text,
        "results": results_df.to_dict('records') if results_df is not None else [],
        "execution_time_ms": execution_time_ms,
        "result_rows": result_rows,
        "features": features,
        "estimated_cost": estimated_cost,
        "message": "Query executed and logged successfully."
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the SQL Performance API!"}