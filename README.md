# SQL-PerfPredictor-Optimizer

**🚧 WORK IN PROGRESS 🚧**

This project is currently under active development as part of a potentional Bachelor's Thesis. Features and functionality are subject to change, and the project is not yet in a stable state.

## Overview

This project focuses on developing an intelligent web tool designed to analyze and optimize SQL query performance on analytical DuckDB databases. The system takes user-submitted SQL queries, executes them, and collects key performance metrics (execution time, result size) along with query-specific features.

Utilizing this collected data, a Machine Learning model will be trained to predict the "cost" (execution time) of new queries before they are even executed. Furthermore, the tool aims to provide automated optimization suggestions to the user, leveraging either rule-based heuristics or more advanced AI techniques.

## Key Features (Planned/Implemented)

*   **Query Execution & Logging:** Execute SQL queries on DuckDB and log performance metrics and query features. (Implemented)
*   **ML-driven Performance Prediction:** Predict query execution time using trained Machine Learning models (e.g., Random Forest, LightGBM). (Future)
*   **Automated Optimization Suggestions:** Provide recommendations for query optimization. (Future)
*   **Interactive Web Interface:** A user-friendly web application for input, analysis, and visualization. (Initial UI implemented)
*   **Visualizations:** Compare predicted vs. actual execution times, analyze query characteristics, and more. (Future)

## Technologies Used

*   **Backend:** Python (FastAPI)
*   **Frontend/UI:** Python (Dash by Plotly)
*   **Database:** DuckDB (for analytics queries and logs)
*   **Machine Learning:** scikit-learn, LightGBM (Future)
*   **SQL Parsing:** sqlglot

## Setup & Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GeorgeTzan/SQL-PerfPredictor-Optimizer.git
    cd SQL-PerfPredictor-Optimizer
    ```
2.  **Create and activate a virtual environment using `uv`:**
    ```bash
    uv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
    *(If `requirements.txt` is missing or outdated, you can use `uv sync` instead)*

4.  **Initialize the DuckDB databases:**
    ```bash
    python db/init_db.py
    ```
    This will create `db/analytical_db.duckdb` and `db/query_logs.duckdb`.

## How to Run

Ensure your virtual environment is activated (`source venv/bin/activate`).

1.  **Start the FastAPI backend (in one terminal):**
    ```bash
    uvicorn backend.api:app --reload
    ```
    The backend will be available at `http://127.0.0.1:8000/`.

2.  **Start the Dash frontend (in a separate terminal):**
    ```bash
    python frontend/app.py
    ```
    The Dash application will be available at `http://127.0.0.1:8050/`.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.