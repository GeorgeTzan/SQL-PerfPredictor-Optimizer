import random

import dash_bootstrap_components as dbc
import pandas as pd
import requests
from dash import Dash, Input, Output, State, dash_table, dcc, html

THEME = dbc.themes.DARKLY
app = Dash(
    title="SQL-QPPO",
    external_stylesheets=[THEME, dbc.themes.BOOTSTRAP],
)

app.layout = dbc.Container(
    [
        html.H1(
            "SQL Query Performance Predictor & Optimizer", className="text-primary my-4"
        ),
        dbc.Card(
            [
                dbc.CardHeader(html.H2("Select DuckDB Database")),
                dbc.CardBody(
                    [
                        dcc.Dropdown(
                            id="db-selector",
                            options=[],
                            placeholder="Select a database (.duckdb file)",
                            clearable=False,
                            style={"width": "100%"},
                            persistence=True,
                            persistence_type="local",
                        ),
                        html.Div(id="db-selection-status", className="text-info"),
                        html.P(
                            "Select the DuckDB database to run queries against.",
                            className="text-muted mt-2",
                        ),
                    ]
                ),
            ],
            className="mb-4",
        ),
        dbc.Card(
            [
                dbc.CardHeader(html.H2("Execute SQL Query")),
                dbc.CardBody(
                    [
                        dcc.Textarea(
                            id="sql-input",
                            value="SELECT * FROM sales LIMIT 10;",
                            style={
                                "width": "100%",
                                "height": 150,
                                "backgroundColor": "#343a40",
                                "color": "white",
                                "fontFamily": "monospace",
                                "border": "1px solid #495057",
                            },
                        ),
                        dbc.Button(
                            "Execute Query",
                            id="execute-button",
                            n_clicks=0,
                            color="primary",
                            className="mt-3",
                        ),
                    ]
                ),
            ],
            className="mb-4",
        ),
        dbc.Card(
            [
                dbc.CardHeader(html.H3("Query Execution Results")),
                dbc.CardBody(
                    [
                        html.Div(id="output-message", className="alert alert-success"),
                        html.Div(id="error-message", className="alert alert-danger"),
                        html.P(id="execution-time"),
                        html.P(id="result-rows"),
                        html.P(id="estimated-cost-display"),
                        html.Details(
                            [
                                html.Summary("Extracted Features"),
                                html.Div(id="extracted-features"),
                            ]
                        ),
                        html.H4("Query Results Table", className="mt-4"),
                        dash_table.DataTable(
                            id="query-results-table",
                            columns=[{"name": i, "id": i} for i in []],
                            data=[],
                            style_table={
                                "overflowX": "auto",
                                "backgroundColor": "#343a40",
                                "color": "white",
                                "border": "1px solid #495057",
                            },
                            style_cell={
                                "textAlign": "left",
                                "backgroundColor": "#343a40",
                                "color": "white",
                                "border": "1px solid #495057",
                            },
                            style_header={
                                "backgroundColor": "#212529",
                                "color": "white",
                                "fontWeight": "bold",
                            },
                            page_size=10,
                        ),
                    ]
                ),
            ],
            className="mb-4",
        ),
        dbc.Card(
            [
                dbc.CardHeader(html.H3("Optimization Suggestions (Coming Soon!)")),
                dbc.CardBody(
                    [
                        html.Div(
                            "This section will provide intelligent SQL optimization recommendations."
                        )
                    ]
                ),
            ],
            className="mb-4",
        ),
    ],
    fluid=False,
    className="p-8",
)


@app.callback(
    Output("db-selector", "options"),
    Output("db-selector", "value"),
    Output("db-selection-status", "children"),
    Input("db-selector", "id"),
)
def load_databases(id):
    try:
        response = requests.get("http://127.0.0.1:8000/list_databases/")
        response.raise_for_status()
        data = response.json()
        db_files = data.get("databases", [])

        options = [{"label": db, "value": db} for db in db_files]

        default_value = (
            "analytical_db.duckdb"
            if "analytical_db.duckdb" in db_files
            else (db_files[0] if db_files else None)
        )

        return options, default_value, f"Found {len(db_files)} database(s)."
    except requests.exceptions.RequestException as e:
        return [], None, f"Error fetching databases: {e}"


@app.callback(
    Output("output-message", "children"),
    Output("error-message", "children"),
    Output("execution-time", "children"),
    Output("result-rows", "children"),
    Output("estimated-cost-display", "children"),
    Output("extracted-features", "children"),
    Output("query-results-table", "columns"),
    Output("query-results-table", "data"),
    Input("execute-button", "n_clicks"),
    State("sql-input", "value"),
    State("db-selector", "value"),
)
def handle_query_execution(n_clicks, query_text, selected_db):
    if n_clicks == 0:
        return "", "", "", "", "", "", [], []

    if not selected_db:
        return "", "Please select a database first.", "", "", "", "", [], []

    backend_url = "http://127.0.0.1:8000/execute_query/"  # development URL
    try:
        response = requests.post(
            backend_url, json={"query": query_text, "db_name": selected_db}
        )
        response.raise_for_status()
        data = response.json()

        data_dict = data.get("features", {})
        for k, v in data_dict.items():
            if isinstance(v, bool):
                data_dict[k] = 1 if v else 0
            elif isinstance(v, (int, float)):
                data_dict[k] = f"{v:.2f}" if isinstance(v, float) else str(v)
            else:
                data_dict[k] = str(v)

        tick_choice = random.randint(0, 1)

        df = pd.DataFrame(data["results"])
        columns = [{"name": i, "id": i} for i in df.columns]
        table_data = df.to_dict("records")
        return (
            data.get("message", "Query executed."),
            "",
            f"Execution Time: {data.get('execution_time_ms', 0):.2f} ms",
            f"Result Rows: {data.get('result_rows', 0)}",
            f"Estimated Cost (ML): {data.get('estimated_cost', -1.0):.2f}",
            (
                html.Pre(str(list(data_dict.values())))
                if tick_choice == 1
                else html.Pre(str(data_dict))
            ),
            columns,
            table_data,
        )
    except requests.exceptions.RequestException as e:
        return "", str(e), "", "", "", "", [], []


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
