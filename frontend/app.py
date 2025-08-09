import random
import sys
import os
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dash_table, dcc, html, callback_context
import dash

THEME = dbc.themes.DARKLY
app = Dash(
    __name__,
    external_stylesheets=[THEME],
    title="SQL-QPPO - Universal DB Analytics",
)

app.layout = dbc.Container(
    [
        dcc.Store(id="prediction-store"),
        dcc.Store(id="execution-results-store"),
        dcc.Interval(
            id="prediction-interval", interval=1000, n_intervals=0, disabled=True
        ),
        html.H1(
            "SQL Query Performance Predictor & Optimizer",
            className="text-primary my-4 text-center",
        ),
        dbc.Alert(
            [
                "Select any DuckDB database and trained ML model to predict and optimize SQL queries."
            ],
            color="info",
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H3(
                                            "Database & Model Selection",
                                            className="mb-0",
                                        ),
                                        dbc.Badge(
                                            "Connected",
                                            color="success",
                                            id="db-status-badge",
                                            className="ms-2",
                                        ),
                                    ]
                                ),
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
                                        html.Div(
                                            id="db-selection-status",
                                            className="text-info mt-2",
                                        ),
                                        dcc.Dropdown(
                                            id="model-selector",
                                            options=[],
                                            placeholder="Select a trained ML model",
                                            clearable=False,
                                            style={"width": "100%"},
                                            persistence=True,
                                            persistence_type="local",
                                        ),
                                        html.Div(
                                            id="model-selection-status",
                                            className="text-info mt-2",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H3(
                                            "Real-time Query Analysis", className="mb-0"
                                        ),
                                        dbc.Badge(
                                            "Ready",
                                            color="info",
                                            id="analysis-status",
                                            className="ms-2",
                                        ),
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        html.Label(
                                            "Estimated Cost:",
                                            className="fw-bold text-warning",
                                        ),
                                        html.H4(
                                            id="realtime-cost",
                                            children="Type a query...",
                                            className="text-warning mb-2",
                                        ),
                                        html.Label(
                                            "Query Complexity:",
                                            className="fw-bold text-info",
                                        ),
                                        html.P(
                                            id="complexity-score",
                                            children="N/A",
                                            className="text-info mb-2",
                                        ),
                                        html.Label(
                                            "Optimization Level:", className="fw-bold"
                                        ),
                                        dbc.Progress(
                                            id="optimization-progress",
                                            value=0,
                                            className="mb-2",
                                        ),
                                        html.Small(
                                            "Higher is better", className="text-muted"
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H3("SQL Query Editor", className="mb-0"),
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(
                                                    "Execute",
                                                    id="execute-button",
                                                    color="primary",
                                                    size="sm",
                                                ),
                                                dbc.Button(
                                                    "Clear",
                                                    id="clear-button",
                                                    color="secondary",
                                                    size="sm",
                                                ),
                                            ],
                                            className="ms-auto",
                                        ),
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Textarea(
                                            id="sql-input",
                                            value="SELECT * FROM my_table LIMIT 10;",
                                            style={
                                                "width": "100%",
                                                "height": 200,
                                                "backgroundColor": "#2b3035",
                                                "color": "#f8f9fa",
                                                "fontFamily": "Monaco, Consolas, monospace",
                                                "fontSize": "14px",
                                                "border": "2px solid #495057",
                                                "borderRadius": "8px",
                                                "padding": "12px",
                                                "resize": "vertical",
                                            },
                                            placeholder="Enter your SQL query here...",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Alert(
                                                    id="syntax-alert",
                                                    is_open=False,
                                                    dismissable=True,
                                                    className="mt-2",
                                                ),
                                                html.Div(
                                                    id="query-suggestions",
                                                    className="mt-2",
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
        dbc.Card(
            [
                dbc.CardHeader(
                    [
                        html.H3("Execution Results", className="mb-0"),
                        dbc.Badge(
                            "Idle",
                            color="secondary",
                            id="execution-status",
                            className="ms-2",
                        ),
                    ]
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            id="output-message",
                                            className="alert alert-success",
                                            style={"display": "none"},
                                        ),
                                        html.Div(
                                            id="error-message",
                                            className="alert alert-danger",
                                            style={"display": "none"},
                                        ),
                                    ],
                                    width=12,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(
                                                    "⏱️ Execution Time",
                                                    className="card-title text-primary",
                                                ),
                                                html.H3(
                                                    id="execution-time",
                                                    children="--",
                                                    className="text-primary",
                                                ),
                                            ]
                                        )
                                    )
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(
                                                    "📊 Result Rows",
                                                    className="card-title text-success",
                                                ),
                                                html.H3(
                                                    id="result-rows",
                                                    children="--",
                                                    className="text-success",
                                                ),
                                            ]
                                        )
                                    )
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(
                                                    "🤖 Predicted Cost",
                                                    className="card-title text-warning",
                                                ),
                                                html.H3(
                                                    id="estimated-cost-display",
                                                    children="--",
                                                    className="text-warning",
                                                ),
                                            ]
                                        )
                                    )
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(
                                                    "🎯 Accuracy",
                                                    className="card-title text-info",
                                                ),
                                                html.H3(
                                                    id="prediction-accuracy",
                                                    children="--",
                                                    className="text-info",
                                                ),
                                            ]
                                        )
                                    )
                                ),
                            ],
                            className="mb-4",
                        ),
                        dbc.Tabs(
                            [
                                dbc.Tab(label="📋 Query Results", tab_id="results-tab"),
                                dbc.Tab(
                                    label="🔍 Query Features", tab_id="features-tab"
                                ),
                                dbc.Tab(
                                    label="💡 Optimization Tips",
                                    tab_id="optimization-tab",
                                ),
                            ],
                            id="result-tabs",
                            active_tab="results-tab",
                        ),
                        html.Div(id="tab-content", className="mt-3"),
                    ]
                ),
            ],
            className="mb-4",
        ),
        dbc.Card(
            [
                dbc.CardHeader(
                    [
                        html.H3("📈 Performance Analytics", className="mb-0"),
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Refresh",
                                    id="refresh-graph-button",
                                    color="secondary",
                                    size="sm",
                                ),
                            ],
                            className="ms-auto",
                        ),
                    ]
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="performance-graph",
                                        style={"height": "400px"},
                                    ),
                                    width=8,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id="feature-importance-graph",
                                        style={"height": "400px"},
                                    ),
                                    width=4,
                                ),
                            ]
                        )
                    ]
                ),
            ],
            className="mb-4",
        ),
    ],
    fluid=True,
    className="p-4",
)


@app.callback(
    [
        Output("db-selector", "options"),
        Output("db-selector", "value"),
        Output("db-selection-status", "children"),
        Output("db-status-badge", "children"),
        Output("db-status-badge", "color"),
    ],
    Input("db-selector", "id"),
)
def load_databases(_):
    try:
        response = requests.get("http://127.0.0.1:8000/list_databases/")
        response.raise_for_status()
        data = response.json()
        db_files = data.get("databases", [])
        options = [{"label": db, "value": db} for db in db_files]
        default_value = db_files[0] if db_files else None
        return (
            options,
            default_value,
            f"Found {len(db_files)} database(s).",
            "Connected",
            "success",
        )
    except requests.exceptions.RequestException as e:
        return [], None, f"Error: {e}", "Disconnected", "danger"


@app.callback(
    [
        Output("model-selector", "options"),
        Output("model-selector", "value"),
        Output("model-selection-status", "children"),
    ],
    Input("model-selector", "id"),
)
def load_models(_):
    try:
        response = requests.get("http://127.0.0.1:8000/list_models/")
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        options = [{"label": m, "value": m} for m in models]
        default_value = models[0] if models else None
        return options, default_value, f"Found {len(models)} model(s)."
    except Exception as e:
        return [], None, f"Error loading models: {e}"


@app.callback(
    [
        Output("realtime-cost", "children"),
        Output("complexity-score", "children"),
        Output("optimization-progress", "value"),
        Output("analysis-status", "children"),
        Output("analysis-status", "color"),
        Output("syntax-alert", "children"),
        Output("syntax-alert", "is_open"),
        Output("syntax-alert", "color"),
        Output("query-suggestions", "children"),
    ],
    [Input("sql-input", "value")],
    [State("db-selector", "value"), State("model-selector", "value")],
)
def realtime_analysis(query_text, selected_db, selected_model):
    if not query_text or not query_text.strip():
        return "Type a query...", "N/A", 0, "Ready", "info", "", False, "info", ""
    if not selected_db or not selected_model:
        return (
            "Select database and model first",
            "N/A",
            0,
            "No DB/Model",
            "warning",
            "",
            False,
            "info",
            "",
        )
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict_cost/",
            json={
                "query": query_text.strip(),
                "db_name": selected_db,
                "model_name": selected_model,
            },
        )
        if response.status_code == 200:
            data = response.json()
            cost = data.get("estimated_cost", -1)
            features = data.get("features", {})
            suggestions = data.get("suggestions", [])
            complexity = features.get("query_complexity_score", 0)
            optimization_level = min(100, max(0, 100 - (complexity * 8)))
            cost_display = f"{cost:.2f} ms" if cost > 0 else "Unable to predict"
            complexity_display = f"Score: {complexity:.1f}"
            suggestion_elements = []
            if suggestions:
                suggestion_elements = [
                    html.H6("💡 Quick Tips:", className="text-warning mb-2"),
                    html.Ul(
                        [
                            html.Li(s, className="small text-muted")
                            for s in suggestions[:3]
                        ]
                    ),
                ]
            return (
                cost_display,
                complexity_display,
                optimization_level,
                "Analyzing",
                "success",
                "",
                False,
                "info",
                suggestion_elements,
            )
        else:
            return "Analysis failed", "N/A", 0, "Error", "danger", "", False, "info", ""
    except:
        return "Connection error", "N/A", 0, "Offline", "danger", "", False, "info", ""


@app.callback(
    [
        Output("output-message", "children"),
        Output("output-message", "style"),
        Output("error-message", "children"),
        Output("error-message", "style"),
        Output("execution-time", "children"),
        Output("result-rows", "children"),
        Output("estimated-cost-display", "children"),
        Output("prediction-accuracy", "children"),
        Output("execution-status", "children"),
        Output("execution-status", "color"),
        Output("execution-results-store", "data"),
    ],
    Input("execute-button", "n_clicks"),
    [
        State("sql-input", "value"),
        State("db-selector", "value"),
        State("model-selector", "value"),
    ],
)
def handle_query_execution(n_clicks, query_text, selected_db, selected_model):
    if not n_clicks:
        return (
            "",
            {"display": "none"},
            "",
            {"display": "none"},
            "--",
            "--",
            "--",
            "--",
            "Idle",
            "secondary",
            None,
        )
    if not selected_db or not selected_model:
        return (
            "",
            {"display": "none"},
            "Please select a database and model first.",
            {"display": "block"},
            "--",
            "--",
            "--",
            "--",
            "Error",
            "danger",
            None,
        )
    try:
        response = requests.post(
            "http://127.0.0.1:8000/execute_query/",
            json={
                "query": query_text,
                "db_name": selected_db,
                "model_name": selected_model,
            },
        )
        response.raise_for_status()
        data = response.json()
        execution_time = data.get("execution_time_ms", 0)
        estimated_cost = data.get("estimated_cost", -1)
        result_rows = data.get("result_rows", 0)
        features = data.get("features", {})
        suggestions = data.get("suggestions", [])
        results = data.get("results", [])
        accuracy = "N/A"
        if execution_time > 0 and estimated_cost > 0:
            error_pct = abs(execution_time - estimated_cost) / execution_time * 100
            accuracy = f"{max(0, 100 - error_pct):.1f}%"
        execution_results = {
            "results": results,
            "features": features,
            "suggestions": suggestions,
            "execution_time": execution_time,
            "estimated_cost": estimated_cost,
            "result_rows": result_rows,
        }
        return (
            data.get("message", "Query executed successfully."),
            {"display": "block"},
            "",
            {"display": "none"},
            f"{execution_time:.2f} ms",
            f"{result_rows:,}",
            f"{estimated_cost:.2f} ms" if estimated_cost > 0 else "N/A",
            accuracy,
            "Completed",
            "success",
            execution_results,
        )
    except requests.exceptions.RequestException as e:
        return (
            "",
            {"display": "none"},
            f"Execution failed: {str(e)}",
            {"display": "block"},
            "--",
            "--",
            "--",
            "--",
            "Failed",
            "danger",
            None,
        )


@app.callback(
    Output("tab-content", "children"),
    [Input("result-tabs", "active_tab"), Input("execution-results-store", "data")],
)
def update_tab_content(active_tab, execution_results):
    if not execution_results:
        return html.P(
            "No execution results available. Please execute a query first.",
            className="text-muted",
        )
    results = execution_results.get("results", [])
    features = execution_results.get("features", {})
    suggestions = execution_results.get("suggestions", [])
    if active_tab == "results-tab":
        if results:
            df = pd.DataFrame(results)
            return dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict("records"),
                style_table={
                    "overflowX": "auto",
                    "backgroundColor": "#343a40",
                    "border": "1px solid #495057",
                },
                style_cell={
                    "textAlign": "left",
                    "backgroundColor": "#343a40",
                    "color": "white",
                    "border": "1px solid #495057",
                    "padding": "8px",
                    "fontSize": "12px",
                },
                style_header={
                    "backgroundColor": "#212529",
                    "color": "white",
                    "fontWeight": "bold",
                },
                page_size=15,
                sort_action="native",
                filter_action="native",
            )
        else:
            return html.P("No results to display.", className="text-muted")
    elif active_tab == "features-tab":
        feature_cards = []
        for key, value in features.items():
            if isinstance(value, bool):
                value = "Yes" if value else "No"
                color = "success" if value == "Yes" else "secondary"
            elif isinstance(value, float):
                value = f"{value:.2f}"
                color = "info"
            else:
                color = "primary"
            feature_cards.append(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(
                                    key.replace("_", " ").title(),
                                    className="card-title",
                                ),
                                dbc.Badge(str(value), color=color, className="h5"),
                            ]
                        )
                    ),
                    width=3,
                    className="mb-2",
                )
            )
        return dbc.Row(feature_cards)
    elif active_tab == "optimization-tab":
        if suggestions:
            suggestion_items = []
            for i, suggestion in enumerate(suggestions):
                icon = "🚀" if i == 0 else "💡" if i == 1 else "⚡"
                suggestion_items.append(
                    dbc.ListGroupItem([html.Span(icon, className="me-2"), suggestion])
                )
            return dbc.ListGroup(suggestion_items, flush=True)
        else:
            return dbc.Alert(["No specific suggestions at this time."], color="success")
    return html.P("Select a tab to view content.", className="text-muted")


@app.callback(
    Output("performance-graph", "figure"),
    Input("refresh-graph-button", "n_clicks"),
)
def update_performance_graph(_):
    try:
        response = requests.get("http://127.0.0.1:8000/get_logs/")
        response.raise_for_status()
        logs_data = response.json()
        if not logs_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No query performance data yet.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
            fig.update_layout(
                title="Query Performance Analytics",
                template="plotly_dark",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            return fig
        df = pd.DataFrame(logs_data)
        fig = px.scatter(
            df,
            x="execution_time_ms",
            y="estimated_cost",
            title="ML Prediction vs. Actual Execution Time",
            labels={
                "execution_time_ms": "Actual Time (ms)",
                "estimated_cost": "ML Predicted Time (ms)",
            },
            template="plotly_dark",
            trendline="ols",
            hover_data={"execution_time_ms": ":.2f", "estimated_cost": ":.2f"},
        )
        min_val = min(df["execution_time_ms"].min(), df["estimated_cost"].min())
        max_val = max(df["execution_time_ms"].max(), df["estimated_cost"].max())
        fig.add_shape(
            type="line",
            x0=min_val,
            y0=min_val,
            x1=max_val,
            y1=max_val,
            line=dict(color="red", width=2, dash="dash"),
        )
        fig.update_layout(height=400, showlegend=True)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading performance data:<br>{str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        fig.update_layout(template="plotly_dark")
        return fig


@app.callback(
    Output("feature-importance-graph", "figure"),
    Input("refresh-graph-button", "n_clicks"),
)
def update_feature_importance_graph(_):
    try:
        response = requests.get("http://127.0.0.1:8000/get_feature_importance/")
        if response.status_code == 200:
            data = response.json()
            importance_data = data.get("feature_importance", {})
            if importance_data:
                features = list(importance_data.keys())[:10]
                importance = list(importance_data.values())[:10]
                fig = px.bar(
                    x=importance,
                    y=features,
                    orientation="h",
                    title="Top 10 ML Feature Importance",
                    template="plotly_dark",
                    labels={"x": "Importance Score", "y": "Query Features"},
                    color=importance,
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(height=400, showlegend=False)
                return fig
        fig = go.Figure()
        fig.add_annotation(
            text="ML Feature importance will appear after training",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(
            title="ML Model Feature Importance",
            template="plotly_dark",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400,
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=12, color="red"),
        )
        fig.update_layout(template="plotly_dark", height=400)
        return fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8050)
