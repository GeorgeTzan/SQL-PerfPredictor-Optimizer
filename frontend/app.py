from dash import Dash, html, dcc, Input, Output, State, dash_table
import requests
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.H1("SQL Query Performance Predictor & Optimizer"),
    
    html.H2("Execute SQL Query"),
    dcc.Textarea(
        id='sql-input',
        value='SELECT * FROM sales LIMIT 10;',
        style={'width': '80%', 'height': 150}
    ),
    html.Button('Execute Query', id='execute-button', n_clicks=0),
    html.Hr(),

    html.H3("Query Execution Results"),
    html.Div(id='output-message', style={'color': 'green'}),
    html.Div(id='error-message', style={'color': 'red'}),
    html.P(id='execution-time'), 
    html.P(id='result-rows'),
    html.P(id='estimated-cost-display'),
    html.Details([
        html.Summary("Extracted Features"),
        html.Div(id='extracted-features')
    ]),
    html.H4("Query Results Table"),
    dash_table.DataTable(
        id='query-results-table', 
        columns=[{"name": i, "id": i} for i in []], 
        data=[],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10
    ),
    html.Hr(),

    html.H3("Optimisation Suggestions (Coming Soon)"),
    html.Div("This section will provide intelligent SQL optimization recommendations.")
])

@app.callback(
    Output('output-message', 'children'),
    Output('error-message', 'children'),
    Output('execution-time', 'children'),
    Output('result-rows', 'children'),
    Output('estimated-cost-display', 'children'),
    Output('extracted-features', 'children'),
    Output('query-results-table', 'columns'),
    Output('query-results-table', 'data'),
    Input('execute-button', 'n_clicks'),
    State('sql-input', 'value')
)

def handle_query_execution(n_clicks, query_text):
    if n_clicks == 0:
        return "", "", "", "", "", "", [], []
    backend_url = "http://127.0.0.1:8000/execute_query/" # development URL
    try:
        response = requests.post(backend_url, json={'query': query_text})
        response.raise_for_status()
        data = response.json()

        # results
        df = pd.DataFrame(data['results'])
        columns = [{"name": i, "id": i} for i in df.columns]
        table_data = df.to_dict('records')

        return (
                data.get('message', 'Query executed.'),
                '', 
                f"Execution Time: {data.get('execution_time_ms', 0):.2f} ms",
                f"Result Rows: {data.get('result_rows', 0)}",
                f"Estimated Cost (ML): {data.get('estimated_cost', -1.0):.2f}",
                html.Pre(str(data.get('features', {}))), 
                columns,
                table_data
            )
    except requests.exceptions.RequestException as e:
        return "", str(e), "", "", "", "", [], []
    
if __name__ == '__main__':
    app.run(debug=True)