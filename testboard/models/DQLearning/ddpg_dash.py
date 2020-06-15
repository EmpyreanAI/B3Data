import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
import pandas
from dash.dependencies import Input, Output
import urllib.request, json
import os
import sys

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('DDPG DashBoard'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=10000, # in milliseconds
            n_intervals=0
        )
    ])
)

# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    data = get_data()

    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=4, cols=1, vertical_spacing=0.05)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 1000

    fig.append_trace({
        'x': data['Epoch'],
        'y': data['LossQ'],
        'name': 'LossQ',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': data['Epoch'],
        'y': data['LossPi'],
        'name': 'LossPi',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 2, 1)
    fig.append_trace({
        'x': data['Epoch'],
        'y': data['AverageEpRet'],
        'name': 'AverageEpRet',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 3, 1)
    fig.append_trace({
        'x': data['Epoch'],
        'y': data['AverageTestEpRet'],
        'name': 'AverageTestEpRet',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 4, 1)

    return fig


def get_data():
    progress = []
    result = {}
    for root, _, files in os.walk(os.path.join(sys.argv[1], 'data')):
        for file in files:
            if file.endswith(".txt"):
                progress.append(os.path.join(root, file))

    last_train = progress[-1]

    columns = open(last_train).readlines()[0].split('\t')

    for i in columns:
        result[i] = []

    for line in open(last_train).readlines()[1:]:
        for i, value in enumerate(line.split('\t')):
            result[columns[i]].append(float(value))
    
    return pandas.DataFrame(result)

if __name__ == '__main__':
    app.run_server(debug=True)