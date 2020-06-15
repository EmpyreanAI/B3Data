import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt

import plotly
import plotly.graph_objects as go
import pandas
from dash.dependencies import Input, Output
import urllib.request, json
import os
import sys

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('B3 Resource Allocation Manager'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='loss'),
        dcc.Graph(id='ep_ret'),
        dcc.Graph(id='qval'),
        dcc.Interval(
            id='interval',
            interval=10000, # in milliseconds
            n_intervals=0
        )
    ])
)

def graph_loss(data):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 250

    fig.append_trace({
        'x': data['Epoch'],
        'y': data['AverageEpRet'],
        'name': 'AverageEpRet',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': data['Epoch'],
        'y': data['AverageTestEpRet'],
        'name': 'AverageTestEpRet',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)

    return fig

def graph_ep_ret(data):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 250

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
    }, 1, 1)

    return fig

def graph_qval(data):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 50
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 250
    fig['layout']['title'] = "Q-Valor"

    fig.append_trace({
        'x': data['Epoch'],
        'y': data['AverageQVals'],
        'name': 'QValMedio',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)

    fig.append_trace({
        'x': data['Epoch'],
        'y': data['MaxQVals'],
        'name': 'QValMax',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)

    fig.append_trace({
        'x': data['Epoch'],
        'y': data['MinQVals'],
        'name': 'QValMin',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)

    return fig

@app.callback([Output('loss', 'figure'), Output('ep_ret', 'figure'),  Output('qval', 'figure')],
              [Input('interval', 'n_intervals')])
def get_data(n):
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
    
    df_data = pandas.DataFrame(result)

    return graph_loss(df_data), graph_ep_ret(df_data), graph_qval(df_data)


if __name__ == '__main__':
    app.run_server(debug=True)