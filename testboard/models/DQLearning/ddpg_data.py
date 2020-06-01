from flask import Flask
import os
import sys
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/get_last_train')
def get_last_train():
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
    
    return json.dumps(result)


app.run(debug=True)