import numpy as np
import pandas as pd
import ujson
import os
from flask import Flask, jsonify, request
from flask import flash, redirect, url_for
from flask import send_from_directory
from flask_cors import CORS
from helper.dataset_v3 import load_dataset, load_dataset_overview, available_datasets

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_root():
  return "Hello World! v3", 200

@app.route('/echo',  methods = ['GET', 'POST'])
def echo():
  if request.method == 'GET':
    return ujson.dumps(request.args.to_dict()), 200
  if request.method == 'POST':
    print(request.args)
    print(request.form)
    return "this is post", 200

@app.route('/v3/builtin/available')
def builtin_available():
  return available_datasets(), 200

@app.route('/v3/builtin/load')
def builtin_load_df():
  dataset_name = request.args.get("dataset")
  file_name = request.args.get("filename")
  try:
    variant_type = request.args.get("variant",default="train")
  except:
    variant_type = "train"
  result = load_dataset(dataset_name=dataset_name,variant=variant_type,file_name=file_name)
  return result.to_json(orient="index"), 200

@app.route('/v3/builtin/load/overview')
def builtin_load_overview():
  dataset_name=request.args.get("dataset")
  file_name=request.args.get("filename")
  result = load_dataset_overview(dataset_name=dataset_name,file_name=file_name)
  return result.to_json(orient="records"), 200

@app.route('/v3/builtin/stat/label')
def builtin_stat_label():
  dataset_name=request.args.get("dataset")
  y_train = load_dataset(dataset_name=dataset_name,variant="train",file_name="label")
  stat_per_data = y_train.drop("id",axis=1).sum(axis=1).to_json(orient="index")
  stat_per_label = y_train.drop("id",axis=1).sum(axis=0).to_json(orient="index")
  result = {"stat_per_data": ujson.loads(stat_per_data), "stat_per_label": ujson.loads(stat_per_label)}
  return ujson.dumps(result), 200

if __name__ == "__main__":
  app.run(debug=True,host="127.0.0.1",port=5003)