import bson
import datetime
import fitbit
from flask import Flask, jsonify, render_template, request, Response
import logging
import os
import requests

# App
from configure import app

# add the mlpred folder
import sys
sys.path.insert(0, '../mlPredictor')
import mitAI_predEng

# Log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('logs/server.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

top_k_features = None

@app.route('/')
def register():
    return render_template('index.html')


#http://127.0.0.1:8080/top-k

@app.route('/revisedtop-k', methods=['POST'])
def revisedtop_k():


    print("request.get_json() >>>>>> ", (request.get_json(force=True)))
    white_list = request.get_json(force=True)
    print(white_list['revisedtop_k'])
    ans = mitAI_predEng.train_model(white_list['revisedtop_k'],True)
    print("top Features sorted by importance", ans)
    return jsonify(ans)



@app.route('/top-k', methods=['POST'])
def top_k():

    white_list = []
    ans = mitAI_predEng.train_model(white_list,True)
    print("top Features sorted by importance", ans)
    return jsonify(ans)


def pred_accuracy(top_k_features):

    white_list = top_k_features
    pred_accuracy = mitAI_predEng.train_model(white_list,False)
    print("pred_accuracy", pred_accuracy)

    return pred_accuracy


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("8080"), debug=True)
