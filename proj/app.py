from flask_cors import CORS
from flask import request
from flask import Flask, Response
from flask import render_template

import traceback
import numpy as np
import pandas as pd

import traceback
import json
import sys
import os

import pickle

from datetime import datetime

from utils import TextProcessor

from models import DistanceModel
from predictor import Predictor, Formatter



DATA_PATH = 'data'

tp = TextProcessor()

model = DistanceModel(que_dim= 34 - 2 + 8 - 2,
                                  que_input_embs=[102, 42], que_output_embs=[2, 2],
                                  pro_dim=27 - 2,
                                  pro_input_embs=[102, 102, 42], pro_output_embs=[2, 2, 2],
                                  inter_dim=20, output_dim=10)
with open('dump.pkl', 'rb') as file:
    d = pickle.load(file)
    que_data = d['que_data']
    stu_data = d['stu_data']
    pro_data = d['pro_data']
    que_proc = d['que_proc']
    pro_proc = d['pro_proc']
    que_to_stu = d['que_to_stu']
    pos_pairs = d['pos_pairs']

professionals_sample = pd.read_csv(os.path.join(DATA_PATH, 'pro_sample.csv')).drop(columns='Unnamed: 0')
pro_tags_sample = pd.read_csv(os.path.join(DATA_PATH, 'tag_users_sample.csv')).drop(columns='Unnamed: 0')

answers = pd.read_csv(os.path.join(DATA_PATH, 'answers.csv'))
questions = pd.read_csv(os.path.join(DATA_PATH, 'questions.csv'))

answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'], infer_datetime_format=True)
answers['answers_body'] = answers['answers_body'].apply(tp.process)

questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'], infer_datetime_format=True)
questions['questions_title'] = questions['questions_title'].apply(tp.process)
questions['questions_body'] = questions['questions_body'].apply(tp.process)
questions['questions_whole'] = questions['questions_title'] + ' ' + questions['questions_body']

pred = Predictor(model, que_data, stu_data, pro_data, que_proc, pro_proc, que_to_stu, pos_pairs)
formatter = Formatter('data')


app = Flask(__name__, static_url_path='', template_folder='views')
CORS(app) 

@app.route('/')
def index():
  return render_template('index.html')


@app.route("/api/question", methods = ['POST'])
def question():
    try:
      que_dict = {
          'questions_id': ['0'],
          'questions_author_id': [],
          'questions_date_added': [str(datetime.now())],
          'questions_title': [],
          'questions_body': [],
          'questions_tags': []
      }

      data = request.get_json()

      for key, val in data.items():
        if key in que_dict and val:
          que_dict[key].append(str(val))


      for key, val in que_dict.items():
        if not val:
           return json.dumps([], default=str)

      que_df, que_tags = Formatter.convert_que_dict(que_dict)
      tmp = pred.find_ques_by_que(que_df, que_tags)
      final_df = formatter.get_que(tmp).fillna('')
      final_data = final_df.to_dict('records')

      return json.dumps(final_data, allow_nan=False) 

    except Exception as e:
      return json.dumps([], default=str)



@app.route("/api/professional", methods = ['POST'])
def professional():
  # try:
  pro_dict = {
      'professionals_id': [],
      'professionals_location': [],
      'professionals_industry': [],
      'professionals_headline': [],
      'professionals_date_joined': [],
      'professionals_subscribed_tags': []
    }

  data = request.get_json()

  print(1)

  pro = professionals_sample[professionals_sample['professionals_id'] == data['professionals_id']]
  pro['professionals_date_joined'] = pd.to_datetime(pro['professionals_date_joined'])
  pro = pro.to_dict('records')[0]



  print(2)
  tag = pro_tags_sample[pro_tags_sample['tag_users_user_id'] == data['professionals_id']]
  print(3)
  for key, val in pro.items():
      if key in pro_dict and val:
        pro_dict[key].append(str(val))
  print(4)
  pro_dict['professionals_subscribed_tags'].append(' '.join(list(tag['tags_tag_name'])))    
  print(5)
  for key, val in pro_dict.items():
    if not val:
       return json.dumps([], default=str)
  print(5)

  pro_df, pro_tags = Formatter.convert_pro_dict(pro_dict)
  print(6)
  tmp = pred.find_ques_by_pro(pro_df, questions, answers, pro_tags)
  print(7)
  final_df = formatter.get_que(tmp).fillna('')
  print(8)
  final_data = final_df.to_dict('records')
  print(9)
  print("RETURN")
  print(len(final_data))
  print(final_data)
  return json.dumps(final_data, allow_nan=False) 
    
  # except Exception as e:
  #   return json.dumps([], default=str)

if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0', port = 8000)
