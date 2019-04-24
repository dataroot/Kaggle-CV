from flask_cors import CORS
from flask import request
from flask import Flask, Response
from flask import render_template

import json
import sys
import os

import pandas as pd

import pickle
from datetime import datetime

from utils.utils import TextProcessor
from models.distance import DistanceModel
from recommender.predictor import Predictor, Formatter
from preprocessors.queproc import QueProc
from preprocessors.proproc import ProProc

pd.set_option('display.max_columns', 100, 'display.width', 1024)

# Set oath to data
DATA_PATH = 'data'
SAMPLE_PATH = 'demo_data'
DUMP_PATH = 'dump'

# init model
model = DistanceModel(que_dim= 34 - 2 + 8 - 2,
                                  que_input_embs=[102, 42], que_output_embs=[2, 2],
                                  pro_dim=42 - 2,
                                  pro_input_embs=[102, 102, 42], pro_output_embs=[2, 2, 2],
                                  inter_dim=20, output_dim=10)
# load weights
model.load_weights(os.path.join(DUMP_PATH, 'model.h5'))

# load dumped data
with open(os.path.join(DUMP_PATH, 'dump.pkl'), 'rb') as file:
    d = pickle.load(file)
    que_data = d['que_data']
    stu_data = d['stu_data']
    pro_data = d['pro_data']
    que_proc = d['que_proc']
    pro_proc = d['pro_proc']
    que_to_stu = d['que_to_stu']
    pos_pairs = d['pos_pairs']

# init text processor
tp = TextProcessor()

# prepare the data
professionals_sample = pd.read_csv(os.path.join(SAMPLE_PATH, 'pro_sample.csv'))
pro_tags_sample = pd.read_csv(os.path.join(SAMPLE_PATH, 'tag_users_sample.csv'))

answers = pd.read_csv(os.path.join(DATA_PATH, 'answers.csv'))
questions = pd.read_csv(os.path.join(DATA_PATH, 'questions.csv'))

professionals_sample['professionals_date_joined'] = pd.to_datetime(professionals_sample['professionals_date_joined'], infer_datetime_format=True)

answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'], infer_datetime_format=True)
answers['answers_body'] = answers['answers_body'].apply(tp.process)

questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'], infer_datetime_format=True)
questions['questions_title'] = questions['questions_title'].apply(tp.process)
questions['questions_body'] = questions['questions_body'].apply(tp.process)
questions['questions_whole'] = questions['questions_title'] + ' ' + questions['questions_body']

pred = Predictor(model, que_data, stu_data, pro_data, que_proc, pro_proc, que_to_stu, pos_pairs)
formatter = Formatter(DATA_PATH)

# init flask server
app = Flask(__name__, static_url_path='', template_folder='views')
CORS(app) 

# Routes
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
  try:
    pro_dict = {
        'professionals_id': [],
        'professionals_location': [],
        'professionals_industry': [],
        'professionals_headline': [],
        'professionals_date_joined': [],
        'professionals_subscribed_tags': []
      }

    data = request.get_json()

    pro = professionals_sample[professionals_sample['professionals_id'] == data['professionals_id']]
    pro = pro.to_dict('records')[0]

    tag = pro_tags_sample[pro_tags_sample['tag_users_user_id'] == data['professionals_id']]

    for key, val in pro.items():
        if key in pro_dict and val:
          pro_dict[key].append(str(val))
    
    pro_dict['professionals_subscribed_tags'].append(' '.join(list(tag['tags_tag_name'])))    
    
    for key, val in pro_dict.items():
      if not val:
         return json.dumps([], default=str)
    
    pro_df, pro_tags = Formatter.convert_pro_dict(pro_dict)
    tmp = pred.find_ques_by_pro(pro_df, questions, answers, pro_tags)
    final_df = formatter.get_que(tmp).fillna('')
    
    final_data = final_df.to_dict('records')
    
    return json.dumps(final_data, allow_nan=False) 
      
  except Exception as e:
    return json.dumps([], default=str)

if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0', port = 8000)
