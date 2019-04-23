from flask_cors import CORS
from flask import request
from flask import Flask, Response
from flask import render_template

import numpy as np
import pandas as pd

import traceback
import json
import sys

import pickle

from datetime import datetime

from models import DistanceModel
from predictor import Predictor, Formatter

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

pred = Predictor(model, que_data, stu_data, pro_data, que_proc, pro_proc, que_to_stu, pos_pairs)
formatter = Formatter('data/')


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

      print(que_dict)

      for key, val in que_dict.items():
        if not val:
           return json.dumps([], default=str)

      que_df, que_tags = Formatter.convert_que_dict(que_dict)
      tmp = pred.find_ques_by_que(que_df, que_tags)
      final_df = formatter.get_que(tmp)

      final_data = final_df.to_dict('records')

      print("RETURN")
      print(len(final_data))
      print(final_data)
      return json.dumps(final_data, allow_nan=False) 

    except Exception as e:
      print(e)
      return json.dumps([], default=str)


@app.route("/api/professional", methods = ['POST'])
def professional():
    a = [{'professionals_id': '51a38e3c7e264f27a71ccd481ee46321',
          'professionals_location': 'India',
          'professionals_industry': 'Telecommunications',
          'professionals_headline': 'Software Developer',
          'professionals_date_joined': '2017-02-10 08:15:31 UTC+0000',
          'tags_tag_name': 'computer-science science technology tech academic-advising telecommunications',
          'id': '332a511f1569444485cf7a7a556a5e54',
          'match_id': '51a38e3c7e264f27a71ccd481ee46321',
          'match_score': 0.7625184105656923},
         {'professionals_id': '642328d3d46b44f4b515c7c47f305d02',
          'professionals_location': None,
          'professionals_industry': 'Telecommunications',
          'professionals_headline': 'Security Operations Engineer',
          'professionals_date_joined': '2019-01-16 20:44:55 UTC+0000',
          'tags_tag_name': 'computer-science college-major technology stem programming cyber cybersecurity2 digitalforensics',
          'id': '332a511f1569444485cf7a7a556a5e54',
          'match_id': '642328d3d46b44f4b515c7c47f305d02',
          'match_score': 0.7511400120199462},
         {'professionals_id': 'f8cc03f45f1c4484936e08deec76f66e',
          'professionals_location': 'Wilmington, North Carolina',
          'professionals_industry': 'Telecommunications',
          'professionals_headline': 'CORe Supervisor',
          'professionals_date_joined': '2018-08-14 23:03:31 UTC+0000',
          'tags_tag_name': 'science math telecommunications',
          'id': '332a511f1569444485cf7a7a556a5e54',
          'match_id': 'f8cc03f45f1c4484936e08deec76f66e',
          'match_score': 0.7503653940265411},
         {'professionals_id': '07fb7f2d716e40198042d0e8f7f91f2d',
          'professionals_location': None,
          'professionals_industry': 'Telecommunications',
          'professionals_headline': 'Software Engineer',
          'professionals_date_joined': '2018-01-05 15:42:01 UTC+0000',
          'tags_tag_name': 'college engineering physics electrical-engineering software-development telecommunications lte',
          'id': '332a511f1569444485cf7a7a556a5e54',
          'match_id': '07fb7f2d716e40198042d0e8f7f91f2d',
          'match_score': 0.7360997486535724},
         {'professionals_id': '8fdb6e2afe1d4e07bb34c64a6512dc80',
          'professionals_location': 'Huntsville, Alabama',
          'professionals_industry': 'Telecommunications',
          'professionals_headline': 'Corrdinator-Government Operations',
          'professionals_date_joined': '2016-10-10 15:58:43 UTC+0000',
          'tags_tag_name': 'technology tech telecommunications',
          'id': '332a511f1569444485cf7a7a556a5e54',
          'match_id': '8fdb6e2afe1d4e07bb34c64a6512dc80',
          'match_score': 0.726532927897221},
         {'professionals_id': 'bbfce5bf0f624031bbd1169661f25f5a',
          'professionals_location': None,
          'professionals_industry': 'Telecommunications',
          'professionals_headline': 'Director, Network Systems',
          'professionals_date_joined': '2018-07-31 18:26:55 UTC+0000',
          'tags_tag_name': 'computer programming software telecommunications computer-science2 computer-software2 engineering3',
          'id': '332a511f1569444485cf7a7a556a5e54',
          'match_id': 'bbfce5bf0f624031bbd1169661f25f5a',
          'match_score': 0.7201444190321711},
         {'professionals_id': 'd017af856f7e480bb623faa5d7570d52',
          'professionals_location': 'Cedar Grove, New Jersey',
          'professionals_industry': 'Telecommunications',
          'professionals_headline': 'IT Senior Manager - Data GoverNonece & Advanced Analytics',
          'professionals_date_joined': '2018-02-20 22:52:08 UTC+0000',
          'tags_tag_name': 'computer-science science technology internships stem math mathematics stem-education data-science interviewing interview telecommunications analytics big-data #analytics bigdata big-data-technologies #computer-science data-analytics',
          'id': '332a511f1569444485cf7a7a556a5e54',
          'match_id': 'd017af856f7e480bb623faa5d7570d52',
          'match_score': 0.7183257675060286},
         {'professionals_id': '3e9374e92a8f4b72b86b2e7f907985fb',
          'professionals_location': 'New Jersey, New Jersey',
          'professionals_industry': 'Telecommunications',
          'professionals_headline': 'FiNonece Professional',
          'professionals_date_joined': '2016-10-10 14:50:19 UTC+0000',
          'tags_tag_name': 'technology tech pharmaceuticals telecommunications pharma',
          'id': '332a511f1569444485cf7a7a556a5e54',
          'match_id': '3e9374e92a8f4b72b86b2e7f907985fb',
          'match_score': 0.7158327791045083},
         {'professionals_id': '89a4ab42ac0548cf919ca2d8ec161d74',
          'professionals_location': 'Hurst, Texas',
          'professionals_industry': 'Telecommunications',
          'professionals_headline': 'IT Solutions Architect',
          'professionals_date_joined': '2018-02-23 17:07:15 UTC+0000',
          'tags_tag_name': 'physics information-technology it telecommunications 4g 5g',
          'id': '332a511f1569444485cf7a7a556a5e54',
          'match_id': '89a4ab42ac0548cf919ca2d8ec161d74',
          'match_score': 0.7135324425815932},
         {'professionals_id': '0ebeaa2745704d6bb57939288e53401e',
          'professionals_location': None,
          'professionals_industry': 'Telecommunications',
          'professionals_headline': 'System Administrator',
          'professionals_date_joined': '2018-06-22 14:21:08 UTC+0000',
          'tags_tag_name': 'telecommunications #computer #computer-science #devops #engineering #servers #unix',
          'id': '332a511f1569444485cf7a7a556a5e54',
          'match_id': '0ebeaa2745704d6bb57939288e53401e',
          'match_score': 0.7123323957151446}]

    return json.dumps(a, default=str)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port = 8000)