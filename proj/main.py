import os

# comment

import pandas as pd

from nlp import pipeline_d2v, pipeline_lda
from processors import QueProc, StuProc, ProProc
from generator import BatchGenerator
from models import DistanceModel, SimpleModel, ConcatModel, Adam
from evaluation import permutation_importance, plot_fi
from utils import TextProcessor

pd.set_option('display.max_columns', 100, 'display.width', 1024)
pd.options.mode.chained_assignment = None

DATA_PATH, SPLIT_DATE = '../../data/', '2019-01-01'

tp = TextProcessor()

# ######################################################################################################################
#
#                                                       READ
#
# ######################################################################################################################

ans = pd.read_csv(os.path.join(DATA_PATH, 'answers.csv'), parse_dates=['answers_date_added'])
ans['answers_body'] = ans['answers_body'].apply(tp.process)
ans_train = ans[ans['answers_date_added'] < SPLIT_DATE]

que = pd.read_csv(os.path.join(DATA_PATH, 'questions.csv'), parse_dates=['questions_date_added'])
que['questions_title'] = que['questions_title'].apply(tp.process)
que['questions_body'] = que['questions_body'].apply(tp.process)
que_train = que[que['questions_date_added'] < SPLIT_DATE]

pro = pd.read_csv(os.path.join(DATA_PATH, 'professionals.csv'), parse_dates=['professionals_date_joined'])
pro['professionals_headline'] = pro['professionals_headline'].apply(tp.process)
pro['professionals_industry'] = pro['professionals_industry'].apply(tp.process)
pro_train = pro[pro['professionals_date_joined'] < SPLIT_DATE]

stu = pd.read_csv(os.path.join(DATA_PATH, 'students.csv'), parse_dates=['students_date_joined'])
stu_train = stu[stu['students_date_joined'] < SPLIT_DATE]

tags = pd.read_csv(os.path.join(DATA_PATH, 'tags.csv'))
tags['tags_tag_name'] = tags['tags_tag_name'].apply(lambda x: tp.process(x, allow_stopwords=True))

tag_que = pd.read_csv(os.path.join(DATA_PATH, 'tag_questions.csv')) \
    .merge(tags, left_on='tag_questions_tag_id', right_on='tags_tag_id')
tag_users = pd.read_csv(os.path.join(DATA_PATH, 'tag_users.csv')) \
    .merge(tags, left_on='tag_users_tag_id', right_on='tags_tag_id')

# ######################################################################################################################
#
#                                               ADDITIONAL PREPARATION
#
# ######################################################################################################################

# mappings from question's id to its author id. Used in Predictor
que_to_stu = {row['questions_id']: row['questions_author_id'] for i, row in que.iterrows()}

# mappings from professional's id to his registration date. Used in batch generator
pro_to_date = {row['professionals_id']: row['professionals_date_joined'] for i, row in pro.iterrows()}

# construct dataframe used to extract positive pairs
pairs_df = que.merge(ans, left_on='questions_id', right_on='answers_question_id') \
    .merge(pro, left_on='answers_author_id', right_on='professionals_id') \
    .merge(stu, left_on='questions_author_id', right_on='students_id')

pairs_df = pairs_df[['questions_id', 'students_id', 'professionals_id', 'answers_date_added']]

# ######################################################################################################################
#
#                                                       TRAIN
#
# ######################################################################################################################

# calculate and save tag and industry embeddings on train data
tag_embs, ind_embs, ques_d2v = pipeline_d2v(que_train, ans_train, pro_train, tag_que, 10)
lda_dic, lda_tfidf, lda_model = pipeline_lda(que_train, tag_que, 10)

# extract positive pairs
pos_pairs = list(pairs_df.loc[pairs_df['answers_date_added'] < SPLIT_DATE].itertuples(index=False, name=None))

# extract and preprocess feature for all three main entities

que_proc = QueProc(tag_embs, ques_d2v, lda_dic, lda_tfidf, lda_model)
que_data = que_proc.transform(que_train, tag_que)

stu_proc = StuProc()
stu_data = stu_proc.transform(stu_train, que_train, ans_train)

pro_proc = ProProc(tag_embs, ind_embs)
pro_data = pro_proc.transform(pro_train, que_train, ans_train, tag_users)

bg = BatchGenerator(que_data, stu_data, pro_data, 64, pos_pairs, pos_pairs, pro_to_date)

# ######################################################################################################################
#
#                                                       MODEL
#
# ######################################################################################################################

# in train mode, build, compile train and save model
model = DistanceModel(que_dim=len(que_data.columns) - 2 + len(stu_data.columns) - 2,
                      que_input_embs=[102, 42], que_output_embs=[2, 2],
                      pro_dim=len(pro_data.columns) - 2,
                      pro_input_embs=[102, 102, 42], pro_output_embs=[2, 2, 2],
                      inter_dim=20, output_dim=10)

model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(bg, epochs=5, verbose=2)

model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(bg, epochs=10, verbose=2)

# ######################################################################################################################
#
#                                                   EVALUATION
#
# ######################################################################################################################

# dummy batch generator used to extract single big batch of data to calculate feature importance
bg = BatchGenerator(que_data, stu_data, pro_data, 1024, pos_pairs, pos_pairs, pro_to_date)

# dict with descriptions of feature names, used for visualization of feature importance
fn = {"que": list(stu_data.columns[2:]) + list(que_data.columns[2:]),
      "pro": list(pro_data.columns[2:])}

# calculate and plot feature importance
fi = permutation_importance(model, bg[0][0][0], bg[0][0][1], bg[0][1], fn, n_trials=3)
plot_fi(fi)

# ######################################################################################################################
#
#                                                       TEST
#
# ######################################################################################################################

# non-negative pairs are all known positive pairs to the moment
nonneg_pairs = pos_pairs

# extract positive pairs
pos_pairs = list(pairs_df.loc[pairs_df['answers_date_added'] >= SPLIT_DATE].itertuples(index=False, name=None))
nonneg_pairs += pos_pairs

# extract and preprocess feature for all three main entities

que_proc = QueProc(tag_embs, ques_d2v, lda_dic, lda_tfidf, lda_model)
que_data = que_proc.transform(que, tag_que)

stu_proc = StuProc()
stu_data = stu_proc.transform(stu, que, ans)

pro_proc = ProProc(tag_embs, ind_embs)
pro_data = pro_proc.transform(pro, que, ans, tag_users)

bg = BatchGenerator(que_data, stu_data, pro_data, 64, pos_pairs, nonneg_pairs, pro_to_date)

# ######################################################################################################################
#
#                                                   EVALUATION
#
# ######################################################################################################################

loss, acc = model.evaluate_generator(bg)
print(f'Loss: {loss}, accuracy: {acc}')

# dummy batch generator used to extract single big batch of data to calculate feature importance
bg = BatchGenerator(que_data, stu_data, pro_data, 1024, pos_pairs, nonneg_pairs, pro_to_date)

# dict with descriptions of feature names, used for visualization of feature importance
fn = {"que": list(stu_data.columns[2:]) + list(que_data.columns[2:]),
      "pro": list(pro_data.columns[2:])}

# calculate and plot feature importance
fi = permutation_importance(model, bg[0][0][0], bg[0][0][1], bg[0][1], fn, n_trials=3)
plot_fi(fi)
