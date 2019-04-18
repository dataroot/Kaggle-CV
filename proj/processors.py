import pickle

import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from tqdm import tqdm

from baseproc import BaseProc
from utils import Averager

pd.options.mode.chained_assignment = None  # warning suppression

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.ldamodel import LdaModel
from utils import TextProcessor

# TODO: add regular expressions for HTML tags removal somewhere
# TODO: make feature names shorter

class QueProc(BaseProc):
    """
    Questions data preprocessor
    """

    def __init__(self, oblige_fit, path):
        super().__init__(oblige_fit, path)

        with open(path + 'tags_embs.pkl', 'rb') as file:
            self.embs = pickle.load(file)

        self.tp = TextProcessor(path)
        self.lda_dic = Dictionary.load(path + 'questions.lda_dic')
        self.lda_tfidf = TfidfModel.load(path + 'questions.lda_tfidf')
        self.lda_model = LdaModel.load(path + 'questions.lda_model')

        self.features = {
            'categorical': [],
            'numerical': {
                'zero': ['questions_body_length', 'questions_tag_count'],
                'mean': []
            },
            'date': []  # ['questions_date_added']
        }

        self._unroll_features()

    def transform(self, que, tags):
        # process tags
        tags['tags_tag_name'] = tags['tags_tag_name'].apply(lambda x: self.tp.process(x, allow_stopwords=True))

        que['questions_time'] = que['questions_date_added']
        que['questions_body_length'] = que['questions_body'].apply(lambda s: len(str(s)))

        # append aggregated tags to each question
        tags_grouped = tags.groupby('tag_questions_question_id', as_index=False)[['tags_tag_name']] \
            .agg(lambda x: ' '.join(set(x)))
        tags_grouped['questions_tag_count'] = tags_grouped['tags_tag_name'].apply(lambda x: len(x.split()))
        df = que.merge(tags_grouped, how='left', left_on='questions_id', right_on='tag_questions_question_id')

        # launch feature pre-processing
        self.preprocess(df)

        # prepare tag embeddings

        tag_emb_len = list(self.embs.values())[0].shape[0]

        def __convert(s):
            embs = []
            for tag in str(s).split():
                if tag in self.embs:
                    embs.append(self.embs[tag])
            if len(embs) == 0:
                embs.append(np.zeros(tag_emb_len))
            return np.vstack(embs).mean(axis=0)

        mean_embs = df['tags_tag_name'].apply(__convert)

        emb_len = len(self.lda_model[[]])
        print("LDA embedding length:", emb_len)

        lda_tokens = df['questions_title'] + ' ' + df['questions_body']
        lda_tokens = lda_tokens.apply(self.tp.process, allow_stopwords=False)
        lda_tokens = lda_tokens.apply(lambda s: s.split())

        lda_corpus = [self.lda_dic.doc2bow(doc) for doc in lda_tokens]
        lda_corpus = self.lda_tfidf[lda_corpus]

        que_embs = self.lda_model.inference(lda_corpus)[0]
        print("LDA question embeddings shape:", que_embs.shape)

        # re-order the columns
        df = df[['questions_id', 'questions_time'] + self.features['all']]

        # append tag embeddings
        for i in range(tag_emb_len):
            df[f'que_tag_emb_{i}'] = mean_embs.apply(lambda x: x[i])

        for i in range(emb_len):
            df[f'que_emb_{i}'] = que_embs[:, i]
        print("Done!")

        return df


# TODO: also compute features for students with no questions and professionals with no answers

class StuProc(BaseProc):
    """
    Students data preprocessor
    """

    def __init__(self, oblige_fit, path):
        super().__init__(oblige_fit, path)

        self.features = {
            'categorical': [('students_location', 100), ('students_state', 40)],
            'numerical': {
                'zero': ['students_questions_asked'],
                'mean': ['students_average_question_body_length', 'students_average_answer_body_length',
                         'students_average_answer_amount']
            },
            'date': []  # ['students_date_joined', 'students_previous_question_time']
        }

        self._unroll_features()

    # TODO: add average time between questions
    # TODO: add average questions age

    def transform(self, stu, que, ans) -> pd.DataFrame:
        stu['students_state'] = stu['students_location'].apply(lambda s: str(s).split(', ')[-1])

        que['questions_body_length'] = que['questions_body'].apply(lambda s: len(str(s)))
        ans['answers_body_length'] = ans['answers_body'].apply(lambda s: len(str(s)))

        # prepare all the dataframes needed for iteration
        que_change = stu.merge(que, left_on='students_id', right_on='questions_author_id')
        ans_change = que_change.merge(ans, left_on='questions_id', right_on='answers_question_id') \
            .rename(columns={'answers_date_added': 'students_time'})

        # add new columns which will be used to determine to which change corressponds stacked DataFrame row
        ans_change['change_type'] = 'answer'
        que_change['change_type'] = 'question'
        que_change = que_change.rename(columns={'questions_date_added': 'students_time'})

        # stack two DataFrame to form resulting one for iteration
        df = pd.concat([que_change, ans_change], ignore_index=True).sort_values('students_time')

        # data is a dist with mapping from student's id to his list of features
        # each list contains dicts with mapping from feature name to its value on a particular moment
        data = {}
        avgs = {}

        for i, row in df.iterrows():
            cur_stu = row['students_id']

            # default case, student's feature values before he left any questions
            if cur_stu not in data:
                new = {'students_questions_asked': 0,
                       'students_previous_question_time': row['students_date_joined']}
                for feature in ['students_time'] + self.features['numerical']['mean']:
                    new[feature] = None
                data[cur_stu] = [new]
                avgs[cur_stu] = {feature: Averager() for feature in self.features['numerical']['mean']}

            # features on previous timestamp
            prv = data[cur_stu][-1]
            new = prv.copy()

            new['students_time'] = row['students_time']

            # if current change is new question, update question-depended features
            if row['change_type'] == 'question':
                new['students_questions_asked'] += 1
                new['students_previous_question_time'] = row['questions_date_added']
                new['students_average_question_body_length'] = row['questions_body_length']
            else:  # if new answer is added, update answer-depended features
                new['students_average_answer_body_length'] = row['answers_body_length']
                new['students_average_answer_amount'] = new['students_average_answer_amount'] + 1 \
                    if new['students_average_answer_amount'] is not None else 1

            # normalize average features
            for feature in ['students_average_question_body_length'] if row['change_type'] == 'question' else \
                    ['students_average_answer_body_length', 'students_average_answer_amount']:
                avgs[cur_stu][feature].upd(new[feature])
                new[feature] = avgs[cur_stu][feature].get()

            data[cur_stu].append(new)

        # construct a DataFrame out of dict of list of feature dicts
        df = pd.DataFrame([{**f, **{'students_id': id}} for (id, fs) in data.items() for f in fs])

        df = df.merge(stu, on='students_id')
        # launch feature pre-processing
        self.preprocess(df)

        # re-order the columns
        df = df[['students_id', 'students_time'] + self.features['all']]

        return df


class ProProc(BaseProc):
    """
    Professionals data preprocessor
    """

    def __init__(self, oblige_fit, path):
        super().__init__(oblige_fit, path)

        with open(path + 'tags_embs.pkl', 'rb') as file:
            self.tag_embs = pickle.load(file)

        with open(path + 'industries_embs.pkl', 'rb') as file:
            self.industry_embs = pickle.load(file)

        self.features = {
            'categorical': [('professionals_industry', 100), ('professionals_location', 100),
                            ('professionals_state', 40)],
            'numerical': {
                'zero': [],  # ['professionals_questions_answered'],
                'mean': ['professionals_average_question_body_length',
                         'professionals_average_answer_body_length']
            },
            'date': []  # ['professionals_date_joined', 'professionals_previous_answer_date']
        }

        self._unroll_features()

    # TODO: add average question age
    # TODO: add average time between answers

    def transform(self, pro, que, ans, tags) -> pd.DataFrame:
        tags['tags_tag_name'] = tags['tags_tag_name'].apply(lambda x: self.tp.process(x, allow_stopwords=True))

        # aggregate tags for each professional
        tags_grouped = tags.groupby('tag_users_user_id', as_index=False)[['tags_tag_name']] \
            .aggregate(lambda x: ' '.join(set(x)))

        pro['professionals_state'] = pro['professionals_location'].apply(lambda loc: str(loc).split(', ')[-1])
        pro['professionals_industry_processed'] = pro['professionals_industry'].apply(lambda x: self.tp.process(x))
        que['questions_body_length'] = que['questions_body'].apply(lambda s: len(str(s)))
        ans['answers_body_length'] = ans['answers_body'].apply(lambda s: len(str(s)))

        # prepare all the dataframes needed for iteration
        df = pro.merge(ans, left_on='professionals_id', right_on='answers_author_id') \
            .merge(que, left_on='answers_question_id', right_on='questions_id') \
            .sort_values('answers_date_added')
        data = {}

        for i, row in df.iterrows():
            cur_pro = row['professionals_id']

            # default case, professional's feature values before he left any questions
            if cur_pro not in data:
                new = {'professionals_questions_answered': 0,
                       'professionals_previous_answer_date': row['professionals_date_joined']}
                for feature in ['professionals_time', 'professionals_average_question_age',
                                'professionals_average_question_body_length',
                                'professionals_average_answer_body_length']:
                    new[feature] = None
                data[cur_pro] = [new]

            prv = data[cur_pro][-1]
            # feature update rules
            new = {'professionals_time': row['answers_date_added'],
                   'professionals_questions_answered': prv['professionals_questions_answered'] + 1,
                   'professionals_previous_answer_date': row['answers_date_added'],
                   'professionals_average_question_age':
                       (row['answers_date_added'] - row['questions_date_added']) / np.timedelta64(1, 's'),
                   'professionals_average_question_body_length': row['questions_body_length'],
                   'professionals_average_answer_body_length': row['answers_body_length']}
            length = len(data[cur_pro])
            if length != 1:
                # normalize average feature values
                for feature in ['professionals_average_question_age', 'professionals_average_question_body_length',
                                'professionals_average_answer_body_length']:
                    new[feature] = (prv[feature] * (length - 1) + new[feature]) / length
            data[cur_pro].append(new)

        # construct a dataframe out of dict of list of feature dicts
        # data is a dist with mapping from professional's id to his list of features
        # each list contains dicts with mapping from feature name to its value on a particular moment
        df = pd.DataFrame([{**f, **{'professionals_id': id}} for (id, fs) in data.items() for f in fs])

        df = df.merge(pro, on='professionals_id').merge(tags_grouped, how='left', left_on='professionals_id',
                                                        right_on='tag_users_user_id')
        # launch feature pre-processing
        self.preprocess(df)

        # prepare subscribed tag embeddings

        tag_emb_len = list(self.tag_embs.values())[0].shape[0]

        def __convert(s):
            embs = []
            for tag in str(s).split():
                if tag in self.tag_embs:
                    embs.append(self.tag_embs[tag])
            if len(embs) == 0:
                embs.append(np.zeros(tag_emb_len))
            return np.vstack(embs).mean(axis=0)

        mean_tag_embs = df['tags_tag_name'].apply(__convert)

        # prepare industry embeddings
        industry_emb_len = list(self.industry_embs.values())[0].shape[0]
        industry_embs = df['professionals_industry_processed'] \
            .apply(lambda x: self.industry_embs.get(x, np.zeros(industry_emb_len)))

        # re-order the columns
        df = df[['professionals_id', 'professionals_time'] + self.features['all']]

        # append subscribed tag embeddings
        for i in range(tag_emb_len):
            df[f'pro_tag_emb_{i}'] = mean_tag_embs.apply(lambda x: x[i])

        for i in range(industry_emb_len):
            df[f'pro_ind_emb_{i}'] = industry_embs.apply(lambda x: x[i])

        return df
