import pickle

import pandas as pd
import numpy as np

from baseproc import BaseProc


# TODO: add regular expressions somewhere

class QueProc(BaseProc):
    """
    Questions data preprocessor
    """

    def __init__(self, oblige_fit, path=''):
        super().__init__(oblige_fit, path)

        with open('tags_embs.pkl', 'rb') as file:
            self.embs = pickle.load(file)

        self.features = {
            'categorical': [],
            'numerical': {
                'zero': ['questions_body_length'],
                'mean': []
            },
            'date': ['questions_date_added']
        }
        self._unroll_features()

    # TODO: add number of tags feature

    def transform(self, que, tags):
        tags['tags_tag_name'] = tags['tags_tag_name'].apply(lambda x: self.tp.process(x, allow_stopwords=True))

        que['questions_date_added'] = pd.to_datetime(que['questions_date_added'])
        que['questions_body_length'] = que['questions_body'].apply(lambda s: len(str(s)))

        tags_grouped = tags.groupby('tag_questions_question_id', as_index=False)[['tags_tag_name']] \
            .aggregate(lambda x: ' '.join(x))
        df = que.merge(tags_grouped, how='left', left_on='questions_id', right_on='tag_questions_question_id')

        # launch feature pre-processing

        self.preprocess(df)

        # prepare tag embeddings

        emb_len = list(self.embs.values())[0].shape[0]

        def __convert(s):
            embs = []
            for tag in str(s).split():
                if tag in self.embs:
                    embs.append(self.embs[tag])
            if len(embs) == 0:
                embs.append(np.zeros(emb_len))
            return np.vstack(embs).mean(axis=0)

        mean_embs = df['tags_tag_name'].apply(__convert)

        # re-order the columns
        df = df[['questions_id', 'answers_author_id'] + self.features['all']]

        # append tag embeddings
        for i in range(emb_len):
            df[f'que_emb_{i}'] = mean_embs.apply(lambda x: x[i])

        return df


class StuProc(BaseProc):
    """
    Students data preprocessor
    """

    def __init__(self, oblige_fit, path=''):
        super().__init__(oblige_fit, path)

        self.features = {
            'categorical': [('students_location', 100), ('students_state', 40)],
            'numerical': {
                'zero': ['students_questions_asked', 'students_time_since_last_question'],
                'mean': ['students_average_question_age', 'students_average_question_body_length',
                         'students_average_answer_body_length', 'students_average_answer_amount']
            },
            'date': ['students_date_joined']
        }

    def transform(self, que, ans, stu) -> pd.DataFrame:
        stu['students_state'] = stu['students_location'].apply(lambda s: str(s).split(', ')[-1])

        for df, feature in [(stu, 'students_date_joined'),
                            (que, 'questions_date_added'), (ans, 'answers_date_added')]:
            df[feature] = pd.to_datetime(df[feature])

        ans_grouped = ans.groupby('answers_question_id')
        df = stu.merge(que, left_on='students_id', right_on='questions_author_id') \
            .sort_values('answers_date_added')
        data = {}

        for i, row in df.iterrows():
            cur_stu = row['students_id']

            if cur_stu not in data:
                data[cur_stu] = {'questions': [cur_stu['questions_id']],
                                 'students_questions_asked': [0],
                                 'students_time_since_last_question':
                                     [row['questions_date_added'] - row['students_date_joined']]}
            # under construction


class ProProc(BaseProc):
    """
    Professionals data preprocessor
    """

    def __init__(self, oblige_fit, path=''):
        super().__init__(oblige_fit, path)

        with open('industries_embs.pkl', 'rb') as file:
            self.embs = pickle.load(file)
        '''
        self.features = {
            'categorical': [('professionals_industry', 100), ('professionals_location', 100),
                            ('professionals_state', 40)],
            'numerical': {
                'zero': ['professionals_questions_answered'],
                'mean': ['professionals_average_question_age', 'professionals_average_question_body_length',
                         'professionals_average_answer_body_length']
            },
            'date': ['professionals_date_joined']
        }
        '''
        self._unroll_features()

    # TODO: add email activated feature
    # TODO: add previous answer date feature

    # TODO: make the same changes as in QueProc and StuProc

    def transform(self, pro, que, ans) -> pd.DataFrame:
        pro['professionals_industry_raw'] = pro['professionals_industry'].apply(lambda x: self.tp.process(x))

        for df, feature in [(pro, 'professionals_date_joined'),
                            (que, 'questions_date_added'), (ans, 'answers_date_added')]:
            df[feature] = pd.to_datetime(df[feature])

        # Count the number of answered questions by each professional
        number_answered = ans[['answers_author_id', 'answers_question_id']].groupby('answers_author_id').count() \
            .rename(columns={'answers_question_id': 'professionals_questions_answered'})

        # Add professionals_questions_answered feature to prof_data
        df = pro.merge(number_answered, how='left', left_on='professionals_id', right_index=True)

        # Extract state or country from location
        df['professionals_state'] = df['professionals_location'].apply(lambda loc: str(loc).split(', ')[-1])

        # Get average question age for every professional among questions he answered

        que['questions_body_length'] = que['questions_body'].apply(lambda s: len(str(s)))
        ans['answers_body_length'] = ans['answers_body'].apply(lambda s: len(str(s)))

        ans_date = ans[['answers_question_id', 'answers_date_added']].groupby('answers_question_id').min() \
            .rename(columns={'answers_date_added': 'questions_first_answer_date_added'})

        que = que.merge(ans_date, left_on='questions_id', right_on='answers_question_id')
        que['questions_age'] = que['questions_first_answer_date_added'] - que['questions_date_added']

        que_ans = que.merge(ans, how='left', left_on='questions_id', right_on='answers_question_id')

        average_age_length = que_ans[['answers_author_id',
                                      'questions_age', 'questions_body_length', 'answers_body_length']] \
            .groupby('answers_author_id').mean(numeric_only=False) \
            .rename(columns={'questions_age': 'professionals_average_question_age',
                             'questions_body_length': 'professionals_average_question_body_length',
                             'answers_body_length': 'professionals_average_answer_body_length'})

        # Add professionals_average_question_age feature to prof_data
        df = df.merge(average_age_length, how='left', left_on='professionals_id', right_index=True)

        # list of links to questions
        pro_que = ans[['answers_question_id', 'answers_author_id']].groupby('answers_author_id') \
            .aggregate(lambda x: ' '.join(x))
        df = df.merge(pro_que, how='left', left_on='professionals_id', right_on='answers_author_id')

        self.preprocess(df)

        emb_len = list(self.embs.values())[0].shape[0]
        embs = df['professionals_industry_raw'].apply(lambda x: self.embs.get(x, np.zeros(emb_len)))

        df = df[['professionals_id', 'answers_question_id'] + self.features['all']]

        # append averaged tag embeddings
        for i in range(emb_len):
            df[f'pro_emb_{i}'] = embs.apply(lambda x: x[i])

        return df
