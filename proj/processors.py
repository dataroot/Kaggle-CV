import pandas as pd
import numpy as np

from baseproc import BaseProc
from utils import Averager


# TODO: make feature names shorter
# TODO: split transform methods

class QueProc(BaseProc):
    """
    Questions data preprocessor
    """

    def __init__(self, tag_embs, ques_d2v, lda_dic, lda_tfidf, lda_model):
        super().__init__()

        self.tag_embs = tag_embs
        self.ques_d2v = ques_d2v

        self.lda_dic = lda_dic
        self.lda_tfidf = lda_tfidf
        self.lda_model = lda_model

        self.features = {
            'numerical': {
                'zero': ['questions_body_length', 'questions_tag_count'],
                'mean': []
            }
        }

        self._unroll_features()

    def transform(self, que, tags):
        """
        Main method to calculate, preprocess question's features and append textual embeddings

        :param que: questions dataframe with preprocessed textual columns
        :param tags: merged tags and tag_questions dataframes with preprocessed textual columns
        :return: dataframe of question's id, question's date added and model-friendly question's features
        """
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

        tag_emb_len = list(self.tag_embs.values())[0].shape[0]

        def __convert(s):
            embs = []
            for tag in str(s).split():
                if tag in self.tag_embs:
                    embs.append(self.tag_embs[tag])
            if len(embs) == 0:
                embs.append(np.zeros(tag_emb_len))
            return np.vstack(embs).mean(axis=0)

        mean_embs = df['tags_tag_name'].apply(__convert)

        lda_emb_len = len(self.lda_model[[]])
        lda_corpus = [self.lda_dic.doc2bow(doc) for doc in df['questions_whole'].apply(lambda x: x.split())]
        lda_corpus = self.lda_tfidf[lda_corpus]
        lda_que_embs = self.lda_model.inference(lda_corpus)[0]

        d2v_emb_len = len(self.ques_d2v.infer_vector([]))

        def __infer_d2v(s):
            self.ques_d2v.random.seed(0)
            return self.ques_d2v.infer_vector(s.split(), steps=100)

        d2v_que_embs = df['questions_whole'].apply(__infer_d2v)

        # re-order the columns
        df = df[['questions_id', 'questions_time'] + self.features['all']]

        # append lda question embeddings
        for i in range(lda_emb_len):
            df[f'que_lda_emb_{i}'] = lda_que_embs[:, i]

        # append d2v question embeddings
        for i in range(d2v_emb_len):
            df[f'que_d2v_emb_{i}'] = d2v_que_embs.apply(lambda x: x[i])

        # append tag embeddings
        for i in range(tag_emb_len):
            df[f'que_tag_emb_{i}'] = mean_embs.apply(lambda x: x[i])

        return df


class StuProc(BaseProc):
    """
    Students data preprocessor
    """

    def __init__(self):
        super().__init__()

        self.features = {
            'categorical': [('students_location', 100), ('students_state', 40)],
            'numerical': {
                'zero': ['students_questions_asked'],
                'mean': ['students_average_question_body_length', 'students_average_answer_body_length',
                         'students_average_answer_amount']
            },
            'date': []
        }

        self._unroll_features()

    def transform(self, stu, que, ans) -> pd.DataFrame:
        """
        Main method to calculate, preprocess students's features and append textual embeddings

        :param stu: students dataframe with preprocessed textual columns
        :param que: questions dataframe with preprocessed textual columns
        :param ans: answers dataframe with preprocessed textual columns
        :return: dataframe of students's id, timestamp and model-friendly students's features after that timestamp
        """
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
        df = pd.concat([que_change, ans_change], ignore_index=True, sort=True).sort_values('students_time')

        # data is a dist with mapping from student's id to his list of features
        # each list contains dicts with mapping from feature name to its value on a particular moment
        data = {}
        avgs = {}

        for i, row in stu.iterrows():
            cur_stu = row['students_id']

            # DEFAULT CASE
            # student's feature values before he left any questions
            if cur_stu not in data:
                new = {'students_questions_asked': 0,
                       'students_previous_question_time': row['students_date_joined']}
                for feature in ['students_time'] + self.features['numerical']['mean']:
                    new[feature] = None
                data[cur_stu] = [new]
                avgs[cur_stu] = {feature: Averager() for feature in self.features['numerical']['mean']}

        for i, row in df.iterrows():
            cur_stu = row['students_id']

            # features on previous timestamp
            prv = data[cur_stu][-1]
            new = prv.copy()

            new['students_time'] = row['students_time']

            # UPDATE RULES
            # if current change is new question, update question-depended features
            if row['change_type'] == 'question':
                new['students_questions_asked'] += 1
                new['students_previous_question_time'] = row['questions_date_added']
                new['students_average_question_body_length'] = row['questions_body_length']
            # if new answer is added, update answer-depended features
            else:
                new['students_average_answer_body_length'] = row['answers_body_length']
                new['students_average_answer_amount'] = new['students_average_answer_amount'] + 1 \
                    if new['students_average_answer_amount'] is not None else 1

            # NORMALIZE AVERAGE FEATURES
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

    def __init__(self, tag_embs, ind_embs, head_d2v, ques_d2v):
        super().__init__()

        self.tag_embs = tag_embs
        self.ind_embs = ind_embs

        self.head_d2v = head_d2v
        self.ques_d2v = ques_d2v

        self.features = {
            'categorical': [('professionals_industry', 100), ('professionals_location', 100),
                            ('professionals_state', 40)],
            'numerical': {
                'zero': [],  # ['professionals_questions_answered'],
                'mean': ['professionals_average_question_body_length',
                         'professionals_average_answer_body_length']
            }
        }

        self._unroll_features()

    # TODO: add average question age
    # TODO: add average time between answers

    def transform(self, pro, que, ans, tags) -> pd.DataFrame:
        """
        Main method to calculate, preprocess students's features and append textual embeddings

        :param pro: professionals dataframe with preprocessed textual columns
        :param que: questions dataframe with preprocessed textual columns
        :param ans: answers dataframe with preprocessed textual columns
        :param tags: merged tags and tag_users dataframes with preprocessed textual columns
        :return: dataframe of professional's id, timestamp and model-friendly professional's features after that timestamp
        """
        # aggregate tags for each professional
        tags_grouped = tags.groupby('tag_users_user_id', as_index=False)[['tags_tag_name']] \
            .aggregate(lambda x: ' '.join(set(x)))

        pro['professionals_industry_raw'] = pro['professionals_industry']
        pro['professionals_state'] = pro['professionals_location'].apply(lambda loc: str(loc).split(', ')[-1])
        que['questions_body_length'] = que['questions_body'].apply(lambda s: len(str(s)))
        ans['answers_body_length'] = ans['answers_body'].apply(lambda s: len(str(s)))

        # prepare all the dataframes needed for iteration
        df = pro.merge(ans, left_on='professionals_id', right_on='answers_author_id') \
            .merge(que, left_on='answers_question_id', right_on='questions_id') \
            .sort_values('answers_date_added')

        # data is a dist with mapping from professional's id to his list of features
        # each list contains dicts with mapping from feature name to its value on a particular moment
        data = {}
        que_emb_len = len(self.ques_d2v.infer_vector([]))

        for i, row in pro.iterrows():
            cur_pro = row['professionals_id']

            # DEFAULT CASE
            # professional's feature values before he left any questions
            if cur_pro not in data:
                new = {'professionals_questions_answered': 0,
                       'professionals_previous_answer_date': row['professionals_date_joined']}
                for feature in ['professionals_time', 'professionals_average_question_age',
                                'professionals_average_question_body_length',
                                'professionals_average_answer_body_length']:
                    new[feature] = None
                new['pro_que_emb'] = np.zeros(que_emb_len)
                data[cur_pro] = [new]

        def __infer_d2v(s):
            self.ques_d2v.random.seed(0)
            return self.ques_d2v.infer_vector(s.split(), steps=100)

        for i, row in df.iterrows():
            cur_pro = row['professionals_id']

            prv = data[cur_pro][-1]
            # UPDATE RULES
            new = {'professionals_time': row['answers_date_added'],
                   'professionals_questions_answered': prv['professionals_questions_answered'] + 1,
                   'professionals_previous_answer_date': row['answers_date_added'],
                   'professionals_average_question_age':
                       (row['answers_date_added'] - row['questions_date_added']) / np.timedelta64(1, 's'),
                   'professionals_average_question_body_length': row['questions_body_length'],
                   'professionals_average_answer_body_length': row['answers_body_length'],
                   'pro_que_emb': __infer_d2v(row['questions_whole'])}
            length = len(data[cur_pro])
            if length != 1:
                # NORMALIZE AVERAGE FEATURES
                for feature in ['professionals_average_question_age', 'professionals_average_question_body_length',
                                'professionals_average_answer_body_length', 'pro_que_emb']:
                    new[feature] = (prv[feature] * (length - 1) + new[feature]) / length
            data[cur_pro].append(new)

        # construct a dataframe out of dict of list of feature dicts
        df = pd.DataFrame([{**f, **{'professionals_id': id}} for (id, fs) in data.items() for f in fs])

        df = df.merge(pro, on='professionals_id').merge(tags_grouped, how='left', left_on='professionals_id',
                                                        right_on='tag_users_user_id')
        # launch feature pre-processing
        self.preprocess(df)

        # prepare subscribed tag embeddings

        tag_emb_len = list(self.tag_embs.values())[0].shape[0]

        def __convert_tag(s):
            embs = []
            for tag in str(s).split():
                if tag in self.tag_embs:
                    embs.append(self.tag_embs[tag])
            if len(embs) == 0:
                embs.append(np.zeros(tag_emb_len))
            return np.vstack(embs).mean(axis=0)

        mean_tag_embs = df['tags_tag_name'].apply(__convert_tag)

        # prepare industry embeddings
        industry_emb_len = list(self.ind_embs.values())[0].shape[0]
        ind_embs = df['professionals_industry_raw'] \
            .apply(lambda x: self.ind_embs.get(x, np.zeros(industry_emb_len)))

        head_emb_len = len(self.head_d2v.infer_vector([]))

        def __convert_headline(s):
            self.head_d2v.random.seed(0)
            return self.head_d2v.infer_vector(s.split(), steps=100)

        head_embs = df['professionals_headline'].apply(__convert_headline)

        que_embs = df['pro_que_emb']

        # re-order the columns
        df = df[['professionals_id', 'professionals_time'] + self.features['all']]

        # append subscribed tag embeddings
        for i in range(tag_emb_len):
            df[f'pro_tag_emb_{i}'] = mean_tag_embs.apply(lambda x: x[i])

        for i in range(industry_emb_len):
            df[f'pro_ind_emb_{i}'] = ind_embs.apply(lambda x: x[i])

        for i in range(head_emb_len):
            df[f'pro_head_emb_{i}'] = head_embs.apply(lambda x: x[i])

        for i in range(que_emb_len):
            df[f'pro_que_emb_{i}'] = que_embs.apply(lambda x: x[i])

        return df
