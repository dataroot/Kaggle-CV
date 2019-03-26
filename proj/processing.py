import pickle
import os
from abc import ABC

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import TextProcessor


class Base(ABC):
    def __init__(self, oblige_fit, path=''):
        self.oblige_fit = oblige_fit
        self.path = path

        self.features = None
        self.embs = None

        if os.path.isfile(self.path + 'preprocessors.pkl'):
            with open(self.path + 'preprocessors.pkl', 'rb') as file:
                self.pp = pickle.load(file)
        else:
            self.pp = {}

        self.tp = TextProcessor(path)

    def __del__(self):
        with open(self.path + 'preprocessors.pkl', 'wb') as file:
            pickle.dump(self.pp, file)

    def _unroll_features(self):
        self.features['all'] = [name for name, deg in self.features['categorical']] + \
                               self.features['numerical']['zero'] + self.features['numerical']['mean'] + \
                               [f + p for f in self.features['date']
                                for p in ['_time', '_doy_sin', '_doy_cos', '_dow', '_hour_sin', '_hour_cos']]

    def datetime(self, df: pd.DataFrame, feature: str, hour: bool = False):
        """
        Generates a bunch of new datetime features and drops the original feature inplace

        :param df: Data to work with.
        :param feature: Name of a column in df that contains date.
        :param hour: Whether feature contains time.
        """
        for suf, fun in [('_time', lambda d: d.year + (d.dayofyear + (d.hour / 24 if hour else 0)) / 365),
                         ('_doy_sin', lambda d: np.sin(2 * np.pi * d.dayofyear / 365)),
                         ('_doy_cos', lambda d: np.cos(2 * np.pi * d.dayofyear / 365)),
                         ('_dow', lambda d: d.weekday())] + \
                        ([('_hour_sin', lambda d: np.sin(2 * np.pi * (d.hour + d.minute / 60) / 24)),
                          ('_hour_cos', lambda d: np.cos(2 * np.pi * (d.hour + d.minute / 60) / 24))]
                        if hour else []):
            df[feature + suf] = df[feature].apply(fun)
            self.features['gen'].append(feature + suf)

        df.drop(columns=feature, inplace=True)

    def timedelta(self, df: pd.DataFrame, feature: str):
        """
        Generates the new timedelta feature

        :param df: Data to work with.
        :param feature: Name of a column in df that contains timedelta.
        """
        df[feature] = pd.to_timedelta(df[feature])
        df[feature] = df[feature] / pd.Timedelta("1 day")

    def __get_preprocessor(self, fit_data: np.array, feature: str, base):
        """
        Creates new preprocessor having class base or uses existing one in preprocessors.pickle
        Returns this preprocessor

        :param fit_data: NumPy array of data to fit new preprocessor.
        :param feature: Feature name to search for in preprocessors.pickle.
        :param base: Preprocessor's class.
        :returns: Preprocessor object.
        """
        if feature in self.pp and not self.oblige_fit:
            preproc = self.pp[feature]
        else:
            preproc = base()
            preproc.fit(fit_data)
            self.pp[feature] = preproc
        return preproc

    def numerical(self, df: pd.DataFrame, feature: str, fillmode: str):
        """
        Transforms via StandardScaler

        :param df: Data to work with.
        :param feature: Name of a column in df that contains numerical data.
        :param fillmode: Method to fill NaNs, either 'mean' or 'zero'.
        """
        na = df[feature].mean() if fillmode == 'mean' else 0
        df[feature].fillna(na, inplace=True)

        fit_data = df[feature].values.reshape(-1, 1).astype('float64')
        sc = self.__get_preprocessor(fit_data, feature, StandardScaler)
        df[feature] = sc.transform(fit_data)

    def categorical(self, df: pd.DataFrame, feature: str, n: int):
        """
        Encodes top n most popular values with different labels from 0 to n-1,
        remaining values with n and NaNs with n+1

        :param df: Data to work with.
        :param feature: Name of a column in df that contains categorical data.
        :param n: Number of top by popularity values to move in separate categories.
                  0 to encode everything with different labels.
        """
        vc = df[feature].value_counts()
        n = len(vc) if n == 0 else n

        top = set(vc[:n].index)
        isin_top = df[feature].isin(top)

        fit_data = df.loc[isin_top, feature]
        le = self.__get_preprocessor(fit_data, feature, LabelEncoder)

        isin_le = df[feature].isin(set(le.classes_))
        df.loc[isin_le, feature] = le.transform(df.loc[isin_le, feature])

        bottom = set(vc.index) - set(le.classes_)
        isin_bottom = df[feature].isin(bottom)
        df.loc[isin_bottom, feature] = n
        df[feature].fillna(n + 1, inplace=True)

    def preprocess(self, df: pd.DataFrame, verbose=True):
        """
        Full preprocessing pipeline

        :param df: Data to work with.
        :param verbose: Whether to print some log messages.
        """
        self.features['gen'] = []
        for feature in self.features['date']:
            self.datetime(df, feature, hour=True)
            if verbose:
                print(feature)

        for fillmode in self.features['numerical']:
            for feature in self.features['numerical'][fillmode] + (self.features['gen'] if fillmode == 'mean' else []):
                if feature in df.columns:
                    self.numerical(df, feature, fillmode)
                    if verbose:
                        print(feature)

        for feature, n in self.features['categorical']:
            self.categorical(df, feature, n)
            if verbose:
                print(feature)


# TODO: add regular expressions somewhere


class Questions(Base):
    def __init__(self, oblige_fit, path=''):
        super().__init__(oblige_fit, path)

        with open('tags_embs.pkl', 'rb') as file:
            self.embs = pickle.load(file)

        self.features = {
            'categorical': [('students_location', 100), ('students_state', 40)],
            'numerical': {
                'zero': ['students_questions_asked', 'questions_body_length'],
                'mean': ['students_average_question_age', 'students_average_question_body_length',
                         'students_average_answer_body_length']
            },
            'date': ['students_date_joined', 'questions_date_added']
        }
        self._unroll_features()

    def transform(self, que, ans, stu, tags, verbose) -> pd.DataFrame:
        tags['tags_tag_name'] = tags['tags_tag_name'].apply(lambda x: self.tp.process(x, allow_stopwords=True))

        # Group tags and merge them with students and questions
        tags_grouped = tags[['tag_questions_question_id', 'tags_tag_name']] \
            .groupby('tag_questions_question_id', as_index=False).aggregate(lambda x: ' '.join(x))
        que_tags = que.merge(tags_grouped, how='left', left_on='questions_id', right_on='tag_questions_question_id')
        df = que_tags.merge(stu, left_on='questions_author_id', right_on='students_id')

        # Transform dates from string representation to datetime object
        for date in self.features['date']:
            df[date] = pd.to_datetime(df[date])
        ans['answers_date_added'] = pd.to_datetime(ans['answers_date_added'])

        # Add questions_age feature, which represents amount of time
        # from question emergence to a particular answer to that question
        ans_date = ans[['answers_question_id', 'answers_date_added']].groupby('answers_question_id').min() \
            .rename(columns={'answers_date_added': 'questions_first_answer_date_added'})

        df = df.merge(ans_date, how='left', left_on='questions_id', right_index=True)

        df['questions_age'] = df['questions_first_answer_date_added'] - df['questions_date_added']

        # average answers_body_length by student
        stu_ans = df.merge(ans, left_on='questions_id', right_on='answers_question_id')[['students_id', 'answers_body']]
        stu_ans['answers_body'] = stu_ans['answers_body'].apply(lambda s: len(str(s)))
        stu_ans = stu_ans.groupby('students_id').mean() \
            .rename(columns={'answers_body': 'students_average_answer_body_length'})
        df = df.merge(stu_ans, how='left', on='students_id')

        # list of links to professionals
        que_pro = ans[['answers_question_id', 'answers_author_id']].groupby('answers_question_id') \
            .aggregate(lambda x: ' '.join(x))
        df = df.merge(que_pro, how='left', left_on='questions_id', right_on='answers_question_id')

        # Count the number of words in question and answer body and add two new features
        df['questions_body_length'] = df['questions_body'].apply(lambda s: len(str(s)))

        # Extract state or country from location
        df['students_state'] = df['students_location'].apply(lambda s: str(s).split(', ')[-1])

        # Count the number of asked questions by each student
        stu_num = df[['students_id', 'questions_id']].groupby('students_id').count() \
            .rename(columns={'questions_id': 'students_questions_asked'})
        # Add students_questions_answered feature to stud_data
        df = df.merge(stu_num, how='left', left_on='students_id', right_index=True)

        # Get average question age for every student among questions he asked that were answered
        average_question_age = df[['students_id', 'questions_age']].groupby('students_id').mean(numeric_only=False) \
            .rename(columns={'questions_age': 'students_average_question_age'})
        # Add professionals_average_question_age feature to prof_data
        df = df.merge(average_question_age, how='left', on='students_id')

        # Compute average question and answer body length for each student
        average_question_body_length = df.groupby('students_id')[['questions_body_length']].mean() \
            .rename(columns={'questions_body_length': 'students_average_question_body_length'})
        # Add average question and answer body length features to stud_data
        df = df.merge(average_question_body_length, left_on='students_id', right_index=True)

        self.preprocess(df, verbose)

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

        df = df[['answers_author_id'] + self.features['all']]

        # append averaged tag embeddings
        for i in range(emb_len):
            df[f'que_emb_{i}'] = mean_embs.apply(lambda x: x[i])

        return df


class Professionals(Base):
    pass
