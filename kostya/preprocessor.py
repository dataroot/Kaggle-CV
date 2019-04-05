import numpy as np
import pandas as pd

import os, json, re, pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder

from dataset_creator import DatasetCreator


class Preprocessor(DatasetCreator):
    """
    Class for qa_data, prof_data and stud_data feature preprocessing
    """
    
    def __init__(self, created=False):
        """
        Initializes DatasetCreator class and loads existing
        preprocessors that were already fit to data
        """
        # Initialize DatasetCreator
        super().__init__(created=created)
        
        # Load existing preprocessors that were already fit to data
        if os.path.isfile('preprocessors.pickle'):
            with open('preprocessors.pickle', 'rb') as file:
                self.pp = pickle.load(file)
        else:
            self.pp = {}
        
        # Load file that contains number of categories for categorical features
        with open('cat_features.json') as f:
            self.cat_features = json.load(f)
        
        # Carry out preprocessing of all datasets
        self.qa_data_preprocessing()
        self.prof_data_preprocessing()
        self.stud_data_preprocessing()
    
    
    def qa_data_preprocessing(self):
        """
        Preprocesses qa_data dataset
        """
        # Preprocess datetime and timedelta features
        Preprocessor.datetime(self.qa_data, 'questions_date_added', hour=True)
        Preprocessor.datetime(self.qa_data, 'answers_date_added', hour=True)
        Preprocessor.datetime(self.qa_data, 'professionals_prev_answer_date', hour=True)
        
        # Preprocess numerical features
        for feature in [
            'questions_date_added_time', 'questions_date_added_dow',
            'answers_date_added_time', 'answers_date_added_dow',
            'professionals_prev_answer_date_time', 'professionals_prev_answer_date_dow',
            'questions_body_length',
        ]:
            Preprocessor.numerical(self.qa_data, feature, self.pp)
    
    
    def prof_data_preprocessing(self):
        """
        Preprocesses prof_data dataset
        """
        # Preprocess datetime and timedelta features
        Preprocessor.datetime(self.prof_data, 'professionals_date_joined')
        Preprocessor.timedelta(self.prof_data, 'professionals_average_question_age')
        
        # Preprocess numerical features
        for feature in [
            'professionals_questions_answered', 'professionals_date_joined_time',
            'professionals_date_joined_dow', 'professionals_average_question_age',
            'professionals_average_question_body_length', 'professionals_average_answer_body_length',
        ]:
            Preprocessor.numerical(self.prof_data, feature, self.pp)
        
        # Will need textual representation of industry in BatchGenerator
        self.prof_data['professionals_industry_textual'] = self.prof_data['professionals_industry']
        
        # Preprocess categorical features
        Preprocessor.categorical(
            self.prof_data, 'professionals_industry',
            self.cat_features['n_cats']['prof']['professionals_industry'],
            self.pp, oblige_fit=True
        )
        Preprocessor.categorical(
            self.prof_data, 'professionals_location',
            self.cat_features['n_cats']['prof']['professionals_location'],
            self.pp, oblige_fit=True
        )
        Preprocessor.categorical(
            self.prof_data, 'professionals_state',
            self.cat_features['n_cats']['prof']['professionals_state'],
            self.pp, oblige_fit=True
        )
    
    
    def stud_data_preprocessing(self):
        """
        Preprocesses stud_data dataset
        """
        # Preprocess datetime and timedelta features
        Preprocessor.datetime(self.stud_data, 'students_date_joined')
        Preprocessor.timedelta(self.stud_data, 'students_average_question_age')
        
        # Preprocess numerical features
        for feature in [
            'students_questions_asked', 'students_date_joined_time',
            'students_date_joined_dow', 'students_average_question_age',
            'students_average_question_body_length', 'students_average_answer_body_length',
        ]:
            Preprocessor.numerical(self.stud_data, feature, self.pp)
        
        # Preprocess categorical features
        Preprocessor.categorical(
            self.stud_data, 'students_location',
            self.cat_features['n_cats']['ques']['students_location'],
            self.pp, oblige_fit=True
        )
        Preprocessor.categorical(
            self.stud_data, 'students_state',
            self.cat_features['n_cats']['ques']['students_state'],
            self.pp, oblige_fit=True
        )
    
    
    @staticmethod
    def datetime(df: pd.DataFrame, feature: str, hour: bool = False):
        """
        Generates a bunch of new datetime features and drops the original feature inplace

        :param df: Data to work with.
        :param feature: Name of a column in df that contains date.
        :param hour: Whether feature contains time.
        """
        df[feature] = pd.to_datetime(df[feature])

        df[feature + '_time'] = df[feature].apply(lambda d: d.year + (d.dayofyear + d.hour / 24) / 365)
        df[feature + '_doy_sin'] = df[feature].apply(lambda d: np.sin(2 * np.pi * d.dayofyear / 365))
        df[feature + '_doy_cos'] = df[feature].apply(lambda d: np.cos(2 * np.pi * d.dayofyear / 365))
        df[feature + '_dow'] = df[feature].apply(lambda d: d.weekday())

        if hour:
            df[feature + '_hour_sin'] = df[feature].apply(lambda d: np.sin(2 * np.pi * (d.hour + d.minute / 60) / 24))
            df[feature + '_hour_cos'] = df[feature].apply(lambda d: np.cos(2 * np.pi * (d.hour + d.minute / 60) / 24))

        df.drop(columns=feature, inplace=True)
    
    
    @staticmethod
    def timedelta(df: pd.DataFrame, feature: str):
        """
        Generates the new timedelta feature

        :param df: Data to work with.
        :param feature: Name of a column in df that contains timedelta.
        """
        df[feature] = pd.to_timedelta(df[feature])

        df[feature] = df[feature] / pd.Timedelta("1 day")
    
    
    @staticmethod
    def _get_preprocessor(fit_data: np.array, feature: str, base, pp: dict, oblige_fit: bool):
        """
        Creates new preprocessor having class base or uses existing one in preprocessors.pickle
        Returns this preprocessor

        :param fit_data: NumPy array of data to fit new preprocessor.
        :param feature: Feature name to search for in preprocessors.pickle.
        :param base: Preprocessor's class.
        :param pp: Object with preprocessors.
        :param oblige_fit: Whether to fit new preprocessor on feature even if there already exists one.
        :returns: Preprocessor object.
        """    
        if feature in pp and not oblige_fit:
            preproc = pp[feature]
        else:
            preproc = base()
            preproc.fit(fit_data)
            pp[feature] = preproc
            with open('preprocessors.pickle', 'wb') as file:
                pickle.dump(pp, file)
        return preproc
    
    
    @staticmethod
    def numerical(df: pd.DataFrame, feature: str, pp: dict, oblige_fit: bool = False):
        """
        Transforms via StandardScaler

        :param df: Data to work with.
        :param feature: Name of a column in df that contains numerical data.
        :param pp: Object with preprocessors.
        :param oblige_fit: Whether to fit new StandardScaler on feature even if there already exists one.
        """
        fit_data = df[feature].values.reshape(-1, 1).astype('float64')
        sc = Preprocessor._get_preprocessor(fit_data, feature, StandardScaler, pp, oblige_fit)
        df[feature] = sc.transform(fit_data)
    
    
    @staticmethod
    def categorical(df: pd.DataFrame, feature: str, n: int, pp: dict, oblige_fit: bool = False):
        """
        Encodes top n most popular values with different labels from 0 to n-1,
        remaining values with n and NaNs with n+1

        :param df: Data to work with.
        :param feature: Name of a column in df that contains categorical data.
        :param n: Number of top by popularity values to move in separate categories.
                  0 to encode everything with different labels.
        :param pp: Object with preprocessors.
        :param oblige_fit: Whether to fit new LabelEncoder on feature even if there already exists one.
        """
        vc = df[feature].value_counts()
        n = len(vc) if n == 0 else n

        top = set(vc[:n].index)
        isin_top = df[feature].isin(top)

        fit_data = df.loc[isin_top, feature]
        le = Preprocessor._get_preprocessor(fit_data, feature, LabelEncoder, pp, oblige_fit)

        isin_le = df[feature].isin(set(le.classes_))
        df.loc[isin_le, feature] = le.transform(df.loc[isin_le, feature])

        bottom = set(vc.index) - set(le.classes_)
        isin_bottom = df[feature].isin(bottom)
        df.loc[isin_bottom, feature] = n
        df[feature].fillna(n + 1, inplace=True)

