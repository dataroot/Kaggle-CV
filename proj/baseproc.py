import pickle
import os
from abc import ABC

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import TextProcessor


class BaseProc(ABC):
    """
    Class with implementation of basic preprocessors logic
    """

    def __init__(self, oblige_fit, path=''):
        """
        :param oblige_fit: whether it is necessary to fit new preprocessor even if the one exists in preprocessors.pkl
        :param path: path to all the data
        """
        self.oblige_fit = oblige_fit
        self.path = path

        self.features = {
            'categorical': [],
            'numerical': {'zero': [], 'mean': []},
            'date': []
        }

        # load preprocessors, if existent
        if os.path.isfile(self.path + 'preprocessors.pkl'):
            with open(self.path + 'preprocessors.pkl', 'rb') as file:
                self.pp = pickle.load(file)
        else:
            self.pp = {}

        self.tp = TextProcessor(path)

    def __del__(self):
        # serialize updated preprocessors
        with open(self.path + 'preprocessors.pkl', 'wb') as file:
            pickle.dump(self.pp, file)

    def _unroll_features(self):
        """
        Called once after self.features specification in constructor of child class,
        unrolls all the features in single separate list self.features['all']
        """
        self.features['all'] = [name for name, deg in self.features['categorical']] + \
                               self.features['numerical']['zero'] + self.features['numerical']['mean'] + \
                               [f + p for f in self.features['date']
                                for p in ['_time', '_doy_sin', '_doy_cos', '_dow']]

    def datetime(self, df: pd.DataFrame, feature: str, hour: bool = False):
        """
        Generates a bunch of new datetime features and drops the original feature inplace

        :param df: data to work with
        :param feature: name of a column in df that contains date
        :param hour: whether feature contains time
        """
        # iterate over suffix of generated features and function to calculate it
        for suf, fun in [('_time', lambda d: d.year + (d.dayofyear + (d.hour / 24 if hour else 0)) / 365),
                         ('_doy_sin', lambda d: np.sin(2 * np.pi * d.dayofyear / 365)),
                         ('_doy_cos', lambda d: np.cos(2 * np.pi * d.dayofyear / 365)),
                         ('_dow', lambda d: d.weekday())] + \
                        ([('_hour_sin', lambda d: np.sin(2 * np.pi * (d.hour + d.minute / 60) / 24)),
                          ('_hour_cos', lambda d: np.cos(2 * np.pi * (d.hour + d.minute / 60) / 24))]
                        if hour else []):
            df[feature + suf] = df[feature].apply(fun)
            # add created feature to the list of generated features
            self.features['gen'].append(feature + suf)

        df.drop(columns=feature, inplace=True)

    def __get_preprocessor(self, fit_data: np.array, feature: str, base):
        """
        Creates new preprocessor object of class base and fits it
        or uses existing one in self.pp and returns it

        :param fit_data: NumPy array of data to fit new preprocessor
        :param feature: feature name to search for in self.pp
        :param base: new preprocessor's class
        :returns: preprocessor object
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
        Transforms via StandardScaler, fills NaNs according to fillmode

        :param df: data to work with
        :param feature: name of a column in df that contains numerical data
        :param fillmode: method to fill NaNs, either 'mean' or 'zero'
        """
        # calculate default value and fill NaNs with it
        na = df[feature].mean() if fillmode == 'mean' else 0
        df[feature].fillna(na, inplace=True)

        # standardize feature values
        fit_data = df[feature].values.reshape(-1, 1).astype('float64')
        sc = self.__get_preprocessor(fit_data, feature, StandardScaler)
        df[feature] = sc.transform(fit_data)

    def categorical(self, df: pd.DataFrame, feature: str, n: int):
        """
        Encodes top n most popular values with different labels from 0 to n-1,
        remaining values with n and NaNs with n+1

        :param df: data to work with
        :param feature: name of a column in df that contains categorical data
        :param n: number of top by popularity values to move in separate categories.
                  0 to encode everything with different labels
        """
        vc = df[feature].value_counts()
        # number of unique values to leave
        n = len(vc) if n == 0 else n
        # unique values to leave
        top = set(vc[:n].index)
        isin_top = df[feature].isin(top)

        fit_data = df.loc[isin_top, feature]
        le = self.__get_preprocessor(fit_data, feature, LabelEncoder)

        # isin_le differs from isin_top if new preprocessor object was fitted
        isin_le = df[feature].isin(set(le.classes_))
        df.loc[isin_le, feature] = le.transform(df.loc[isin_le, feature])

        # unique values to throw away - encode with single label n
        bottom = set(vc.index) - set(le.classes_)
        isin_bottom = df[feature].isin(bottom)
        df.loc[isin_bottom, feature] = n

        df[feature].fillna(n + 1, inplace=True)

    def preprocess(self, df: pd.DataFrame):
        """
        Full preprocessing pipeline

        :param df: data to work with.
        :param verbose: whether to print some stupid log messages
        """
        # preprocess all date features
        self.features['gen'] = []
        for feature in self.features['date']:
            self.datetime(df, feature, hour=False)

        # preprocess all numerical features, including generated features from dates
        for fillmode in self.features['numerical']:
            for feature in self.features['numerical'][fillmode] + (self.features['gen'] if fillmode == 'mean' else []):
                if feature in df.columns:
                    self.numerical(df, feature, fillmode)

        # preprocess all categorical features
        for feature, n in self.features['categorical']:
            self.categorical(df, feature, n)