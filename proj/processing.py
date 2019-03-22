import pickle
import os
from abc import ABC

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder


class Base(ABC):
    def __init__(self, path='', oblige_fit=False):
        self.path = path
        self.oblige_fit = oblige_fit

        self.features = None
        self.embs = None

        if os.path.isfile(self.path + 'preprocessors.pickle'):
            with open(self.path + 'preprocessors.pickle', 'rb') as file:
                self.pp = pickle.load(file)
        else:
            self.pp = {}

    def __del__(self):
        with open(self.path + 'preprocessors.pickle', 'wb') as file:
            pickle.dump(self.pp, file)

    def datetime(self, df: pd.DataFrame, feature: str, hour: bool = False):
        """
        Generates a bunch of new datetime features and drops the original feature inplace

        :param df: Data to work with.
        :param feature: Name of a column in df that contains date.
        :param hour: Whether feature contains time.
        """
        df[feature] = pd.to_datetime(df[feature])

        for suf, fun in [('_time', lambda d: d.year + d.dayofyear / 365),
                         ('_doy_sin', lambda d: np.sin(2 * np.pi * d.dayofyear / 365)),
                         ('_doy_cos', lambda d: np.cos(2 * np.pi * d.dayofyear / 365)),
                         ('_dow', lambda d: d.weekday())] + \
                        [] if not hour else \
                [('_hour_sin', lambda d: np.sin(2 * np.pi * (d.hour + d.minute / 60) / 24)),
                 ('_hour_cos', lambda d: np.cos(2 * np.pi * (d.hour + d.minute / 60) / 24))]:
            df[feature + suf] = df[feature].apply(fun)
            self.features['date']['gen'].append(feature + suf)

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

    def numerical(self, df: pd.DataFrame, feature: str):
        """
        Transforms via StandardScaler

        :param df: Data to work with.
        :param feature: Name of a column in df that contains numerical data.
        """
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

    def preprocess(self, df: pd.DataFrame, verbose=True) -> pd.DataFrame:
        """
        Full preprocessing pipeline

        :param df: Data to work with.
        :param verbose: Whether to print some log messages.
        :return: Modified data.
        """
        self.features['date']['gen'] = []
        for mode in self.features['date']:
            for feature in self.features['date'][mode]:
                self.datetime(df, feature, hour=(mode == 'time'))
                if verbose:
                    print(feature)
        self.features['numerical']['mean'] += self.features['date']['gen']

        for fillmode in self.features['numerical']:
            for feature in self.features['numerical'][fillmode]:
                if feature in df.columns:
                    self.numerical(df, feature)
                    if verbose:
                        print(feature)

        for feature, n in self.features['categorical']:
            self.categorical(df, feature, n)
            if verbose:
                print(feature)

        cat_names = [feature for feature, n in self.features['categorical']]
        non_cat_col = [col for col in df.columns if col not in cat_names]

        return df[cat_names + non_cat_col]


class Questions(Base):
    def __init__(self, path, oblige_fit):
        super().__init__(path, oblige_fit)

        with open('tags_embs.pickle', 'rb') as file:
            self.embs = pickle.load(file)

        self.features = {
            # TODO
        }

    def transform(self):
        # TODO
        pass


class Professional(Base):
    pass
