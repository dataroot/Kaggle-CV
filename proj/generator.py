import random
import pickle

import keras
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# TODO: consider questions without answers

class BatchGenerator(keras.utils.Sequence):
    """
    Class to ingest data from pre-processed DataFrames to model
    in form of batches of NumPy arrays
    """

    def __init__(self, que: pd.DataFrame, stu: pd.DataFrame, pro: pd.DataFrame,
                 batch_size: int, pos_pairs: list, nonneg_pairs: list, ss: StandardScaler, pro_dates: dict):
        """
        :param que: pre-processed questions data
        :param stu: pre-processed students data
        :param pro: pre-processed professionals data
        :param batch_size: actually, half of the real batch size
        Number of both positive and negative pairs present in generated batch
        :param pos_pairs: tuples of question, student and professional, which form positive pair
        (professional answered on the given question from corresponding student)
        :param nonneg_pairs: tuples of question, student and professional, which are known to form a positive pair.
        Superset of pos_pairs, used in sampling of negative pairs
        :param ss: trained preprocessor of current_time feature
        :param pro_dates: mappings from professional's id to his registration date
        """
        self.batch_size = batch_size

        # extract mappings from question's id to question's date and features
        que_ar = que.values
        self.que_feat = {que_ar[i, 0]: que_ar[i, 2:] for i in range(que_ar.shape[0])}
        self.que_time = {que_ar[i, 0]: pd.Timestamp(que_ar[i, 1]) for i in range(que_ar.shape[0])}

        self.pos_pairs = pos_pairs
        self.on_epoch_end()  # shuffle pos_pairs
        self.nonneg_pairs = {(que, stu, pro) for que, stu, pro, time in nonneg_pairs}

        # these lists are used in sampling of negative pairs
        self.ques_stus_times = [(que, stu, self.que_time[que]) for que, stu, pro, time in pos_pairs]

        self.pros = np.array([pro for que, stu, pro, time in nonneg_pairs])
        self.pros_times = np.array([pro_dates[pro] for que, stu, pro, time in nonneg_pairs])

        # simultaneously sort two arrays containing professional features
        sorted_args = np.argsort(self.pros_times)
        self.pros = self.pros[sorted_args]
        self.pros_times = self.pros_times[sorted_args]

        # extract mappings from student's id to student's date and features
        self.stu_feat = {}
        self.stu_time = {}
        for stu_id, group in stu.groupby('students_id'):
            group_ar = group.values[:, 1:]
            self.stu_feat[stu_id] = np.array([group_ar[i, 1:] for i in range(group_ar.shape[0])])
            self.stu_time[stu_id] = np.array([group_ar[i, 0] for i in range(group_ar.shape[0])])

        # extract mappings from professional's id to professional's date and features
        self.pro_feat = {}
        self.pro_time = {}
        for pro_id, group in pro.groupby('professionals_id'):
            group_ar = group.values[:, 1:]
            self.pro_feat[pro_id] = np.array([group_ar[i, 1:] for i in range(group_ar.shape[0])])
            self.pro_time[pro_id] = np.array([group_ar[i, 0] for i in range(group_ar.shape[0])])

        self.ss = ss

    def __len__(self):
        return len(self.pos_pairs) // self.batch_size

    @staticmethod
    def __find(feat_ar: np.ndarray, time_ar: np.ndarray, search_time):
        pos = np.searchsorted(time_ar[1:], search_time)
        assert time_ar[pos] is pd.NaT or time_ar[pos] < search_time
        return feat_ar[pos]

    def __convert(self, pairs: list) -> (np.ndarray, np.ndarray):
        """
        Convert list of pairs of ids to NumPy arrays
        of question and professionals features
        """
        x_que, x_pro, current_times = [], [], []
        for que, stu, pro, current_time in pairs:
            que_data = self.que_feat[que]

            # find student's and professional's feature at current time
            stu_data = BatchGenerator.__find(self.stu_feat[stu], self.stu_time[stu], current_time)
            pro_data = BatchGenerator.__find(self.pro_feat[pro], self.pro_time[pro], current_time)

            # prepare current time as feature itself
            current_time = current_time.year + (current_time.dayofyear + current_time.hour / 24) / 365
            current_times.append(current_time)

            x_que.append(np.hstack([stu_data, que_data]))
            x_pro.append(pro_data)

        # preprocess current times features
        current_times = self.ss.transform(np.array(current_times).reshape(-1, 1))
        # and append them to both questions and professionals
        return np.vstack(x_que), np.vstack(x_pro)
        # return np.hstack([np.vstack(x_que), current_times]), np.vstack(x_pro)

    def __getitem__(self, index):
        """
        Generate the batch
        """
        pos_pairs = self.pos_pairs[self.batch_size * index: self.batch_size * (index + 1)]
        neg_pairs = []

        for i in range(len(pos_pairs)):
            while True:
                # sample question, its student and time
                que, stu, zero = random.choice(self.ques_stus_times)
                # calculate shift between question's and current time
                while True:
                    shift = np.random.exponential(50) - 35
                    if shift > 0:
                        break
                current_time = zero + pd.Timedelta(int(shift * 24 * 60), 'm')
                # find number of professionals with registration date before current time
                i = np.searchsorted(self.pros_times, current_time)
                if i != 0:
                    break

            while True:
                # sample professional for negative pair
                pro = random.choice(self.pros[:i])
                # check if he doesn't form a positive pair
                if (que, stu, pro) not in self.nonneg_pairs:
                    neg_pairs.append((que, stu, pro, current_time))
                    break

        # convert lists of pairs to NumPy arrays of features
        x_pos_que, x_pos_pro = self.__convert(pos_pairs)
        x_neg_que, x_neg_pro = self.__convert(neg_pairs)

        # return the data in its final form
        return [np.vstack([x_pos_que, x_neg_que]), np.vstack([x_pos_pro, x_neg_pro])], \
               np.vstack([np.ones((len(x_pos_que), 1)), np.zeros((len(x_neg_que), 1))])

    def on_epoch_end(self):
        # shuffle positive pairs
        self.pos_pairs = random.sample(self.pos_pairs, len(self.pos_pairs))
