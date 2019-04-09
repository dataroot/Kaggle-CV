import random

import keras
import numpy as np
import pandas as pd


# TODO: add current time feature

class BatchGenerator(keras.utils.Sequence):
    """
    Class to ingest data from pre-processed DataFrames to model
    in form of batches of NumPy arrays
    """

    def __init__(self, que: pd.DataFrame, stu: pd.DataFrame, pro: pd.DataFrame,
                 batch_size: int, pos_pairs: list, nonneg_pairs: list):
        """
        :param que: pre-processed questions data
        :param pro: pre-processed professionals data
        :param batch_size: actually, half of the real batch size
        Number of both positive and negative pairs present in generated batch
        """
        self.batch_size = batch_size
        que_ar = que.values
        self.que_feat = {que_ar[i, 0]: que_ar[i, 2:] for i in range(que_ar.shape[0])}
        self.que_time = {que_ar[i, 0]: que_ar[i, 1] for i in range(que_ar.shape[0])}

        self.pos_pairs = [(que, stu, pro, self.que_time[que]) for que, stu, pro in pos_pairs]
        self.pos_pairs = random.sample(self.pos_pairs, len(self.pos_pairs))
        self.nonneg_pairs = set(nonneg_pairs)

        self.ques_stus_times = [(que, stu, time) for que, stu, pro, time in self.pos_pairs]
        self.pros = [pro for que, stu, pro, time in self.pos_pairs]

        self.stu_feat = {}
        self.stu_time = {}
        for stu_id, group in stu.groupby('students_id'):
            group_ar = group.values[:, 1:]
            self.stu_feat[stu_id] = np.array([group_ar[i, 1:] for i in range(group_ar.shape[0])])
            self.stu_time[stu_id] = np.array([group_ar[i, 0] for i in range(group_ar.shape[0])])

        self.pro_feat = {}
        self.pro_time = {}
        for pro_id, group in pro.groupby('professionals_id'):
            group_ar = group.values[:, 1:]
            self.pro_feat[pro_id] = np.array([group_ar[i, 1:] for i in range(group_ar.shape[0])])
            self.pro_time[pro_id] = np.array([group_ar[i, 0] for i in range(group_ar.shape[0])])

    def __len__(self):
        return len(self.pos_pairs) // self.batch_size

    @staticmethod
    def __find(feat_ar: np.ndarray, time_ar: np.ndarray, search_time):
        pos = np.searchsorted(time_ar[1:], search_time) - 1
        return feat_ar[pos]

    def __convert(self, pairs: list) -> (np.ndarray, np.ndarray):
        """
        Convert list of pairs of ids to NumPy arrays
        of question and professionals features
        """
        x_que, x_pro = [], []
        for que, stu, pro, time in pairs:
            que_data = self.que_feat[que]

            stu_data = BatchGenerator.__find(self.stu_feat[stu], self.stu_time[stu], time)
            pro_data = BatchGenerator.__find(self.pro_feat[pro], self.pro_time[pro], time)

            x_que.append(np.hstack([stu_data, que_data]))
            x_pro.append(pro_data)
        return np.vstack(x_que), np.vstack(x_pro)

    def __getitem__(self, index):
        """
        Generate the batch
        """
        pos_pairs = self.pos_pairs[self.batch_size * index: self.batch_size * (index + 1)]

        neg_pairs = []
        neg_times = []

        for i in range(len(pos_pairs)):
            que, stu, zero = random.choice(self.ques_stus_times)
            while True:
                pro = random.choice(self.pros)
                if (que, stu, pro) not in self.nonneg_pairs:
                    while True:
                        shift = np.random.exponential(50) - 35
                        if shift > 0:
                            break
                    time = np.datetime64(zero) + np.timedelta64(int(shift * 24 * 60), 'm')
                    neg_pairs.append((que, stu, pro, time))
                    break

        x_pos_que, x_pos_pro = self.__convert(pos_pairs)
        x_neg_que, x_neg_pro = self.__convert(neg_pairs)

        return [np.vstack([x_pos_que, x_neg_que]), np.vstack([x_pos_pro, x_neg_pro])], \
               np.vstack([np.ones((len(x_pos_que), 1)), np.zeros((len(x_neg_que), 1))])

    def on_epoch_end(self):
        self.pos_pairs = random.sample(self.pos_pairs, len(self.pos_pairs))
