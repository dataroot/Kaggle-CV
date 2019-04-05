import random

import keras
import numpy as np
import pandas as pd


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

        self.pos_pairs = pos_pairs
        self.nonneg_pairs = set(nonneg_pairs)

        self.ques_stus = [(que, stu) for que, stu, pro in self.pos_pairs]
        self.pros = [pro for que, stu, pro in self.pos_pairs]

        que_ar = que.values
        self.que_feat = {que_ar[i, 0]: que_ar[i, 2:] for i in range(que_ar.shape[0])}
        self.que_time = {que_ar[i, 0]: que_ar[i, 1] for i in range(que_ar.shape[0])}

        self.pro_feat = {}
        for pro_id, group in pro.groupby('professionals_id').groups():
            group_ar = group.values()
            self.pro_feat[pro_id] = [(group_ar[i, 0], group_ar[i, 1:]) for i in range(group_ar.shape[0])]

        self.stu_feat = {}
        for stu_id, group in stu.groupby('students_id').groups():
            group_ar = group.values()
            self.stu_feat[stu_id] = [(group_ar[i, 0], group_ar[i, 1:]) for i in range(group_ar.shape[0])]

    def __len__(self):
        return len(self.pos_pairs) // self.batch_size

    def __convert(self, pairs: list, times: list) -> (np.ndarray, np.ndarray):
        """
        Convert list of pairs of ids to NumPy arrays
        of question and professionals features
        """
        x_que, x_pro = [], []
        for i, (que, pro) in enumerate(pairs):
            x_que.append(self.que_feat[que])
            x_pro.append(self.pro_feat[pro])
        return np.vstack(x_que), np.vstack(x_pro)

    def __getitem__(self, index):
        """
        Generate the batch
        """
        pos_pairs = self.pos_pairs[self.batch_size * index: self.batch_size * (index + 1)]
        pos_times = [self.que_time[que] for que, stu, pro in self.pos_pairs]

        neg_pairs = []
        neg_times = []

        for i in range(len(pos_pairs)):
            que, stu = random.choice(self.ques_stus)
            while True:
                pro = random.choice(self.pros)
                if (que, stu, pro) not in self.nonneg_pairs:
                    neg_pairs.append((que, stu, pro))

                    zero = self.que_time[que]
                    while True:
                        shift = np.random.exponential(50) - 35
                        if shift > 0:
                            break
                    neg_times.append(zero + np.timedelta64(shift, 'D'))
                    break

        x_pos_que, x_pos_pro = self.__convert(pos_pairs, pos_times)
        x_neg_que, x_neg_pro = self.__convert(neg_pairs, neg_times)

        return [np.vstack([x_pos_que, x_neg_que]), np.vstack([x_pos_pro, x_neg_pro])], \
               np.vstack([np.ones((len(x_pos_que), 1)), np.zeros((len(x_neg_que), 1))])

    def on_epoch_end(self):
        self.pos_pairs = random.sample(self.pos_pairs, len(self.pos_pairs))
