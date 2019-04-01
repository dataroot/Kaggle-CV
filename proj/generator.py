import random

import keras
import numpy as np
import pandas as pd


class BatchGenerator(keras.utils.Sequence):
    """
    Class to ingest data from pre-processed DataFrames to model
    in form of batches of NumPy arrays
    """

    def __init__(self, que: pd.DataFrame, pro: pd.DataFrame, batch_size):
        """
        :param que: pre-processed questions data
        :param pro: pre-processed professionals data
        :param batch_size: actually, half of the real batch size
        Number of both positive and negative pairs present in generated batch
        """
        self.batch_size = batch_size

        # lists of question and professionals ids
        self.ques = list(que['questions_id'])
        self.pros = list(pro['professionals_id'])

        # sets of question-professional pairs achieved from question and professionals data
        que_pairs = {(row['questions_id'], author) for i, row in que.iterrows()
                     for author in str(row['answers_author_id']).split()}
        pro_pairs = {(question, row['professionals_id']) for i, row in pro.iterrows()
                     for question in str(row['answers_question_id']).split()}

        # actual set of pairs is the intersection
        self.pairs_set = que_pairs & pro_pairs
        self.pairs_list = list(self.pairs_set)

        # construct dicts mapping from entity id to its features
        que_ar = que.values
        self.que_feat = {que_ar[i, 0]: que_ar[i, 2:] for i in range(que_ar.shape[0])}
        pro_ar = pro.values
        self.pro_feat = {pro_ar[i, 0]: pro_ar[i, 2:] for i in range(pro_ar.shape[0])}

    def __len__(self):
        # number of unique batches which can be generated
        return len(self.pairs_list) // self.batch_size

    def __convert(self, pairs: list) -> (np.ndarray, np.ndarray):
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
        pos_pairs = self.pairs_list[self.batch_size * index: self.batch_size * (index + 1)]
        neg_pairs = []

        for i in range(len(pos_pairs)):
            while True:
                # sample negative pair candidate
                que = random.choice(self.ques)
                pro = random.choice(self.pros)
                # check if it's not a positive pair
                if (que, pro) not in self.pairs_set:
                    neg_pairs.append((que, pro))
                    break

        # convert lists of pairs to NumPy arrays of features
        x_pos_que, x_pos_pro = self.__convert(pos_pairs)
        x_neg_que, x_neg_pro = self.__convert(neg_pairs)

        # return the data in its final form
        return [np.vstack([x_pos_que, x_neg_que]), np.vstack([x_pos_pro, x_neg_pro])], \
               np.vstack([np.ones((len(x_pos_que), 1)), np.zeros((len(x_neg_que), 1))])

    def on_epoch_end(self):
        # shuffle positive pairs
        self.pairs_list = random.sample(self.pairs_list, len(self.pairs_list))
