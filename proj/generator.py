import random

import keras
import numpy as np
import pandas as pd


class BatchGenerator(keras.utils.Sequence):
    def __init__(self, que: pd.DataFrame, pro: pd.DataFrame, batch_size):
        self.batch_size = batch_size

        self.ques = list(que['questions_id'])
        self.pros = list(pro['professionals_id'])

        a = {(row['questions_id'], author) for i, row in que.iterrows()
             for author in str(row['answers_author_id']).split()}
        b = {(question, row['professionals_id']) for i, row in pro.iterrows()
             for question in str(row['answers_question_id']).split()}

        print(len(a), len(b), len(a & b))

        self.pairs_set = a & b
        self.pairs_list = list(self.pairs_set)

        que_ar = que.values
        self.que_feat = {que_ar[i, 0]: que_ar[i, 2:] for i in range(que_ar.shape[0])}
        pro_ar = pro.values
        self.pro_feat = {pro_ar[i, 0]: pro_ar[i, 2:] for i in range(pro_ar.shape[0])}

    def __len__(self):
        return len(self.pairs_list) // self.batch_size

    def __convert(self, pairs):
        x_que, x_pro = [], []
        for i, (que, pro) in enumerate(pairs):
            x_que.append(self.que_feat[que])
            x_pro.append(self.pro_feat[pro])
        return np.vstack(x_que), np.vstack(x_pro)

    def __getitem__(self, index):
        pos_pairs = self.pairs_list[self.batch_size * index: self.batch_size * (index + 1)]
        neg_pairs = []

        for i in range(len(pos_pairs)):
            while True:
                que = random.choice(self.ques)
                pro = random.choice(self.pros)
                if (que, pro) not in self.pairs_set:
                    neg_pairs.append((que, pro))
                    break

        x_pos_que, x_pos_pro = self.__convert(pos_pairs)
        x_neg_que, x_neg_pro = self.__convert(neg_pairs)

        return [np.vstack([x_pos_que, x_neg_que]), np.vstack([x_pos_pro, x_neg_pro])], \
               np.vstack([np.ones((len(x_pos_que), 1)), np.zeros((len(x_neg_que), 1))])

    def on_epoch_end(self):
        self.pairs_list = random.sample(self.pairs_list, len(self.pairs_list))
