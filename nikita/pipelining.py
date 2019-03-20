import pickle
import random

import keras
import pandas as pd
import numpy as np

# Some useful functions and TextProcessor are defined here
from utils import TextProcessor


class BatchGenerator(keras.utils.Sequence):
    def __init__(self, pos_size, neg_size, data_path):
        self.pos_size = pos_size
        self.neg_size = neg_size
        
        que = pd.read_csv(data_path + 'questions.csv')
        tag_que = pd.read_csv(data_path + 'tag_questions.csv')
        tags = pd.read_csv(data_path + 'tags.csv')
        pro = pd.read_csv(data_path + 'professionals.csv')
        ans = pd.read_csv(data_path + 'answers.csv')
        
        # Preprocess professionals industries
        self.tp = TextProcessor()
        pro['professionals_industry'] = pro['professionals_industry'].apply(self.tp.process)
        pro['professionals_industry'] = pro['professionals_industry'].apply(lambda x: ' '.join(x))
        
        self.pro_ind = {row['professionals_id']: row['professionals_industry'] for i, row in pro.iterrows()}
        
        que_tags = que.merge(tag_que, left_on = 'questions_id', right_on = 'tag_questions_question_id').merge(tags, left_on = 'tag_questions_tag_id', right_on = 'tags_tag_id')
        que_tags = que_tags[['questions_id', 'tags_tag_name']].groupby(by = 'questions_id', as_index = False).aggregate(lambda x: ' '.join(x))
        self.que_tag = {row['questions_id']: row['tags_tag_name'].split() for i, row in que_tags.iterrows()}
        
        ans_que = ans.merge(que, left_on = 'answers_question_id', right_on = 'questions_id')
        ans_que_pro = ans_que.merge(pro, left_on = 'answers_author_id', right_on = 'professionals_id')
        
        self.ques = list(set(ans_que_pro['questions_id']))
        self.pros = list(set(ans_que_pro['professionals_id']))
        
        self.que_pro_set = {(row['questions_id'], row['professionals_id']) for i, row in ans_que_pro.iterrows()}
        self.que_pro_list = list(self.que_pro_set)
        
        # Import precomputed embeddings
        with open('tags_embs.pickle', 'rb') as file:
            self.tag_emb = pickle.load(file)
        with open('industries_embs.pickle', 'rb') as file:
            self.ind_emb = pickle.load(file)
    
    
    def __len__(self):
        return len(self.que_pro_list) // self.pos_size
    
    
    # Example of how to compute embeddings for questions based on average of precomputed tag embeddings
    # and for professionals based on precomputed industry embedding
    def __convert(self, pairs):
        x_que, x_pro = [], []
        for que, pro in pairs:
            tmp = []
            for tag in self.que_tag.get(que, ['#']):
                tmp.append(self.tag_emb.get(tag, np.zeros(10)))
            x_que.append(np.vstack(tmp).mean(axis = 0))
            x_pro.append(self.ind_emb.get(self.pro_ind[pro], np.zeros(10)))
        return np.vstack(x_que), np.vstack(x_pro)
    
    
    # This is what I should merge my batch generator with
    def __getitem__(self, index):
        pos_pairs = self.que_pro_list[self.pos_size * index: self.pos_size * (index + 1)]
        neg_pairs = []
        
        # Generate every negative professional one by one
        # Done differently in my batch generator
        for i in range(self.neg_size):
            while True:
                que = random.choice(self.ques)
                pro = random.choice(self.pros)
                if (que, pro) not in self.que_pro_set:
                    neg_pairs.append((que, pro))
                    break
        
        x_pos_que, x_pos_pro = self.__convert(pos_pairs)
        x_neg_que, x_neg_pro = self.__convert(neg_pairs)
        
        return [np.vstack([x_pos_que, x_neg_que]), np.vstack([x_pos_pro, x_neg_pro])], \
                np.vstack([np.ones((len(x_pos_que), 1)), np.zeros((len(x_neg_que), 1))])
    
    
    # Already have one in my batch generator
    def on_epoch_end(self):
        self.que_pro_list = random.sample(self.que_pro_list, len(self.que_pro_list))
