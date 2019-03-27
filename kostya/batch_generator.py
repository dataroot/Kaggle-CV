import pickle
import random

import keras
import numpy as np
import pandas as pd

from utils import TextProcessor


class BatchGenerator(keras.utils.Sequence):
    def __init__(self, pos_size, neg_size, data_path='../../data/'):
        self.pos_size = pos_size
        self.neg_size = neg_size
        
        que = pd.read_csv(data_path + 'questions.csv')
        tag_que = pd.read_csv(data_path + 'tag_questions.csv')
        tags = pd.read_csv(data_path + 'tags.csv')
        pro = pd.read_csv(data_path + 'professionals.csv')
        stu = pd.read_csv(data_path + 'students.csv')
        ans = pd.read_csv(data_path + 'answers.csv')
        
        self.tp = TextProcessor()
        pro['professionals_industry'] = pro['professionals_industry'].apply(self.tp.process)
        tags['tags_tag_name'] = tags['tags_tag_name'].apply(lambda x: self.tp.process(x, allow_stopwords=True))
        
        self.pro_ind = {row['professionals_id']: row['professionals_industry'] for i, row in pro.iterrows()}
        
        que_tags = que.merge(tag_que, left_on = 'questions_id', right_on = 'tag_questions_question_id').merge(tags, left_on = 'tag_questions_tag_id', right_on = 'tags_tag_id')
        que_tags = que_tags[['questions_id', 'tags_tag_name']].groupby(by = 'questions_id', as_index = False).aggregate(lambda x: ' '.join(x))
        self.que_tag = {row['questions_id']: row['tags_tag_name'].split() for _, row in que_tags.iterrows()}
        
        ans_que = ans.merge(que, left_on = 'answers_question_id', right_on = 'questions_id')
        ans_que_pro = ans_que.merge(pro, left_on = 'answers_author_id', right_on = 'professionals_id')
        ans_que_pro = ans_que_pro.merge(stu, left_on = 'questions_author_id', right_on = 'students_id')
        
        self.ques = list(set(ans_que_pro['questions_id']))
        self.pros = list(set(ans_que_pro['professionals_id']))
        
        self.que_pro_set = {(row['questions_id'], row['professionals_id']) for _, row in ans_que_pro.iterrows()}
        self.que_pro_list = list(self.que_pro_set)
        
        with open('tags_embs.pickle', 'rb') as file:
            self.tag_emb = pickle.load(file)
        with open('industries_embs.pickle', 'rb') as file:
            self.ind_emb = pickle.load(file)
        
        #------------------------------------------------------------------
        #                    THE CODE I ADDED
        #------------------------------------------------------------------
        
        # Add answer list and a dictionary mapping answer to (question, professional) pair
        self.ans_que_pro_dict = {row['answers_id']:(row['questions_id'], row['professionals_id'])
                                 for _, row in ans_que_pro.iterrows()}
        
        # Load que and pro statistical features
        with open('que_feature_dict.pickle', 'rb') as f:
            self.que_feature_dict = pickle.load(f)
        with open('pro_feature_dict.pickle', 'rb') as f:
            self.pro_feature_dict = pickle.load(f)
        
        # Load pro last answer dates dict and que answer date dict
        with open('pro_last_answer_dates_dict.pickle', 'rb') as f:
            self.pro_last_answer_dates_dict = pickle.load(f)
        with open('ans_date_added_dict.pickle', 'rb') as f:
            self.ans_date_added_dict = pickle.load(f)
        with open('ans_last_answer_date_dict.pickle', 'rb') as f:
            self.ans_last_answer_date_dict = pickle.load(f)
        with open('ans_sampled_profs_dict.pickle', 'rb') as f:
            self.ans_sampled_profs_dict = pickle.load(f)
        with open('ans_list.pickle', 'rb') as f:
            self.ans_list = pickle.load(f)
        #------------------------------------------------------------------
    
    
    def __len__(self):
        return len(self.que_pro_list) // self.pos_size
    
    
    def __convert(self, pairs):
        x_que, x_pro = [], []
        for que, pro in pairs:
            tmp = []
            for tag in self.que_tag.get(que, []):
                tmp.append(self.tag_emb.get(tag, np.zeros(10)))
            if len(tmp) == 0:
                tmp.append(np.zeros(10))
            
            x_que.append(np.vstack(tmp).mean(axis = 0))
            x_pro.append(self.ind_emb.get(self.pro_ind[pro], np.zeros(10)))
        
        return np.vstack(x_que), np.vstack(x_pro)
    
    
    def __negative_que_last_answer_date(self, ans, pro):
        ans_date = self.ans_date_added_dict[ans]
        pro_dates = self.pro_last_answer_dates_dict[pro]
        
        index = np.searchsorted(pro_dates, ans_date)
        if index == 0:
            raise ValueError("Index cannot be zero.")
        return pro_dates[index-1]     
    
    
    def __getitem__(self, index):
        pos_pairs = []
        neg_pairs = []
        
        pos_last_dates = []
        neg_last_dates = []
        
        pos_que_features, pos_pro_features = [], []
        neg_que_features, neg_pro_features = [], []
        
        pos_ans = self.ans_list[self.pos_size * index: self.pos_size * (index + 1)]
        for ans in pos_ans:
            que, pro = self.ans_que_pro_dict[ans]
            pos_pairs.append((que, pro))
            pos_que_features.append(self.que_feature_dict[que])
            pos_pro_features.append(self.pro_feature_dict[pro])
            pos_last_dates.append(self.ans_last_answer_date_dict[ans])
        
        for i in range(self.neg_size):
            ans = random.choice(self.ans_list)
            que, _ = self.ans_que_pro_dict[ans]
            pro = random.choice(self.ans_sampled_profs_dict[ans])
            
            # Add que and pro data to all required lists
            last_date = self.__negative_que_last_answer_date(ans, pro)
            neg_pairs.append((que, pro))
            neg_que_features.append(self.que_feature_dict[que])
            neg_pro_features.append(self.pro_feature_dict[pro])
            neg_last_dates.append(last_date)
        
        pos_que_embeddings, pos_pro_embeddings = self.__convert(pos_pairs)
        neg_que_embeddings, neg_pro_embeddings = self.__convert(neg_pairs)
        
        x_pos_que = np.hstack([np.array(pos_que_features), pos_que_embeddings])
        x_neg_que = np.hstack([np.array(neg_que_features), neg_que_embeddings])
        
        # print(np.array(pos_pro_features).shape, np.array(pos_last_dates)[:, np.newaxis].shape, pos_pro_embeddings.shape)
        x_pos_pro = np.hstack([np.array(pos_pro_features), np.array(pos_last_dates)[:, np.newaxis], pos_pro_embeddings])
        x_neg_pro = np.hstack([np.array(neg_pro_features), np.array(neg_last_dates)[:, np.newaxis], neg_pro_embeddings])
        
        return [np.vstack([x_pos_que, x_neg_que]), np.vstack([x_pos_pro, x_neg_pro])], \
                np.vstack([np.ones((self.pos_size, 1)), np.zeros((self.neg_size, 1))])
    
    
    def on_epoch_end(self):
        np.random.shuffle(self.ans_list)

