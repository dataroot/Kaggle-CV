import pickle
import random

import keras
import numpy as np
import pandas as pd

from utils import TextProcessor


class BatchGenerator(keras.utils.Sequence):
    """
    Generates batch of data in train and test modes
    """
    
    def __init__(self, pos_size, neg_size, mode='train', return_stat=True, data_path='../../data/'):
        self.pos_size = pos_size
        self.neg_size = neg_size
        self.return_stat = return_stat
        
        que = pd.read_csv(data_path + 'questions.csv')
        tag_que = pd.read_csv(data_path + 'tag_questions.csv')
        tags = pd.read_csv(data_path + 'tags.csv')
        pro = pd.read_csv(data_path + 'professionals.csv')
        stu = pd.read_csv(data_path + 'students.csv')
        ans = pd.read_csv(data_path + 'answers.csv')
        tag_users = pd.read_csv(data_path + 'tag_users.csv')
        
        self.tp = TextProcessor()
        pro['professionals_industry'] = pro['professionals_industry'].apply(self.tp.process)
        tags['tags_tag_name'] = tags['tags_tag_name'].apply(lambda x: self.tp.process(x, allow_stopwords=True))
        
        self.pro_ind = {row['professionals_id']: row['professionals_industry'] for i, row in pro.iterrows()}
        
        que_tags = que.merge(tag_que, left_on = 'questions_id', right_on = 'tag_questions_question_id').merge(tags, left_on = 'tag_questions_tag_id', right_on = 'tags_tag_id')
        que_tags = que_tags[['questions_id', 'tags_tag_name']].groupby(by = 'questions_id', as_index = False).aggregate(lambda x: ' '.join(x))
        self.que_tags = {row['questions_id']: row['tags_tag_name'].split() for _, row in que_tags.iterrows()}
        
        pro_tags = pro.merge(tag_users, left_on='professionals_id', right_on='tag_users_user_id').merge(tags, left_on = 'tag_users_tag_id', right_on='tags_tag_id')
        pro_tags = pro_tags[['professionals_id', 'tags_tag_name']].groupby(by='professionals_id', as_index = False).aggregate(lambda x: ' '.join(x))
        self.pro_tags = {row['professionals_id']: row['tags_tag_name'].split() for _, row in pro_tags.iterrows()}
        
        stu_tags = stu.merge(tag_users, left_on='students_id', right_on='tag_users_user_id').merge(tags, left_on = 'tag_users_tag_id', right_on='tags_tag_id')
        stu_tags = stu_tags[['students_id', 'tags_tag_name']].groupby(by='students_id', as_index = False).aggregate(lambda x: ' '.join(x))
        self.stu_tags = {row['students_id']: row['tags_tag_name'].split() for _, row in stu_tags.iterrows()}
        
        ans_que = ans.merge(que, left_on = 'answers_question_id', right_on = 'questions_id')
        ans_que_pro = ans_que.merge(pro, left_on = 'answers_author_id', right_on = 'professionals_id')
        ans_que_pro = ans_que_pro.merge(stu, left_on = 'questions_author_id', right_on = 'students_id')
        
        # Add a dictionary mapping answer to (question, professional) pair
        self.ans_que_pro_dict = {row['answers_id']:(row['questions_id'], row['professionals_id'])
                                 for _, row in ans_que_pro.iterrows()}
        
        self.que_stu_dict = {row['questions_id']: row['students_id'] for _, row in ans_que_pro.iterrows()}
        self.que_pro_set = {(row['questions_id'], row['professionals_id']) for _, row in ans_que_pro.iterrows()}
        
        with open('tags_embs.pkl', 'rb') as f:
            self.tag_emb = pickle.load(f)
        with open('industries_embs.pkl', 'rb') as f:
            self.ind_emb = pickle.load(f)
        
        # Load que and pro statistical features
        with open('que_feature_dict.pickle', 'rb') as f:
            self.que_feature_dict = pickle.load(f)
        with open('pro_feature_dict.pickle', 'rb') as f:
            self.pro_feature_dict = pickle.load(f)
        
        # Load time related dicts and lists
        with open('pro_answer_dates_dict.pickle', 'rb') as f:
            self.pro_answer_dates_dict = pickle.load(f)
        with open('ans_date_added_dict.pickle', 'rb') as f:
            self.ans_date_added_dict = pickle.load(f)
        with open('que_date_added_dict.pickle', 'rb') as f:
            self.que_date_added_dict = pickle.load(f)
        with open('ans_prev_answer_date_dict.pickle', 'rb') as f:
            self.ans_prev_answer_date_dict = pickle.load(f)
        with open('pro_list.pickle', 'rb') as f:
            self.pro_list = pickle.load(f)
        with open('pro_reg_date_list.pickle', 'rb') as f:
            self.pro_reg_date_list = pickle.load(f)
        
        # Load preprocessors
        with open('preprocessors.pickle', 'rb') as f:
            self.preproc = pickle.load(f)
        
        # Load answer list depending on the mode
        if mode == 'train':
            with open('train_ans_list.pickle', 'rb') as f:
                self.ans_list = pickle.load(f)
        elif mode == 'test':
            with open('test_ans_list.pickle', 'rb') as f:
                self.ans_list = pickle.load(f)
        
        # Load questions embeddings
        with open('questions_embs.pkl', 'rb') as f:
            self.que_emb = pickle.load(f)
    
    
    def __len__(self):
        return len(self.ans_list) // self.pos_size
    
    
    def __getitem__(self, index):
        pos_pairs = []
        neg_pairs = []
        
        pos_prev_dates = []
        neg_prev_dates = []
        
        pos_cur_times = []
        neg_cur_times = []
        
        pos_que_features, pos_pro_features = [], []
        neg_que_features, neg_pro_features = [], []
        
        pos_ans = self.ans_list[self.pos_size * index: self.pos_size * (index + 1)]
        for ans in pos_ans:
            # Add que and pro features and dates to appropriate lists
            que, pro = self.ans_que_pro_dict[ans]
            pos_pairs.append((que, pro))
            pos_que_features.append(self.que_feature_dict[que])
            pos_pro_features.append(self.pro_feature_dict[pro])
            pos_prev_dates.append(self.ans_prev_answer_date_dict[ans])
            pos_cur_times.append(self.ans_date_added_dict[ans])
        
        for i in range(self.neg_size):
            ans = random.choice(self.ans_list)
            que, _ = self.ans_que_pro_dict[ans]
            
            # Current time is e^(exponential with mean 3 truncated at 6) - 1
            mean = 3
            sample = np.random.exponential(mean)
            while sample >= 6:
                sample = np.random.exponential(mean)
            cur_time = np.exp(sample) - 1
            
            # Inverse transform current time
            cur_time = (self.preproc['questions_date_added_time']
                        .inverse_transform([[self.que_date_added_dict[que]]])[0][0] + cur_time / 365)
            
            # Include professionals whos registration date is belove threshold
            threshold = np.searchsorted(self.pro_reg_date_list, cur_time)
            valid_pros = self.pro_list[:threshold]
            
            # Transform current time with preprocessor for professionals_prev_answer_date_time
            cur_time = self.preproc['professionals_prev_answer_date_time'].transform([[cur_time]])[0][0]
            
#             #-------------------------------------------------------------------------
#             #                          WITH DISTRIBUTION
            
#             # Sample 50 (or less) pros among valid ones
#             sampled_pros = random.sample(valid_pros, min(50, len(valid_pros)))
            
#             pros = []
#             prev_answer_dates = []
            
#             # Compute previous answer date for every sampled professional
#             for pro in sampled_pros:
#                 if (que, pro) not in self.que_pro_set:
#                     prev_answer_date = self.__negative_que_prev_answer_date(pro, cur_time)
                    
#                     pros.append(pro)
#                     prev_answer_dates.append(prev_answer_date)
            
#             if len(pros) == 0:
#                 continue
            
#             # Substact prev answer dates from cur_time
#             distances = cur_time - np.array(prev_answer_dates)
            
#             # Apply log1p transformation to 1 / distances and normalize each entry
#             distances = np.log1p(1 / distances)
#             distances /= distances.sum()
            
#             # Sample one professional from distribution of distances
#             pro = np.random.choice(pros, p=distances)
#             #-------------------------------------------------------------------------
        
            #-------------------------------------------------------------------------
            #                         WITHOUT DISTRIBUTION

            pro = random.choice(valid_pros)
            while (que, pro) in self.que_pro_set:
                pro = random.choice(valid_pros)
            #-------------------------------------------------------------------------
            
            # Add que and pro data to all required lists
            prev_date = self.__negative_que_prev_answer_date(pro, cur_time)
            neg_pairs.append((que, pro))
            neg_que_features.append(self.que_feature_dict[que])
            neg_pro_features.append(self.pro_feature_dict[pro])
            neg_prev_dates.append(prev_date)
            neg_cur_times.append(cur_time)
        
        pos_que_embeddings, pos_pro_embeddings = self.__convert(pos_pairs)
        neg_que_embeddings, neg_pro_embeddings = self.__convert(neg_pairs)
        
        pos_que_stat = np.hstack([
            np.array(pos_que_features),
#             np.array(pos_cur_times)[:, np.newaxis],
#             pos_que_embeddings,
        ])
        neg_que_stat = np.hstack([
            np.array(neg_que_features),
#             np.array(neg_cur_times)[:, np.newaxis],
#             neg_que_embeddings,
        ])
        
        pos_pro_stat = np.hstack([
            np.array(pos_pro_features),
#             np.array(pos_prev_dates)[:, np.newaxis],
#             np.array(pos_cur_times)[:, np.newaxis],
#             pos_pro_embeddings,
        ])
        neg_pro_stat = np.hstack([
            np.array(neg_pro_features),
#             np.array(neg_prev_dates)[:, np.newaxis],
#             np.array(neg_cur_times)[:, np.newaxis],
#             neg_pro_embeddings,
        ])
        
        return_list = [
            np.vstack([pos_que_embeddings, neg_que_embeddings]), np.vstack([pos_pro_embeddings, neg_pro_embeddings])]
        
        # TODO: change features to stat
        if self.return_stat:
            return_list.append(np.vstack([pos_que_stat, neg_que_stat]))
            return_list.append(np.vstack([pos_pro_stat, neg_pro_stat]))
        
        target = np.vstack([np.ones((self.pos_size, 1)), np.zeros((self.neg_size, 1))])
        
        return return_list, target
    
    
    def __negative_que_prev_answer_date(self, pro, cur_time):
        pro_dates = self.pro_answer_dates_dict[pro]
        
        index = np.searchsorted(pro_dates, cur_time)
        if index == 0:
            raise ValueError("Index cannot be zero.")
        return pro_dates[index-1]
    
    
    def __convert(self, pairs):
        x_que, x_pro = [], []
        for que, pro in pairs:
            stu = self.que_stu_dict[que]
            
            que_tags = []
            pro_tags = []
            stu_tags = []
            
            # Average embedding of question tags
            for tag in self.que_tags.get(que, []):
                que_tags.append(self.tag_emb.get(tag, np.zeros(10)))
            if len(que_tags) == 0:
                que_tags.append(np.zeros(10))
            que_tag_emb = np.vstack(que_tags).mean(axis = 0).reshape(-1)
            
            # Average embedding of professional tags
            for tag in self.pro_tags.get(pro, []):
                pro_tags.append(self.tag_emb.get(tag, np.zeros(10)))
            if len(pro_tags) == 0:
                pro_tags.append(np.zeros(10))
            pro_tag_emb = np.vstack(pro_tags).mean(axis = 0).reshape(-1)
            
#             # Average embedding of student tags
#             for tag in self.stu_tags.get(stu, []):
#                 stu_tags.append(self.tag_emb.get(tag, np.zeros(10)))
#             if len(stu_tags) == 0:
#                 stu_tags.append(np.zeros(10))
#             stu_tag_emb = np.vstack(stu_tags).mean(axis = 0).reshape(-1)
            
            # Collect all question and student embeddings
            que_emb = self.que_emb[que]
            x_que.append(np.hstack([que_emb,
                                    que_tag_emb,
#                                     stu_tag_emb
                                    ]))
            
            # Collect all professional embeddings
            ind_emb = self.ind_emb.get(self.pro_ind[pro], np.zeros(10))
            x_pro.append(np.hstack([ind_emb,
                                    pro_tag_emb
                                    ]))
        
        return np.vstack(x_que), np.vstack(x_pro)
    
    
    def on_epoch_end(self):
        np.random.shuffle(self.ans_list)

