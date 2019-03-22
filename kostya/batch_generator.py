import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import keras
from keras.utils import Sequence

from utils import TextProcessor


class BatchGenerator(Sequence):
    """
    Generates batches of data to feed into the model
    """
    
    def __init__(self, pp: Preprocessor, batch_size: int = 50, shuffle: bool = True):
        """
        Loads required datasets from pp, batch_size and shuffle parameters
        """
        self.qa_data = pp.qa_data.merge(pp.stud_data, on='students_id')
        self.prof_data = pp.prof_data
        
        # Select unique professionals from the ones that answered at least one question
        self.unique_profs = pp.prof_data.professionals_id.unique()
        
        #----------------------------------------------------------------------------
        #               INTEGRATION WITH NIKITA'S BATCH GENERATOR
        #----------------------------------------------------------------------------
        
        # Load required datasets (their names are left as they were in Nikita's batch generator)
        tag_que = pp.tag_questions
        tags = pp.tags
        pro = pp.prof_data
        que = pp.qa_data
        
        # Import precomputed embeddings
        with open('tags_embs.pickle', 'rb') as file:
            self.tag_emb = pickle.load(file)
        with open('industries_embs.pickle', 'rb') as file:
            self.ind_emb = pickle.load(file)
        
        # Preprocess professionals industries
        self.tp = TextProcessor()
        pro['professionals_industry_textual'] = (pro['professionals_industry_textual']
                                                 .apply(self.tp.process)
                                                 .apply(lambda x: ' '.join(x)))
        
        # Map professionals_id to professionals_industry_textual
        self.pro_ind = {row['professionals_id']: row['professionals_industry_textual'] for i, row in pro.iterrows()}
        
        # Create string of tags for every question
        que_tags = (que.merge(tag_que, left_on='questions_id', right_on='tag_questions_question_id')
                       .merge(tags, left_on='tag_questions_tag_id', right_on='tags_tag_id'))
        que_tags = (que_tags[['questions_id', 'tags_tag_name']]
                    .groupby('questions_id', as_index=False)
                    .aggregate(lambda x: ' '.join(x)))
        
        # Map questions_id to string of tags
        self.que_tag = {row['questions_id']: row['tags_tag_name'].split() for i, row in que_tags.iterrows()}
        
        #----------------------------------------------------------------------------
        
        # Set batch_size and shuffle parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Initial shuffle 
        self.on_epoch_end()
    
    
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return self.qa_data.shape[0] // (self.batch_size)
    
    
    def __getitem__(self, index):
        """
        Generates one batch of data
        """
        # Positive batch is selected by index
        positive_batch = self.qa_data.iloc[index * self.batch_size : (index + 1) * self.batch_size, :]
        negative_batch = positive_batch
        
        # Choose random professionals for negative batch
        cur_profs = negative_batch.professionals_id
        new_profs = np.random.choice(self.unique_profs, self.batch_size)
        
        # Check if all professionals from negative batch are different from true professionals
        while np.sum(cur_profs == new_profs) > 0:
            # If not (tiny probability), resample random professionals
            new_profs = np.random.choice(self.unique_profs, self.batch_size)
        
        # Assign random professionals to negative batch
        negative_batch.assign(professionals_id=new_profs)
        
        # Concatenate positive and negative batches into a single batch
        single_batch = pd.concat([positive_batch, negative_batch])
        
        # Add professionals data to single_batch
        single_batch = single_batch.merge(self.prof_data, on='professionals_id')
        
        # Select statistical question features
        x_que_features = single_batch[[
            'students_location', 'students_state', 'students_questions_asked',
            'students_average_question_age', 'students_average_question_body_length',
            'students_average_answer_body_length',
            
            'students_date_joined_time', 'students_date_joined_doy_sin',
            'students_date_joined_doy_cos', 'students_date_joined_dow',
            
            'questions_body_length',
            
            'questions_date_added_time', 'questions_date_added_doy_sin',
            'questions_date_added_doy_cos', 'questions_date_added_dow',
            'questions_date_added_hour_sin', 'questions_date_added_hour_cos',
        ]].values
        
        # Select statistical professional features
        x_pro_features = single_batch[[
            'professionals_industry', 'professionals_location', 'professionals_state',
            'professionals_questions_answered', 'professionals_average_question_age',
            'professionals_average_question_body_length', 'professionals_average_answer_body_length',
            'professionals_email_activated',
            
            'professionals_date_joined_time', 'professionals_date_joined_doy_sin',
            'professionals_date_joined_doy_cos', 'professionals_date_joined_dow',
            
            'professionals_last_answer_date_time', 'professionals_last_answer_date_doy_sin',
            'professionals_last_answer_date_doy_cos', 'professionals_last_answer_date_dow',
            'professionals_last_answer_date_hour_sin', 'professionals_last_answer_date_hour_cos',
        ]].values
        
        #----------------------------------------------------------------------------
        #               INTEGRATION WITH NIKITA'S BATCH GENERATOR
        #----------------------------------------------------------------------------
        
        # Extract embeddings from batch questions and professionals
        x_que_embeddings, x_pro_embeddings = self.__convert(
            single_batch[['questions_id', 'professionals_id']].values)
        
        # Stack statistical features and embeddings
        x_que = np.hstack((x_que_features, x_que_embeddings))
        x_pro = np.hstack((x_pro_features, x_pro_embeddings))
        
        #----------------------------------------------------------------------------
        
        # Create target array
        y = np.concatenate([np.ones(self.batch_size), np.zeros(self.batch_size)])
        
        return (x_que, x_pro), y
    
    
    def on_epoch_end(self):
        """
        Shuffle qa_data after each epoch
        """
        if self.shuffle:
            self.qa_data = shuffle(self.qa_data)
    
    
    def __convert(self, batch):
        """
        Computes embeddings for questions based on average of precomputed tag embeddings
        and embeddings for professionals based on precomputed industry embeddings
        """
        x_que, x_pro = [], []
        
        for que, pro in batch:
            tmp = []
            
            for tag in self.que_tag.get(que, ['#']):
                tmp.append(self.tag_emb.get(tag, np.zeros(10)))
            x_que.append(np.vstack(tmp).mean(axis = 0))
            x_pro.append(self.ind_emb.get(self.pro_ind[pro], np.zeros(10)))
        
        return np.vstack(x_que), np.vstack(x_pro)

