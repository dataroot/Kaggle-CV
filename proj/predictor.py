import pickle

import pandas as pd
import numpy as np

from sklearn.neighbors import KDTree

from processors import QueProc, ProProc


class Predictor:
    """
    Class that creates KNN tree for professionals
    and which is used to find closest professionals for a particular question
    """

    def __init__(self, model, path):
        """
        Prepare required datasets and create KNN tree for professionals
        based on latent vectors from content model
        :param model: compiled Keras model
        """
        self.model = model

        # form question-student pairs dataframe
        with open(path + 'que_stu_pairs.pkl', 'rb') as file:
            pairs = pickle.load(file)

        # load datasets with preprocessed features
        store = pd.HDFStore(path + 'processed.h5', 'r')
        que_data = pd.read_hdf(store, 'que')
        stu_data = pd.read_hdf(store, 'stu')
        pro_data = pd.read_hdf(store, 'pro')

        self.que_dict = {row.values[0]: row.values[2:] for i, row in que_data.iterrows()}
        self.stu_dict = {stu: group.values[-1, 2:] for stu, group in stu_data.groupby('students_id')}
        self.pro_dict = {pro: group.values[-1, 2:] for pro, group in pro_data.groupby('professionals_id')}

        que_feat, que_ids, pro_feat, pro_ids = [], [], [], []

        for que in self.que_dict.keys():
            cur_stu = pairs[que]
            if cur_stu in self.stu_dict:
                que_feat.append(np.hstack([self.stu_dict[cur_stu], self.que_dict[que]]))
                que_ids.append(que)

        for pro in self.pro_dict.keys():
            pro_feat.append(self.pro_dict[pro])
            pro_ids.append(pro)

        self.pro_feat = np.vstack(pro_feat)
        self.pro_ids = np.vstack(pro_ids)
        self.que_feat = np.vstack(que_feat)
        self.que_ids = np.vstack(que_ids)

        # create two encoders
        self.que_model = model.que_model
        self.pro_model = model.pro_model

        # compute latent vectors for questions and professionals
        self.que_lat_vecs = self.que_model.predict(self.que_feat)

        print(self.pro_feat[2637])
        self.pro_lat_vecs = self.pro_model.predict(self.pro_feat)
        print(self.pro_lat_vecs[2637])

        # create two KNN trees consisting of question and professional latent vectors
        self.que_tree = KDTree(self.que_lat_vecs)
        self.pro_tree = KDTree(self.pro_lat_vecs)

        # initialize QueProc and ProProc
        self.que_proc = QueProc(oblige_fit=False, path=path)
        self.pro_proc = ProProc(oblige_fit=False, path=path)

    def __get_que_latent(self, que_df: pd.DataFrame, que_tags: pd.DataFrame):
        que_df['questions_date_added'] = pd.to_datetime(que_df['questions_date_added'])

        que_feat = self.que_proc.transform(que_df, que_tags).values[:, 2:]

        stu_feat = np.vstack([self.stu_dict[stu] for stu in que_df['questions_author_id']])
        que_feat = np.hstack([stu_feat, que_feat])

        lat_vecs = self.que_model.predict(que_feat)

        return lat_vecs

    def __get_pro_latent(self, pro_df: pd.DataFrame, que_df: pd.DataFrame, ans_df: pd.DataFrame,
                         pro_tags: pd.DataFrame):
        pro_df['professionals_date_joined'] = pd.to_datetime(pro_df['professionals_date_joined'])
        que_df['questions_date_added'] = pd.to_datetime(que_df['questions_date_added'])
        ans_df['answers_date_added'] = pd.to_datetime(que_df['questions_date_added'])

        pro_feat = self.pro_proc.transform(pro_df, que_df, ans_df, pro_tags)
        pro_feat = pro_feat.groupby('professionals_id').last().values[:, 1:]

        print(pro_feat)
        lat_vecs = self.pro_model.predict(pro_feat)
        print(lat_vecs)

        return lat_vecs

    @staticmethod
    def __construct_df(ids, sims, scores):
        tuples = []
        for i, id in enumerate(ids):
            for j, sim_que in enumerate(sims[i]):
                tuples.append((id, sim_que, scores[i, j]))
        score_df = pd.DataFrame(tuples, columns=['id', 'match_id', 'match_score'])
        return score_df

    def __get_ques_by_latent(self, ids, lat_vecs, top):
        dists, ques = self.que_tree.query(lat_vecs, k=top)
        ques = self.que_ids[ques]
        scores = np.exp(-dists)
        return Predictor.__construct_df(ids, ques, scores)

    def __get_pros_by_latent(self, ids, lat_vecs, top):
        dists, pros = self.pro_tree.query(lat_vecs, k=top)
        pros = self.pro_ids[pros]
        scores = np.exp(-dists)
        return Predictor.__construct_df(ids, pros, scores)

    def find_pros_by_que(self, que_df: pd.DataFrame, que_tags: pd.DataFrame, top: int = 10) -> pd.DataFrame:
        lat_vecs = self.__get_que_latent(que_df, que_tags)
        return self.__get_pros_by_latent(que_df['questions_id'].values, lat_vecs, top)

    def find_ques_by_que(self, que_df: pd.DataFrame, que_tags: pd.DataFrame, top: int = 10) -> pd.DataFrame:
        lat_vecs = self.__get_que_latent(que_df, que_tags)
        return self.__get_ques_by_latent(que_df['questions_id'].values, lat_vecs, top)

    # TODO: consider professionals which are already in base

    def find_ques_by_pro(self, pro_df: pd.DataFrame, que_df: pd.DataFrame, ans_df: pd.DataFrame,
                         pro_tags: pd.DataFrame, top: int = 10) -> pd.DataFrame:
        lat_vecs = self.__get_pro_latent(pro_df, que_df, ans_df, pro_tags)
        return self.__get_ques_by_latent(pro_df['professionals_id'].values, lat_vecs, top)

    def find_pros_by_pro(self, pro_df: pd.DataFrame, que_df: pd.DataFrame, ans_df: pd.DataFrame,
                         pro_tags: pd.DataFrame, top: int = 10) -> pd.DataFrame:
        lat_vecs = self.__get_pro_latent(pro_df, que_df, ans_df, pro_tags)
        return self.__get_pros_by_latent(pro_df['professionals_id'].values, lat_vecs, top)

    # TODO: refactor this

    @staticmethod
    def convert_que_dict(que_dict: dict) -> (pd.DataFrame, pd.DataFrame):
        """
        Converts dictionary of question data into desired form
        :param que_dict: dictionary of question data
        """
        # get DataFrame from dict
        que_df = pd.DataFrame.from_dict(que_dict)
        ques = que_df['questions_id'].values

        # create question-tag tuples
        tuples = []
        for i, tags in enumerate(que_df['questions_tags'].values):
            que = ques[i]
            for tag in tags.split(' '):
                tuples.append((que, tag))

        # create DataFrame from tuples
        que_tags = pd.DataFrame(tuples, columns=['tag_questions_question_id', 'tags_tag_name'])
        que_df.drop(columns='questions_tags', inplace=True)

        return que_df, que_tags

    @staticmethod
    def convert_pro_dict(pro_dict: dict) -> (pd.DataFrame, pd.DataFrame):
        """
        Converts dictionary of professional data into desired form
        :param pro_dict: dictionary of professional data
        """
        # get DataFrame from dict
        pro_df = pd.DataFrame.from_dict(pro_dict)
        pros = pro_df['professionals_id'].values

        # create professional-tag tuples
        tuples = []
        for i, tags in enumerate(pro_df['professionals_subscribed_tags'].values):
            pro = pros[i]
            for tag in tags.split(' '):
                tuples.append((pro, tag))

        # create DataFrame from tuples
        pro_tags = pd.DataFrame(tuples, columns=['tag_users_user_id', 'tags_tag_name'])
        pro_df.drop(columns='professionals_subscribed_tags', inplace=True)

        return pro_df, pro_tags
