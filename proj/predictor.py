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
        with open(path + 'pairs.pkl', 'rb') as file:
            pairs = pickle.load(file)

        # load datasets with preprocessed features
        store = pd.HDFStore(path + 'processed.h5', 'r')
        que_data = pd.read_hdf(store, 'que')
        stu_data = pd.read_hdf(store, 'stu')
        pro_data = pd.read_hdf(store, 'pro')

        que_dict = {row.values[0]: row.values[2:] for i, row in que_data.iterrows()}
        self.stu_dict = {stu: group.values[-1, 2:] for stu, group in stu_data.groupby('students_id')}
        pro_dict = {pro: group.values[-1, 2:] for pro, group in pro_data.groupby('professionals_id')}

        que_feat, que_ids, pro_feat, pro_ids = [], [], [], []

        ques_stus = {(que, stu) for que, stu, pro, t in pairs}
        pros = {pro for que, stu, pro, t in pairs}

        for que, stu in ques_stus:
            que_feat.append(np.hstack([que_dict[que], self.stu_dict[stu]]))
            que_ids.append(que)

        for pro in pros:
            pro_feat.append(pro_dict[pro])
            pro_ids.append(pro)

        pro_feat = np.vstack(pro_feat)
        self.pros_ids = np.vstack(pro_ids)

        que_feat = np.vstack(que_feat)
        self.que_ids = np.vstack(que_ids)

        # create two encoders
        self.que_model = model.que_model
        self.pro_model = model.pro_model

        # compute latent vectors for questions and professionals
        que_lat_vecs = self.que_model.predict(que_feat)
        pro_lat_vecs = self.pro_model.predict(pro_feat)

        # create two KNN trees consisting of question and professional latent vectors
        self.que_tree = KDTree(que_lat_vecs)
        self.pro_tree = KDTree(pro_lat_vecs)

        # initialize QueProc and ProProc
        self.que_proc = QueProc(oblige_fit=False, path=path)
        self.pro_proc = ProProc(oblige_fit=False, path=path)

    def find_pros_by_que(self, que_df: pd.DataFrame, que_tags: pd.DataFrame, top: int = 10) -> pd.DataFrame:
        """
        Returns top professionals for given questions
        :param que_df: DataFrame of question data
        :param que_tags: DataFrame of question tags
        :param top: how many top professionals to return
        :param expand: whether to add professional data to returned DataFrame
        """
        que_df['questions_date_added'] = pd.to_datetime(que_df['questions_date_added'])

        # prepare question features and add them to student features
        que_feat = self.que_proc.transform(que_df, que_tags).values[:, 2:]
        stu_feat = np.vstack([self.stu_dict[stu] for stu in que_df['questions_author_id']])

        # print(stu_feat, que_feat)

        que_feat = np.hstack([stu_feat, que_feat])

        # get top professionals for questions
        que_lat_vecs = self.que_model.predict(que_feat)
        # print(que_lat_vecs)

        dists, pros = self.pro_tree.query(que_lat_vecs, k=top)
        pros = self.pros_ids[pros]
        scores = np.exp(-dists)
        ques = que_df['questions_id'].values

        # create question-professional-score tuples
        tuples = []
        for i, que in enumerate(ques):
            for j, pro in enumerate(pros[i]):
                tuples.append((que, pro, scores[i, j]))

        # create DataFrame from tuples
        score_df = pd.DataFrame(tuples, columns=['questions_id', 'professionals_id', 'professionals_score'])

        return score_df

    def find_ques_by_que(self, que_df: pd.DataFrame, que_tags: pd.DataFrame, top: int = 10) -> pd.DataFrame:
        """
        Returns top similar questions for given questions
        :param que_df: DataFrame of question data
        :param que_tags: DataFrame of question tags
        :param top: how many top professionals to return
        :param expand: whether to add professional data to returned DataFrame
        """
        que_df['questions_date_added'] = pd.to_datetime(que_df['questions_date_added'])

        # prepare student features

        # prepare question features and add them to student features
        que_feat = self.que_proc.transform(que_df, que_tags).values[:, 2:]
        stu_feat = np.vstack([self.stu_dict[stu] for stu in que_df['questions_author_id']])

        # print(stu_feat, que_feat)

        que_feat = np.hstack([stu_feat, que_feat])

        # get top similar questions for initial questions
        que_lat_vecs = self.que_model.predict(que_feat)
        # print(que_lat_vecs)

        dists, sim_ques = self.que_tree.query(que_lat_vecs, k=top)
        sim_ques = self.que_ids[sim_ques]
        scores = np.exp(-dists)
        ques = que_df['questions_id'].values

        # create question-similar_question-score tuples
        tuples = []
        for i, que in enumerate(ques):
            for j, sim_que in enumerate(sim_ques[i]):
                tuples.append((que, sim_que, scores[i, j]))

        # create DataFrame from tuples
        score_df = pd.DataFrame(tuples, columns=['initial_questions_id', 'questions_id', 'questions_score'])

        return score_df

    # TODO: consider professionals which are already in base

    def find_ques_by_pro(self, pro_df: pd.DataFrame, que_df: pd.DataFrame, ans_df: pd.DataFrame,
                         pro_tags: pd.DataFrame, top: int = 10) -> pd.DataFrame:
        """
        Returns top questions for given professionals
        :param pro_df: DataFrame of professional data
        :param que_df:
        :param ans_df:
        :param pro_tags: DataFrame of professional subscribed tags
        :param top: how many top professionals to return
        :param expand: whether to add professional data to returned DataFrame
        """
        pro_df['professionals_date_joined'] = pd.to_datetime(pro_df['professionals_date_joined'])
        que_df['questions_date_added'] = pd.to_datetime(que_df['questions_date_added'])
        ans_df['answers_date_added'] = pd.to_datetime(que_df['questions_date_added'])

        # prepare professional features
        pro_feat = self.pro_proc.transform(pro_df, que_df, ans_df, pro_tags).values[:, 2:]

        # get top questions for professionals
        pro_lat_vecs = self.pro_model.predict(pro_feat)
        dists, ques = self.que_tree.query(pro_lat_vecs, k=top)
        ques = self.que_ids[ques]
        scores = np.exp(-dists)
        pros = pro_df['professionals_id'].values

        # create professional-question-score tuples
        tuples = []
        for i, pro in enumerate(pros):
            for j, que in enumerate(ques[i]):
                tuples.append((pro, que, scores[i, j]))

        # create DataFrame from tuples
        score_df = pd.DataFrame(tuples, columns=['professionals_id', 'questions_id', 'questions_score'])

        return score_df

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
