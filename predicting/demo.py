import pickle
import pandas as pd

from models.distance import DistanceModel
from predicting.predictor import Predictor, Formatter
from utils.utils import TextProcessor

pd.set_option('display.max_columns', 100, 'display.width', 1024)

if __name__ == '__main__':
    tp = TextProcessor()

    model = DistanceModel(que_dim=34 - 2 + 8 - 2,
                          que_input_embs=[102, 42], que_output_embs=[2, 2],
                          pro_dim=42 - 2,
                          pro_input_embs=[102, 102, 42], pro_output_embs=[2, 2, 2],
                          inter_dim=20, output_dim=10)
    model.load_weights('../training/model.h5')

    with open('../training/dump.pkl', 'rb') as file:
        d = pickle.load(file)
        que_data = d['que_data']
        stu_data = d['stu_data']
        pro_data = d['pro_data']
        que_proc = d['que_proc']
        pro_proc = d['pro_proc']
        que_to_stu = d['que_to_stu']
        pos_pairs = d['pos_pairs']
    pred = Predictor(model, que_data, stu_data, pro_data, que_proc, pro_proc, que_to_stu, pos_pairs)

    formatter = Formatter('../../data/')

    # From ques

    que_dict = {
        'questions_id': ['0'],
        'questions_author_id': ['02946e467bab4fd794e42f9670cb4279'],
        'questions_date_added': ['2017-07-29 13:30:50 UTC+0000'],
        'questions_title': [
            'I want to study law but not sure what subjects need to be taken,so need some advice on that.'],
        'questions_body': ['#law-practice #lawyer #career-details'],
        'questions_tags': ['lawyer law-practice career-details']
    }

    que_df, que_tags = Formatter.convert_que_dict(que_dict)

    tmp = pred.find_pros_by_que(que_df, que_tags)
    print(formatter.get_pro(tmp))

    tmp = pred.find_ques_by_que(que_df, que_tags)
    print(formatter.get_que(tmp))

    # From pros

    ans_df = pd.read_csv('../../data/answers.csv')
    que_df = pd.read_csv('../../data/questions.csv')
    que_df['questions_title'] = que_df['questions_title'].apply(tp.process)
    que_df['questions_body'] = que_df['questions_body'].apply(tp.process)
    ans_df['answers_body'] = ans_df['answers_body'].apply(tp.process)
    que_df['questions_whole'] = que_df['questions_title'] + ' ' + que_df['questions_body']

    pro_dict = {'professionals_id': ['eae09bbc30e34f008e10d5aa70d521b2'],
                'professionals_location': ['Narberth, Pennsylvania'],
                'professionals_industry': ['Veterinary'],
                'professionals_headline': ['Veterinarian'],
                'professionals_date_joined': ['2017-09-26 18:10:23 UTC+0000'],
                'professionals_subscribed_tags': ['veterinary animal']}

    pro_df, pro_tags = Formatter.convert_pro_dict(pro_dict)

    tmp = pred.find_ques_by_pro(pro_df, que_df, ans_df, pro_tags)
    print(formatter.get_que(tmp))

    tmp = pred.find_pros_by_pro(pro_df, que_df, ans_df, pro_tags)
    print(formatter.get_pro(tmp))
