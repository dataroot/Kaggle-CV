import pickle

import pandas as pd
from processors import QueProc, StuProc, ProProc


def drive(data_path: str, dump_path: str):
    pro = pd.read_csv(data_path + 'professionals.csv')
    stu = pd.read_csv(data_path + 'students.csv')
    que = pd.read_csv(data_path + 'questions.csv')
    ans = pd.read_csv(data_path + 'answers.csv')

    pd.set_option('display.width', 640)
    pd.set_option('display.max_columns', 100)

    tag_que = pd.read_csv(data_path + 'tag_questions.csv')
    tags = pd.read_csv(data_path + 'tags.csv').merge(tag_que, left_on='tags_tag_id', right_on='tag_questions_tag_id')

    qp = QueProc(oblige_fit=False, path=dump_path)
    que_data = qp.transform(que, tags)

    print(que_data)

    sp = StuProc(oblige_fit=False, path=dump_path)
    stu_data = sp.transform(stu, que, ans)

    print(stu_data)

    pp = ProProc(oblige_fit=False, path=dump_path)
    pro_data = pp.transform(pro, que, ans)

    print(pro_data)


if __name__ == '__main__':
    drive('../../data/', 'dump/')
