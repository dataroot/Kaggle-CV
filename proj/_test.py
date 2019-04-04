import pandas as pd
from processing import StuProc


def drive(data_path: str, dump_path: str):
    que = pd.read_csv(data_path + 'questions.csv')
    ans = pd.read_csv(data_path + 'answers.csv')
    stu = pd.read_csv(data_path + 'students.csv')

    sp = StuProc(oblige_fit=False, path=dump_path)
    transformed = sp.transform(que, ans, stu)

    with pd.option_context('display.max_columns', 100):
        print(transformed)


if __name__ == '__main__':
    drive('../../data/', 'dump/')
