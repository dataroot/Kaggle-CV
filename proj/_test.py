import pandas as pd
from processing import StuProc, ProProc


def drive(data_path: str, dump_path: str):
    pro = pd.read_csv(data_path + 'professionals.csv')
    stu = pd.read_csv(data_path + 'students.csv')
    que = pd.read_csv(data_path + 'questions.csv')
    ans = pd.read_csv(data_path + 'answers.csv')

    # sp = StuProc(oblige_fit=False, path=dump_path)
    # transformed = sp.transform(stu, que, ans)

    pp = ProProc(oblige_fit=False, path=dump_path)
    transformed = pp.transform(pro, que, ans)

    pd.set_option('display.width', 640)

    with pd.option_context('display.max_columns', 100):
        print(transformed)


if __name__ == '__main__':
    drive('../../data/', 'dump/')
