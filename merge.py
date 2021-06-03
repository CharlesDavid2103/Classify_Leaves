import glob
import os

import datasets
import pandas as pd

'''
{
    imge:{class1:1,class2:1}
}
'''


def main():
    count_dict = {}
    files = glob.glob('./submit_csv/*.csv')
    preds = []
    # print(files)
    for path in files:
        print(path)
        csv = datasets.read_csv_labels(path)
        for k, v in csv.items():
            count_dict[k] = count_dict.get(k, {})
            count_dict[k][v] = count_dict[k].get(v, 0) + 1
    for k, v in count_dict.items():
        v = sorted(v.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        preds.append(v[0][0])

    df_submit = pd.read_csv('./data/sample_submission.csv')
    df_submit['label'] = preds
    df_submit.to_csv('submit.csv', index=None)
    # print(count_dict)


if __name__ == '__main__':
    count_dict = {}
    preds = []
    dirs = os.listdir('./save_weights')
    for dir in dirs:
        path = os.path.join('./save_weights', dir)
        if os.path.isdir(path):
            path = os.path.join(path, 'submit.csv')
            csv = datasets.read_csv_labels(path)
            for k, v in csv.items():
                count_dict[k] = count_dict.get(k, {})
                count_dict[k][v] = count_dict[k].get(v, 0) + 1
    for k, v in count_dict.items():
        v = sorted(v.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        preds.append(v[0][0])
        print(k, v)
    df_submit = pd.read_csv('./data/sample_submission.csv')
    df_submit['label'] = preds
    df_submit.to_csv('submit.csv', index=None)
    # print(count_dict)
