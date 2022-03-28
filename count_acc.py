import json
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,hamming_loss
import glob

def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


with open('test.json') as f:
  dev_data = json.load(f)

with open('train.json') as f:
  train_data = json.load(f)

all_labels = ['经济', '科技', '其他', '社会', '环境', '军事', '文化', '政治']


answers = []
for i in dev_data:
    answers.append(i['label'])


def count_acc(threshold1,threshold2):
    with open('test_results.json') as f:
        test_results = json.load(f)
    true_num = 0
    all_num = 0
    for num_i,i in enumerate(test_results):
        tmp = [0] * 8
        max_1 = max(i)
        max_1_index = i.index(max_1)
        i[max_1_index] = 0
        max_2 = max(i)
        max_2_index = i.index(max_2)

        if max_1 > threshold1:
            pred = [all_labels[max_1_index]]
            if set(pred) == set(answers[num_i]):
                true_num += 1
            all_num += 1
        else:
            if max_2 > threshold2 and (max_1 + max_2) > threshold1:
                pred = [all_labels[max_1_index], all_labels[max_2_index]]
                if set(pred) == set(answers[num_i]):
                    true_num += 1
                all_num += 1

    print(all_num,true_num,true_num/all_num)

if __name__ == '__main__':
    thresholds1 = [i/100 for i in range(90,100)]
    print(thresholds1)
    thresholds2 = [i/100 for i in range(20,41)]
    print(thresholds2)
    for threshold1 in thresholds1:
        for threshold2 in thresholds2:
            print(threshold1,threshold2)
            count_acc(threshold1,threshold2)
