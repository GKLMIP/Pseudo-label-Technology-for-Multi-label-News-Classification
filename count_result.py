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

with open('test_vector.json') as f:
  test_vector = json.load(f)

preds = []
for filename in glob.glob('xiangsidu.json'):
    print(filename)
    with open(filename) as f:
        preds.extend(json.load(f))
print(len(preds))

def count_result():
    preds_ = []
    answers_ = []
    for num, i in enumerate(preds):
        pred_ = [0] * 8
        for _ in i[0]:
            pred_[all_labels.index(_)] = 1
        preds_.append(pred_)
        answer_ = [0] * 8

        for _ in i[1]:
            answer_[all_labels.index(_)] = 1
        answers_.append(answer_)
    answers_ = np.array(answers_)
    preds_ = np.array(preds_)
    print(f1_score(answers_, preds_, average="macro"))
    print(f1_score(answers_, preds_, average="micro"))
    print(hamming_loss(answers_, preds_))
count_result()





