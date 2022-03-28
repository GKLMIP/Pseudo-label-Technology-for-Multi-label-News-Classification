import json
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,hamming_loss
import glob
import tqdm
from multiprocessing import Pool
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

with open('train_vector.json') as f:
  train_vector = json.load(f)

with open('test_vector.json') as f:
  test_vector = json.load(f)
print(len(test_vector))

def creat_sim_data(test_vector_):
    tmp = []
    for j in train_vector:
        tmp.append(cosine_similarity(test_vector_, j))
    max_ = max(tmp)
    labels = train_data[tmp.index(max_)]['label']
    return [labels,max_]

if __name__ == '__main__':
    preds = []
    with Pool(12) as p:
        res = list(tqdm.tqdm(p.imap(creat_sim_data, test_vector), total=len(test_vector)))

    for num in range(len(test_vector)):
        preds.append(
            [
                res[num][0],answers[num],res[num][1]
            ]
        )
    with open('xiangsidu.json','w') as f:
        json.dump(preds,f)
