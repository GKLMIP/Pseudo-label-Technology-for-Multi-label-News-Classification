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

with open('train_vector20201230.json') as f:
  train_vector = json.load(f)

with open('test_vector6.0_20201230.json') as f:
  test_vector = json.load(f)

with open('test_results6.0_20201230.json') as f:
  test_results = json.load(f)

preds = []
for filename in glob.glob('xiangsidu6.0_20201230.json'):
    print(filename)
    with open(filename) as f:
        preds.extend(json.load(f))
print(len(preds))

def count_acc(threshold):
    all_num = 0
    true_num = 0
    for num,i in enumerate(preds):

        if i[2] > threshold and len(i[1]) == 1:
            if i[1] == i[0]:
                true_num += 1
            all_num += 1
        if i[2] > threshold and len(i[1]) == 2:
            if i[1][0] in i[0] and i[1][1] in i[0]:
                true_num += 1
            all_num += 1
    print(threshold,all_num,true_num,true_num/all_num)

if __name__ == '__main__':
    thresholds1 = [i/100 for i in range(90,100)]
    print(thresholds1)
    for threshold1 in thresholds1:
        count_acc(threshold1)

# 0.9 1757 1367 0.778030734206033
# 0.91 1735 1360 0.7838616714697406
# 0.92 1702 1346 0.790834312573443
# 0.93 1662 1326 0.7978339350180506
# 0.94 1604 1297 0.8086034912718204
# 0.95 1523 1253 0.8227183191070256
# 0.96 1422 1195 0.840365682137834
# 0.97 1250 1099 0.8792
# 0.98 942 878 0.9320594479830149
# 0.99 464 458 0.9870689655172413
