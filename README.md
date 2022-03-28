# Pseudo-label-Technology-for-Multi-label-News-Classification


## Citation

If you use our corpus, please consider citing our paper:
```
@InProceedings{10.1007/978-3-030-86331-9_44,
author="Wang, Lianxi
and Lin, Xiaotian
and Lin, Nankai",
editor="Llad{\'o}s, Josep
and Lopresti, Daniel
and Uchida, Seiichi",
title="Research on Pseudo-label Technology for Multi-label News Classification",
booktitle="Document Analysis and Recognition -- ICDAR 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="683--698",
abstract="Multi-label news classification exerts a significant importance with the growing size of news containing multiple semantics. However, most of the existing multi-label classification methods rely on large-scale labeled corpus while publicly available resources for multi-label classification are limited. Although many researches have proposed the application of pseudo-label technology to expand the corpus, few studies explored it for multi-label classification since the number of labels is not prone to determine. To address these problems, we construct a multi-label news classification corpus for Indonesian language and propose a new multi-label news classification framework through using pseudo-label technology in this paper. The framework employs the BERT model as a pre-trained language model to obtain the sentence representation of the texts. Furthermore, the cosine similarity algorithm is utilized to match the text labels. On the basis of matching text labels with similarity algorithms, a pseudo-label technology is used to pick up the classes for unlabeled data. Then, we screen high-confidence pseudo-label corpus to train the model together with original training data, and we also introduce loss weights including class weight adjustment method and pseudo-label loss function balance coefficient to solve the problem of data with class imbalance, as well as reduce the impact of the quantity difference between labeled texts and pseudo-label texts on model training. Experiment results demonstrate that the framework proposed in this paper has achieved significant performance in Indonesian multi-label news classification, and each strategy can perform a certain improvement.",
isbn="978-3-030-86331-9"
}
