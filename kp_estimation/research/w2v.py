# coding:utf8

import numpy as np
import gensim
from gensim.models import word2vec
import jieba
import pandas as pd
import re
regx = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"


def read_corpus(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = f.readlines()
        corpos = [(' '.join(jieba.cut(re.sub(regx, '', line[:-1])))).split() for line in lines if not line[0] == '#']
    return corpos


def read_csv(fname):
    data = pd.read_csv(fname, header=None, delimiter='\t', names=['content', 'label'])
    content = data['content'].values
    label = data['label'].values
    return content, label


def read_doc(fname):
    data = pd.read_csv(fname, header=None, delimiter='\t', names=['label', 'content'])
    label = data['label'].values
    content = data['content'].values
    return content, label


def w2v(corpus, dimention=200):
    model = gensim.models.Word2Vec(corpus, sg=1, hs=1,
                                   size=dimention, min_count=2,
                                   window=8, alpha=0.001)  # 训练词向量
    model.save('../model_w2v/cnews.bin')
    print('train done')


if __name__ == '__main__':
    content0, _ = read_doc('../../dataset/cnews.train.txt')
    content1, _ = read_doc('../../dataset/cnews.test.txt')
    content = np.hstack([content0, content1])
    print(content.shape)
    w2v(content)
