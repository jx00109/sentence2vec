# -*- coding:utf8 -*-
from gensim.models import KeyedVectors
import pickle as pkl
import numpy as np
from typing import List
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import os
import PSLvec as psl
from nltk.tokenize import StanfordTokenizer

word2vec_path = './GoogleNews-vectors-negative300.bin.gz'
glove_path = './glove_model.txt'
psl_path = './PSL_model.txt'
# traindata = './datasets/sts2013.OnWN.pkl'
freq_table = './mydictionary'
embedding_size = 300

pslemb = psl.PSL()

# 载入word2vec模型
# model = KeyedVectors.load_word2vec_format(word2vec_path,binary=True)
# model = KeyedVectors.load_word2vec_format(glove_path,binary=False)
# model = KeyedVectors.load_word2vec_format(psl_path,binary=False)
model = pslemb.w
print('完成模型载入')

tokenizer = StanfordTokenizer(path_to_jar=r"D:\stanford-parser-full-2016-10-31\stanford-parser.jar")


# print(type(model))
# print(model['sdfsfsdfsadfs'])

class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector


class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    def len(self) -> int:
        return len(self.word_list)


def get_word_frequency(word_text, looktable):
    if word_text in looktable:
        return looktable[word_text]
    else:
        return 1.0


def sentence_to_vec(sentence_list: List[Sentence], embedding_size, looktable, a=1e-3):
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        for word in sentence.word_list:
            a_value = a / (a + get_word_frequency(word.text, looktable))  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word.vector))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences

    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))

    return sentence_vecs


with open(freq_table, 'rb') as f:
    mydict = pkl.load(f)
print('完成词频字典载入')

paths = ['./datasets/data']
for path in paths:
    files = []
    for file in os.listdir(path=path):
        if os.path.isfile(path + '/' + file):
            files.append(path + '/' + file)

    for traindata in files:
        with open(traindata, 'rb') as f:
            train = pkl.load(f)

        print('读取' + traindata + '数据完成')

        gs = []
        pred = []
        allsent = []
        for each in train:
            # sent1, sent2, label = each.split('\t')
            if len(train[0]) == 3:
                sent1, sent2, label = each
            else:
                sent1, sent2, label, _ = each
            gs.append(float(label))
            s1 = []
            s2 = []
            # sw1 = sent1.split()
            # sw2 = sent2.split()
            for word in sent1:
                try:
                    vec = model[word]
                except KeyError:
                    vec = np.zeros(embedding_size)
                s1.append(Word(word, vec))
            for word in sent2:
                try:
                    vec = model[word]
                except KeyError:
                    vec = np.zeros(embedding_size)
                s2.append(Word(word, vec))

            ss1 = Sentence(s1)
            ss2 = Sentence(s2)
            allsent.append(ss1)
            allsent.append(ss2)

        sentence_vectors = sentence_to_vec(allsent, embedding_size, looktable=mydict)
        len_sentences = len(sentence_vectors)
        for i in range(len_sentences):
            if i % 2 == 0:
                sim = cosine_similarity([sentence_vectors[i]], [sentence_vectors[i + 1]])
                pred.append(sim[0][0])

        print('len of pred: ', len(pred))
        print('len of gs: ', len(gs))

        r, p = pearsonr(pred, gs)
        print(traindata + '皮尔逊相关系数:', r)


        # sentence_vectors = sentence_to_vec([ss1, ss2], embedding_size, looktable=mydict)
        # sim = cosine_similarity([sentence_vectors[0]], [sentence_vectors[1]])
        # pred.append(sim[0][0])

        # r, p = pearsonr(pred, gs)
        # print(traindata + '皮尔逊相关系数:', r)  # print(sentence_vectors[0])
# print(sentence_vectors[1])
