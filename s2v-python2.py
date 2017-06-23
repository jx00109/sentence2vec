# -*- encoding:utf8 -*-
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from gensim.models.word2vec import KeyedVectors
from sklearn.decomposition import TruncatedSVD

single_data_path = './data/images2014'
vocab_path = './others/enwiki_vocab_min200.txt'
embed_domins = 300
rmpc = True

# 是否测试程序
isTest = False


# 获取单词对应权重
def getWordsFrequency(weightfile, a=1e-3):
    if a <= 0:  # when the parameter makes no sense, use unweighted
        a = 1.0
    # 单词权重字典，key为单词，value是权重
    word2weight = {}
    # N记录词总数
    N = 0.0
    with open(weightfile, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                line = line.split()
                if len(line) == 2:
                    word2weight[line[0]] = float(line[1])
                    N += float(line[1])
                else:
                    print line

    for key, value in word2weight.iteritems():
        word2weight[key] = a / (a + value / N)
    return word2weight


# 返回输入句子list对应的句子向量表
def getSenteceEmbedding(list_sentences, words_dict, weights_dict, istest):
    embeds = []
    for s in list_sentences:
        semb = []
        s = s.strip()
        words = s.split()
        for word in words:
            if istest:
                if word in words_dict.keys():
                    embed = words_dict[word]
                else:
                    embed = np.zeros(embed_domins, dtype=float)
            else:
                if word in words_dict.vocab:
                    embed = words_dict[word]
                else:
                    embed = np.zeros(embed_domins, dtype=float)
            if word in weights_dict:
                w = weights_dict[word]
            else:
                w = 1.0
            semb.append(embed * w)
        emb = np.sum(semb, axis=0) / len(semb)
        embeds.append(emb)
    return np.array(embeds, dtype=float)


def getCosineSimilarities(emb1, emb2):
    return cosine_similarity(emb1, emb2)


def compute_pc(X, npc=1):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


if isTest:
    words_dict = {'I': np.array([1., 1., 1., 1.], dtype=float),
                  'am': np.array([2., 2., 2., 2.], dtype=float),
                  'You': np.array([0.9, 0.9, 0.9, 0.9], dtype=float),
                  'Today': np.array([10.0, 9.0, 8.0, 7.0], dtype=float),
                  'boy': np.array([4.5, 3., 5., 6.], dtype=float),
                  'girl': np.array([4.4, 3.1, 5.2, 6.1], dtype=float)
                  }
else:
    print '读取词向量...'
    # word2vec词向量字典
    words_dict = KeyedVectors.load_word2vec_format('./others/GoogleNews-vectors-negative300.bin.gz', binary=True)

    print '词向量读取完毕!'

if isTest:
    embed_domins = 4
    p1 = ['I am a boy . ', 'You are a girl . ', 'I like playing basketball . ']
    p2 = ['Today is a nice day .', 'Something will happen today . ', 'Do you love me ? ']
    scores = [0.5, 0.4, 0.3]
    weights_dict = {'am': 0.5}
else:
    p1 = []
    p2 = []
    scores = []
    with open(single_data_path, 'r') as f:
        for line in f:
            lines = line.split('\t')
            p1.append(lines[0])
            p2.append(lines[1])
            scores.append(float(lines[2]))
    weights_dict = getWordsFrequency(vocab_path)
sentenceEmbed1 = getSenteceEmbedding(p1, words_dict, weights_dict, isTest)
sentenceEmbed2 = getSenteceEmbedding(p2, words_dict, weights_dict, isTest)
print 'Type of s1: ', type(sentenceEmbed1)
# print sentenceEmbed1
print 'Type of s2: ', type(sentenceEmbed2)
# print sentenceEmbed2
if rmpc:
    sentenceEmbed1 = remove_pc(sentenceEmbed1, npc=1)
    sentenceEmbed2 = remove_pc(sentenceEmbed2, npc=1)
    print '完成主成分移除...'
    # print sentenceEmbed1
    # print sentenceEmbed2
sims = []
for i in range(len(sentenceEmbed1)):
    sims.append(cosine_similarity([sentenceEmbed1[i]], [sentenceEmbed2[i]])[0][0])
# print sims
r, p = pearsonr(sims, scores)
# r1, p1 = pearsonr(scores, sims)
print r
