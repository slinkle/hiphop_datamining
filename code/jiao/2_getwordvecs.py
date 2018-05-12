#!/usr/bin/env python
# -*- coding: utf-8  -*-

# 数据挖掘大作业：挖掘嘻哈歌曲的特点
# 组员：11732026 焦艳梅，11732024 周忠祥

# 提取歌曲每一行的词向量
# 先提取每个词的向量，再对一行所有的词向量取平均

#extract word feature vector from word vector model
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# ignore warning
import logging
import os.path
import codecs,sys
import numpy as np
import pandas as pd
import gensim

# return word vectors
def getWordVecs(wordList,model):
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        #print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')
    

# build article word vectors
def buildVecs(filename,model):
    fileVecs = []
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            logger.info("Start line: " + line)
            wordList = line.split(' ')
            vecs = getWordVecs(wordList,model)
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean
                #print vecsArray
                #sys.exit()
                fileVecs.append(vecsArray)
    return fileVecs   

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # load word2vec model
    fdir = 'data/'
    inp = 'model/' + 'wiki.zh.text.vector'
    model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
    
    posInput = buildVecs(fdir + 'happy_cut.txt',model)
    negInput = buildVecs(fdir + 'sad_cut.txt',model)

    # use 1 for positive sentiment， 0 for negative
    Y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))

    X = posInput[:]
    for neg in negInput:
        X.append(neg)
    X = np.array(X)

    # write in file   
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    data = pd.concat([df_y,df_x],axis = 1)
    #print data
    data.to_csv('vector/' + 'songs_data.csv')
    

    


