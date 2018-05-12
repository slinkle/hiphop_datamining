#!/usr/bin/python
# -*- coding: utf-8  -*-

# 数据挖掘大作业：挖掘嘻哈歌曲的特点
# 组员：11732026 焦艳梅，11732024 周忠祥

# 提取整首歌曲的词向量

import sys
import numpy as np
import pandas as pd
import gensim
import codecs

from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from sklearn.cross_validation import train_test_split

# TaggedDocument = gensim.models.doc2vec.TaggedDocument

#Gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识的label.
#我们使用Gensim自带的LabeledSentence方法. 标识的格式为"TRAIN_i"和"TEST_i"，其中i为序号
def labelizeReviews(reviews, label_type):
    # print 'labelizeReviews'
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(TaggedDocument(v, [label]))
    return labelized

##读取并预处理数据
def mood_dataset(happyfile,sadfile,happyhip,sadhip):
    with codecs.open(happyfile, 'r', encoding='utf-8') as infile:
        happy_songs = infile.readlines()
    with codecs.open(sadfile, 'r', encoding='utf-8') as infile:
        sad_songs = infile.readlines()
    with codecs.open(happyhip, 'r', encoding='utf-8') as infile:
        happy_hip = infile.readlines()
    with codecs.open(sadhip, 'r', encoding='utf-8') as infile:
        sad_hip = infile.readlines()

    y_train = np.concatenate((np.ones(len(happy_songs)), np.zeros(len(sad_songs))))
    y_test = np.concatenate((np.ones(len(happy_hip)), np.zeros(len(sad_hip))))
    x_train = np.concatenate((happy_songs, sad_songs))
    x_train = [z.split() for z in x_train]
    x_test = np.concatenate((happy_hip, sad_hip))
    x_test = [z.split() for z in x_test]

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')

    return x_train,x_test,y_train, y_test

def comments_dataset(songfile_train,commentsfile_train,songfile_test,commentsfile_test):
    with codecs.open(songfile_train, 'r', encoding='utf-8') as infile:
        x_train = infile.readlines()
        x_train = [z.split() for z in x_train]
    with codecs.open(commentsfile_train, 'r', encoding='utf-8') as infile:
        y_train = np.array(infile.readlines())
        y_train = [z.split() for z in y_train]
    with codecs.open(songfile_test, 'r', encoding='utf-8') as infile:
        x_test = infile.readlines()
        x_test = [z.split() for z in x_test]
    with codecs.open(commentsfile_test, 'r', encoding='utf-8') as infile:
        y_test = np.array(infile.readlines())
        y_test = [z.split() for z in y_test]
    
    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')

    return x_train,x_test,y_train, y_test


##读取向量
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

##对数据进行训练
def train(x_train,x_test,size = 400,epoch_num=10):
    #实例DM和DBOW模型
    model_dm = Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    #使用所有的数据建立词
    x = x_train + x_test
    # print 'before build_vocab'
    model_dm.build_vocab(x)
    model_dbow.build_vocab(x)
    # print 'after build_vocab'

    #进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    # print 'before epoch'
    # x_train = np.array(x_train)
    for epoch in range(epoch_num):
        print 'traning x_train epoch ',epoch
        np.random.shuffle(x_train)
        model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=model_dm.iter)
        model_dbow.train(x_train, total_examples=model_dbow.corpus_count, epochs=model_dbow.iter)

    #训练测试数据集
    # x_test = np.array(x_test)
    for epoch in range(epoch_num):
        print 'traning x_test epoch ',epoch
        np.random.shuffle(x_test)
        model_dm.train(x_test, total_examples=model_dm.corpus_count, epochs=model_dm.iter)
        model_dbow.train(x_test, total_examples=model_dbow.corpus_count, epochs=model_dbow.iter)

    return model_dm,model_dbow

##将训练完成的数据转换为vectors
def get_vectors(model_dm,model_dbow):

    #获取训练数据集的文档向量
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    #获取测试数据集的文档向量
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    return train_vecs,test_vecs

##使用分类器对文本向量进行分类训练
def Classifier(train_vecs,y_train,test_vecs, y_test):
    #使用sklearn的SGD分类器
    from sklearn.linear_model import SGDClassifier

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)

    print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)

    return lr

##绘出ROC曲线，并计算AUC
def ROC_curve(lr,test_vecs,y_test):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    pred_probas = lr.predict_proba(test_vecs)[:,1]

    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.show()

##运行模块
if __name__ == "__main__":
    
    size = 400
    epoch_num = 10
    
    is_mood = False

    fdir = 'data/'
    happyfile = fdir + 'happysong_cut.txt'
    sadfile = fdir + 'sadsong_cut.txt'
    happyhip = fdir + 'happyhipsong_cut.txt'
    sadhip = fdir + 'sadhipsong_cut.txt'

    # songfile_train = fdir + 'hiphopsong_train.txt'
    # commentsfile_train = fdir + 'comments_train.txt'
    # songfile_test = fdir + 'hiphopsong_test.txt'
    # commentsfile_test = fdir + 'comments_test.txt'
    songfile_train = fdir + 'hiphopsong_cut.txt'
    commentsfile_train = fdir + 'hiphop_comments.txt'
    songfile_test = fdir + 'hiphopsong_cut.txt'
    commentsfile_test = fdir + 'hiphop_comments.txt'

    model_dm_file = fdir + 'model/' + 'dm_400.model'
    model_dbow_file = fdir +'model/' + 'dbow_400.model'

    if is_mood:
        x_train,x_test,y_train,y_test = mood_dataset(happyfile,sadfile,happyhip,sadhip)
        suffix = 'mood'
    else:
        x_train,x_test,y_train,y_test = comments_dataset(songfile_train,commentsfile_train,songfile_test,commentsfile_test)
        suffix = 'comments'
    
    # train model 
    model_dm,model_dbow = train(x_train,x_test,size,epoch_num)

    model_dm.save(model_dm_file)
    model_dbow.save(model_dbow_file)
    print 'save model finished.'

    # # load pretrained model
    # model_dm = Doc2Vec.load(model_dm_file)
    # model_dbow = Doc2Vec.load(model_dbow_file)
    # print 'load model finished.'
    
    train_vecs,test_vecs = get_vectors(model_dm,model_dbow)

    # write in file   
    df_x = pd.DataFrame(train_vecs)
    df_y = pd.DataFrame(y_train)
    data_train = pd.concat([df_y,df_x],axis = 1)
    #print data
    data_train.to_csv(fdir +'vector/' + 'train_%s.csv'%suffix)

    # write in file   
    df_x = pd.DataFrame(test_vecs)
    df_y = pd.DataFrame(y_test)
    data_test = pd.concat([df_y,df_x],axis = 1)
    #print data
    data_test.to_csv(fdir +'vector/' + 'test_%s.csv'%suffix)
    print 'save data finished.'

