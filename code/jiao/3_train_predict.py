#!/usr/bin/env python
# -*- coding: utf-8  -*-

# 数据挖掘大作业：挖掘嘻哈歌曲的特点
# 组员：11732026 焦艳梅，11732024 周忠祥

# 训练不同的分类器并进行准确度测试

# PCA  SVM
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn import svm,datasets
from sklearn import metrics
from mytictoc import tic,toc
from scipy import interp
from itertools import cycle

# PCA reducing feature dimension
# compute the contribution rate

def pca_reducing(size,x):
    tic()
    pca = PCA(n_components=size)
    pca.fit(x)
    toc()
    #print pca.explained_variance_ratio_

    # PCA drawing
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.savefig(fdir + 'image/train_pca')
    # plt.show()
    plt.close()


    # get 100 dimensions according to the pca drawing
    x_pca = PCA(n_components = 200).fit_transform(x)

    return x_pca


# SVM (RBF)
# using training data with 100 dimensions
def SVM_Classifier(x_train,y_train,x_test,y_test):
    tic()
    clf = svm.SVC(C = 2, probability = True)
    clf.fit(x_train,y_train)
    toc()

    print 'Test Accuracy: %.2f'% clf.score(x_train,y_train)

    #Create ROC curve
    tic()
    pred_probas = clf.predict_proba(x_train)[:,1] #score
    toc()

    # y_train = np.array(y_train)
    print type(y_train)
    print type(pred_probas)
    fpr,tpr,_ = metrics.roc_curve(y_train, pred_probas)
    roc_auc = metrics.auc(fpr,tpr)
    plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc = 'lower right')
    plt.savefig(fdir + 'image/svm_roc')
    plt.show()
    plt.close()

# SVM (RBF)
# using training data with 100 dimensions
def SVM_MultiClassifier(x,y):
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = x.shape
    print 'n_samples = ',n_samples
    print 'n_features = ',n_features
    x = np.c_[x, random_state.randn(n_samples, 20)]
    print 'shuffle finished.'
    # shuffle and split training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print 'split finished.'

    print 'x_train shape',x_train.shape

    # for accuracy
    classifier = svm.SVC(C = 1.5, gamma = 0.1,kernel='rbf', probability=True, decision_function_shape='ovo')
    classifier.fit(x_train, y_train)
    print 'Test Accuracy: %.2f'% classifier.score(x_test,y_test)

    # for roc curve
    # Binarize the output
    y_train = label_binarize(y_train, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
    n_classes = y_test.shape[1]
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(C = 1.5, gamma = 0.1,kernel='rbf', probability=True, random_state=random_state))
    y_score = classifier.fit(x_train, y_train).decision_function(x_test)
    print 'get score.'

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(fdir + 'image/multi_svm_roc')
    plt.show()
    plt.close()


if __name__ == "__main__":

    # loading data [n rows x 800 columns]
    fdir = 'data/'
    is_mood = False
    if is_mood:
        suffix = 'mood'
    else:
        suffix = 'comments'
    df_train = pd.read_csv(fdir +'vector/' + 'train_%s2.csv'%suffix)
    y_train = df_train.iloc[:,1]
    x_train = df_train.iloc[:,2:]
    # df_test = pd.read_csv(fdir +'vector/' + 'test_%s.csv'%suffix)
    # y_test = df_test.iloc[:,1]
    # x_test = df_test.iloc[:,2:]

    # pca
    size = x_train.shape[1]
    print 'original feature dimensions is ',size
    x_train_pca = pca_reducing(size,x_train)
    print 'pca finished.'
    
    # classifier
    SVM_MultiClassifier(x_train_pca,y_train)
    # roc curve