#!/usr/bin/env python
# -*- coding: utf-8  -*-

# 数据挖掘大作业：挖掘嘻哈歌曲的特点
# 组员：11732026 焦艳梅，11732024 周忠祥

# 读取歌词文件并进行清洗和分词
# 其中对于情绪分析，按照不同的提取词向量方法又分为：
## 1.按行存储读取到的所有歌词文件
## 2.按歌曲存储读取到的所有歌词文件
# 对于火热度预测，只有按照整首歌曲提取词向量
## 处理歌词文件的同时，处理对应评论数的离散程度划分

#cut sentences in songs to words with jieba
import logging
import os,os.path
import jieba
import jieba.analyse
import codecs,sys,string,re
from langconv import *

# cut sentences to words line by line or song by song
def prepareSong(fdir,lyric_dir,commnets_dir,foldername,linebyline,getComments,stopkey,delkey):
    if linebyline:
        targetFile = fdir + foldername +'_cut.txt' 
    else:
        targetFile = fdir + foldername +'song_cut.txt' 
    targetf = codecs.open(targetFile, 'w', encoding='utf-8')
    lydir = lyric_dir + foldername
    codir = commnets_dir + foldername
    if getComments:
        comments_targetFile = fdir + foldername + '_comments.txt'
        comments_targetf = codecs.open(comments_targetFile, 'w', encoding='utf-8')

    #three params：1.parent dir 2.all folder name（not include path） 3.all file name
    for parent,dirnames,filenames in os.walk(lydir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.txt':
                sourceFile = lydir + '/' + filename
                sourcef = codecs.open(sourceFile, 'r', encoding='utf-8')
                line = sourcef.readline()
                blank_file = True
                while line:
                    line = clearTxt(line)
                    simp_line = Traditional2Simplified(line)
                    seg_line = sent2word(simp_line)
                    del_stop_line = delstopword(seg_line,stopkey)
                    if keepsentence(del_stop_line,delkey):
                        blank_file = False
                        if linebyline:
                            targetf.writelines(del_stop_line + '\n')
                        else:
                            targetf.writelines(del_stop_line + ' ')
                    line = sourcef.readline()
                # print filename,' is blank file or not :',blank_file
                if not blank_file:
                    if not linebyline:
                        targetf.writelines('\n') 
                    if getComments:
                        comments_sourceFile = codir + '/' + filename
                        comments_sourcef = codecs.open(comments_sourceFile, 'r', encoding='utf-8')
                        comments = int(comments_sourcef.readline().strip())
                        comments_sourcef.close()
                        if comments < 150:
                            degree = 0 # cold
                        elif comments < 1000:
                            degree = 1 # natural
                        elif comments < 4000:
                            degree = 2 # hot
                        else:
                            degree = 3 # very hot
                        comments_targetf.writelines(str(degree).encode('utf-8') + '\n')
                sourcef.close()       
    targetf.close()
    if getComments:
        comments_targetf.close()

# cut sentences to words line by line
def prepareData(sourceFile,targetFile,stopkey,delkey):
    sourcef = codecs.open(sourceFile, 'r', encoding='utf-8')
    targetf = codecs.open(targetFile, 'w', encoding='utf-8')
    print 'open source file: '+ sourceFile
    print 'open target file: '+ targetFile

    lineNum = 1
    line = sourcef.readline()
    while line:
        print '---processing ',lineNum,' article---'
        line = clearTxt(line)
        simp_line = Traditional2Simplified(line)
        seg_line = sent2word(simp_line)
        del_stop_line = delstopword(seg_line,stopkey)
        if keepsentence(del_stop_line,delkey):
            targetf.writelines(del_stop_line + '\n')       
        lineNum = lineNum + 1
        line = sourcef.readline()
    print 'well done.'
    sourcef.close()
    targetf.close()

def Traditional2Simplified(sentence):
    '''
    transform the traditional word in sentence to simplified word
    :param sentence: the sentence to be transformed
    :return: the simplified sentence
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

# data clear
def clearTxt(line):
    if line != '': 
        line = line.strip()
        intab = ""
        outtab = ""
        trantab = string.maketrans(intab, outtab)
        #remove all the punctuation and digits
        pun_num = string.punctuation + string.digits
        line = line.encode('utf-8')
        line = line.translate(trantab,pun_num)
        line = line.decode("utf8")
        #remove all the English and digits
        line = re.sub("[a-zA-Z0-9]","",line)
        #remove all the English and Chinese symbols
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+".decode("utf-8"), "",line) 
    return line

#cut sentences
def sent2word(line):
    segList = jieba.cut(line,cut_all=False)    
    segSentence = ''
    for word in segList:
        if word != '\t':
            segSentence += word + " "
    return segSentence.strip()

#delete the stopword and the non-chinese word 
def delstopword(line,stopkey):
    wordList = line.split(' ')          
    sentence = ''
    for word in wordList:
        word = word.strip()
        if (word not in stopkey) and (u'\u4e00' <= word<=u'\u9fff'):
            if word != '\t':
                sentence += word + " "
    return sentence.strip()
def keepsentence(line,delkey):
    # if the line is blank or has the stopsentence word ,return false, meaning delete this line
    if line:
        wordList = line.split(' ')
        for word in wordList:
            word = word.strip()
            if word in delkey:
                return False # delete the stop sentence
        return True
    else:
        return False # delete the blank line


if __name__ == '__main__':   
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    fdir = 'data/'

    stopkey = [w.strip() for w in codecs.open(fdir + 'stopWord.txt', 'r', encoding='utf-8').readlines()]
    delkey = [w.strip() for w in codecs.open(fdir + 'stopSentence.txt', 'r', encoding='utf-8').readlines()]


    lyric_dir = fdir + 'Lyric/' 
    commnets_dir = fdir + 'Comments/'
    
    # for sentiment analysis
    # folders = ['happy','sad']
    # linebyline = False
    # getComments = False
    # for foldername in folders:
    #     logger.info("running "+ foldername +" files.")

    #     prepareSong(fdir,lyric_dir,commnets_dir,foldername,linebyline,getComments,stopkey,delkey)
    
    #     logger.info(foldername +" files well done.")
 
    # for popularity prediction
    folders = ['hiphop']
    linebyline = False
    getComments = True
    for foldername in folders:
        logger.info("running "+ foldername +" files.")

        prepareSong(fdir,lyric_dir,commnets_dir,foldername,linebyline,getComments,stopkey,delkey)
    
        logger.info(foldername +" files well done.")
    
