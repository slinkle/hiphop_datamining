# SortSong.py
#!/usr/bin/python  
# -*- coding:utf-8 -*-  

import sys
reload(sys)

sys.setdefaultencoding('utf-8')

import jieba
import jieba.analyse
import xlwt 
import codecs
import os,os.path
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.misc import imread
import logging
import os,os.path
import sys,string,re
from langconv import *

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
        line = re.sub("[0-9]","",line)
        #remove all the English and Chinese symbols
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
        if (word >= u'\u0041' and word<=u'\u005a') or (word >= u'\u0061' and word<=u'\u007a'):
            if (word.lower() not in stopkey):
                if word != '\t':
                    sentence += word + " "
        if (u'\u4e00' <= word<=u'\u9fff'):
            if (word not in stopkey):
                if word != '\t':
                    sentence += word + " "
    return sentence.strip()

if __name__=="__main__":

    wbk = xlwt.Workbook(encoding = 'ascii')
    sheet = wbk.add_sheet("wordCount")
    word_lst = ''
    key_list=[]
    jieba.analyse.set_stop_words('data/stopWord.txt')

    fdir = 'data/'

    stopkey = [w.strip() for w in codecs.open(fdir + 'stopWord.txt', 'r', encoding='utf-8').readlines()]


    is_chinese = True
    if is_chinese:
        prefix = 'chinese'
    else:
        prefix = 'english'
    # hip_dir = 'data/%s_hiphop/less'%prefix
    # pro_dir = 'data/%s_hiphop/property'%prefix

    # dict_dir = pro_dir + '/' + 'car_ca.txt'
    # allow_pos = ('ca',)

    # dict_dir = pro_dir + '/' + 'city_ci.txt'
    # allow_pos = ('ci',)

    # dict_dir = pro_dir + '/' + 'drink_dr.txt'
    # allow_pos = ('dr',)

    # dict_dir = pro_dir + '/' + 'world_wo.txt'
    # allow_pos = ('wo',)

    # dict_dir = pro_dir + '/' + 'relation_re.txt'
    # allow_pos = ('re',)

    # jieba.load_userdict(dict_dir)

    # for parent,dirnames,filenames in os.walk(hip_dir):
    #     for filename in filenames:
    #         if os.path.splitext(filename)[1] == '.txt':
    #             sourceFile = hip_dir + '/' + filename
    #             print 'processing ',filename,' ...'
    #             sourcef = codecs.open(sourceFile, 'r', encoding='utf-8')
    #             line = sourcef.readline()
    #             count = 1
    #             while line:
    #                 print 'line ',count
    #                 count += 1
    #                 line = clearTxt(line)
    #                 simp_line = Traditional2Simplified(line)
    #                 seg_line = sent2word(simp_line)
    #                 del_stop_line = delstopword(seg_line,stopkey)
    #                 print del_stop_line
    #                 if del_stop_line:
    #                 	word_lst += del_stop_line + ' '
    #                 	# print word_lst
    #                     # print 'size:',len(t),'word:',t[0],'weight:',str(np.int(t[1]*100))+'%'
    #                 line = sourcef.readline()
    #             sourcef.close()

    # with open("result/wordlist.txt",'w') as wlst: 
    #     wlst.write(word_lst)
    # wlst.close()
    with codecs.open("data/huahua/wordlist.txt",'r',encoding='utf-8') as f:    
       word_lst = f.read()
    f.close()
    # tags = jieba.analyse.extract_tags(word_lst,topK = 500, withWeight = True, allowPOS=allow_pos)
    tags = jieba.analyse.extract_tags(word_lst,topK = 100, withWeight = True)
    # print tags
    word_dict= {}
    weight_scale = 100
    with open("data/huahua/wordCount.txt",'w') as wf2: 
        for item in tags:
            if item[0] not in word_dict: 
                word_dict[item[0]] = np.int(item[1]*weight_scale)
            else:
                word_dict[item[0]] += np.int(item[1]*weight_scale)

        color_mask = imread('image/back.jpg')
        wordcloud = WordCloud(
            font_path = 'SimHei.ttf',
            background_color = 'white',
            mask = color_mask
            )
        wordcloud.fit_words(word_dict)
        print len(word_dict)
        print len(word_dict.keys())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(wordcloud)
        plt.axis('off')
        plt.show()
        fig.savefig('image/huahua22.png',pad = 0, bbox_inches = 'tight')
        plt.close()

        orderList=list(word_dict.values())
        print len(orderList)
        orderList.sort(reverse=True)
        print len(orderList)
        # print orderList
        for i in range(len(orderList)):
            for key in word_dict.keys():
                if word_dict[key]==orderList[i]:
                    wf2.write(key+' '+str(word_dict[key])+'\n') 
                    key_list.append(key)
                    word_dict[key]=0
    
    print len(key_list)
    print len(orderList)
    for i in range(len(key_list)):
        sheet.write(i, 1, label = orderList[i])
        sheet.write(i, 0, label = key_list[i])
    wbk.save('data/huahua/wordCount.xls') 

    

