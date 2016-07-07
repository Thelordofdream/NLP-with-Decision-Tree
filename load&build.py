import thread

import time
from numpy import *
import feedparser
import operator
from trees import *
from treePlotter import *
import os
from os import listdir


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
            # else:
            #    print "the word: %s is not in my Vocabulary!" % word
    return returnVec


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def load_data1():
    docList = []
    classList = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append('spam')
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append('ham')
    print 'number of papers is: %d' % len(docList)
    return docList,classList

def load_data2():
    docList = []
    classList = []
    fullText = []
    files1 = os.listdir('review_polarity/txt_sentoken/neg/')
    files2 = os.listdir('review_polarity/txt_sentoken/pos/')
    num = len(files1)
    for i in range(num):
        wordList = textParse(open('review_polarity/txt_sentoken/neg/' + files1[i]).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append('neg')
        wordList = textParse(open('review_polarity/txt_sentoken/pos/' + files2[i]).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append('pos')
    print 'number of papers is: %d' % len(docList)
    return docList,classList,fullText

def spamTest():
    print '===== Loading Data ====='
    docList, classList, fullText = load_data2()
    num = len(docList)
    print '== Building VocabList =='
    vocabList = createVocabList(docList)
    vocabList0 = vocabList[:]
    needs = len(vocabList)
    finished_thread = [0]
    try:
        thread.start_new_thread(DelLeastFreq, (vocabList0[:int(needs/4)], fullText, vocabList, finished_thread))
        thread.start_new_thread(DelLeastFreq, (vocabList0[int(needs/4):int(needs/2)], fullText, vocabList, finished_thread))
        thread.start_new_thread(DelLeastFreq, (vocabList0[int(needs/2):int(needs*3.0/4)], fullText, vocabList, finished_thread))
        thread.start_new_thread(DelLeastFreq, (vocabList0[int(needs*3.0/4):needs], fullText, vocabList, finished_thread))
    except:
        print "Error: unable to start thread"
    while finished_thread[0] < 4:
        time.sleep(10)
        pass
    count = 0
    fr = open('stopwords.txt')
    stopwords = [inst.strip() for inst in fr.readlines()]
    for eachword in stopwords:
        if eachword in vocabList:
            vocabList.remove(eachword)
            count += 1# delete stopwords
    print 'deleted %d words' % count
    print 'number of vocab is: %d' % len(vocabList)
    print '== Building TrainMat =='
    trainingSet = range(num)
    trainMat = []
    trainMat1 = []
    trainMat2 = []
    trainMat3 = []
    trainMat4 = []
    rate = 0.2
    needs = int((1 - rate) * num)
    finished_thread = [0]
    try:
        thread.start_new_thread(Words2Vec, (vocabList, docList, classList, trainingSet[:int(needs/4)], trainMat1, finished_thread))
        thread.start_new_thread(Words2Vec, (vocabList, docList, classList, trainingSet[int(needs/4):int(needs/2)], trainMat2, finished_thread))
        thread.start_new_thread(Words2Vec, (vocabList, docList, classList, trainingSet[int(needs/2):int(needs*3.0/4)], trainMat3, finished_thread))
        thread.start_new_thread(Words2Vec, (vocabList, docList, classList, trainingSet[int(needs*3.0/4):needs], trainMat4, finished_thread))
    except:
        print "Error: unable to start thread"
    while finished_thread[0] < 4:
        time.sleep(10)
        pass
    trainMat.extend(trainMat1)
    trainMat.extend(trainMat2)
    trainMat.extend(trainMat3)
    trainMat.extend(trainMat4)
    print 'number of train is: %d' % len(trainMat)
    print '==== Building Tree ===='
    labels0 = vocabList[:]
    myTree = createTree(trainMat, labels0)
    print myTree
    storeTree(myTree, 'classifierStorage.txt')
    print '=== Building TestMat =='
    testMat = []
    for docIndex in trainingSet[int((1-rate) * num):]:
        temp = setOfWords2Vec(vocabList, docList[docIndex])
        temp.append(classList[docIndex])
        testMat.append(temp)
    print '======= Testing ======='
    error_count = 0
    for i in range(int(rate * num)):
        if classify(myTree, vocabList, testMat[i]) != classList[i + int((1 - rate) * num)]:
            error_count += 1
    print "error rate: %f" % (float(error_count) / (rate * num))
    # myTree = grabTree('classifierStorage.txt')
    createPlot(myTree)

def Words2Vec(vocabList, docList, classList, trainingSet, trainMat, finished_thread):
    for docIndex in trainingSet:
        temp = setOfWords2Vec(vocabList, docList[docIndex])
        temp.append(classList[docIndex])
        trainMat.append(temp)
    finished_thread[0] += 1


def DelLeastFreq(vocabList0, fullText, vocabList, finished_thread):
    count = 0
    for token in vocabList0:
        if fullText.count(token) < 100:
            vocabList.remove(token)
            count += 1
    finished_thread[0] +=1
    print 'deleted %d words' % count


spamTest()
