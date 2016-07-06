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
    return docList,classList

def load_data2():
    docList = []
    classList = []
    files1 = os.listdir('review_polarity/txt_sentoken/neg/')
    files2 = os.listdir('review_polarity/txt_sentoken/pos/')
    num = len(files1)
    for i in range(num):
        wordList = textParse(open('review_polarity/txt_sentoken/neg/' + files1[i]).read())
        docList.append(wordList)
        classList.append('neg')
        wordList = textParse(open('review_polarity/txt_sentoken/pos/' + files2[i]).read())
        docList.append(wordList)
        classList.append('pos')
    return docList,classList

def spamTest():
    docList, classList = load_data2()
    num = len(docList)
    vocabList = createVocabList(docList)
    trainingSet = range(num)
    trainMat = []
    rate = 0.4
    for docIndex in trainingSet[:int((1 - rate) * num)]:
        temp = setOfWords2Vec(vocabList, docList[docIndex])
        temp.append(classList[docIndex])
        trainMat.append(temp)
    labels0 = vocabList[:]
    myTree = createTree(trainMat, labels0)
    print myTree
    storeTree(myTree, 'classifierStorage.txt')
    grabTree('classifierStorage.txt')

    testMat = []
    for docIndex in trainingSet[int((1-rate) * num):]:
        temp = setOfWords2Vec(vocabList, docList[docIndex])
        temp.append(classList[docIndex])
        testMat.append(temp)
    error_count = 0
    for i in range(int(rate * num)):
        if classify(myTree, vocabList, testMat[i]) != classList[i + int((1 - rate) * num)]:
            error_count += 1
    print "error rate: %f" % (float(error_count) / (rate * num))
    createPlot(myTree)


def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


spamTest()
