from numpy import *
import feedparser
import operator
from trees import *
from treePlotter import *


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


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append('spam')
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append('ham')
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    trainMat = []
    test_num = 20
    for docIndex in trainingSet[:(50-test_num)]:
        temp = setOfWords2Vec(vocabList, docList[docIndex])
        temp.append(classList[docIndex])
        trainMat.append(temp)
    labels0 = vocabList[:]
    myTree = createTree(trainMat, labels0)
    print myTree
    storeTree(myTree, 'classifierStorage.txt')
    grabTree('classifierStorage.txt')

    testMat = []
    for docIndex in trainingSet[(50-test_num):]:
        temp = setOfWords2Vec(vocabList, docList[docIndex])
        temp.append(classList[docIndex])
        testMat.append(temp)
    error_count = 0
    for i in range(test_num):
        if classify(myTree, vocabList, testMat[i]) != classList[i + 50 - test_num]:
            error_count += 1
    print "error rate: %f" % (float(error_count) / test_num)
    createPlot(myTree)


def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


spamTest()
