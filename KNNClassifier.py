'''
Created on Jun 27, 2016

@author: rajajosh
'''
import numpy
from scipy.spatial.distance import euclidean

class KNNClassifier(object):
    "K-Nearest Neighbors classifier class"
    len=0
    x_train=[]
    y_train=[]
    kVal=1
    clusters = set()
    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    def fit(self, x_train, y_train, kVal=3):
        "fir the training data. "
        self.len=len(x_train)
        if self.len>len(y_train): self.len=len(y_train)
        if kVal>self.len: kVal=self.len
        self.x_train=x_train
        self.y_train=y_train
        self.clusters = set(y_train)
        self.kVal=kVal

    def predict(self, x_test):
        retArr = []
        for testData in x_test:
            distArray =[]
            for i in range(0,self.len):
                distArray.append([euclidean(testData, self.x_train[i]), self.y_train[i]])
            distArray.sort()
            counts =  [0] * len(self.clusters)
            for i in range(0,self.kVal):
                index=distArray[i][1]
                counts[index]=counts[index]+1
            largest=0
            indexOfLargest=0
            for i in range(0,len(counts)):
                if counts[i]>largest:
                    largest=counts[i]
                    indexOfLargest=i
            retArr.append(indexOfLargest)
        return numpy.asarray(retArr)    
print("done")
