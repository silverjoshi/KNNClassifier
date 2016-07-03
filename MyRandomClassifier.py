'''
Created on Jun 27, 2016

@author: rajajosh
'''
from _random import Random
import numpy

class MyRandomClassifier(object):
    "Random classifier. To be used for testing/benchmarking purposes"
    len=0
    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    def fit(self, x_train, y_train):
        self.len=len(x_train)
        
    def predict(self, x_test):
        randObj = Random()
        retArr = []
        for _ in range(0,len(x_test)):
            retArr.append(int(randObj.random()*3))
        return numpy.asarray(retArr)


print("done")
