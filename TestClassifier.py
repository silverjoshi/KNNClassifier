'''
Created on Jun 27, 2016

Test code for testing KNNClassifier implementation

@author: rajajosh
'''
from sklearn.datasets.base import load_iris
from sklearn.cross_validation import train_test_split

from sklearn.neighbors.classification import KNeighborsClassifier

from sklearn.metrics.classification import accuracy_score
from KNNClassifier import KNNClassifier
import time
from MyRandomClassifier import MyRandomClassifier

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.95)
print(x_test,x_train,y_test,y_train)

print("Using random classifier:")
clsf = MyRandomClassifier()
clsf.fit(x_train, y_train)
predictions = clsf.predict(x_test)
print(predictions,y_test)
print(accuracy_score(y_test,predictions))

print("Using KNN classifier:")
clsf = KNeighborsClassifier(n_neighbors=3)
start = time.time()
clsf.fit(x_train, y_train)
predictions = clsf.predict(x_test)
end = time.time()
print("time:", end-start)
print(predictions,y_test)
print(accuracy_score(y_test,predictions))

print("Using My KNN classifier:")
clsf = KNNClassifier()
start = time.time()
clsf.fit(x_train, y_train,kVal=3)
predictions = clsf.predict(x_test)
end = time.time()
print("time:", end-start)
print(predictions,y_test)
print(accuracy_score(y_test,predictions))

print (type(y_test), type(predictions))
