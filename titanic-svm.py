from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import pandas
from massage import Massager
import numpy as np


train = pandas.read_csv("train.csv")
target = train["Survived"]

##Search for best fitting values of C,gamma
m = Massager()
train_array = m.transform(train,True)
param_grid = dict(C=np.logspace(-10,2,13), gamma=np.logspace(-9,3,13))
grid = GridSearchCV(svm.SVC(), param_grid=param_grid)
grid.fit(train_array,target)
C = grid.best_params_['C']
gamma = grid.best_params_['gamma']
print C, gamma
classifier = svm.SVC(C=C, gamma=gamma)
classifier.fit(train_array,target)

vscore = cross_val_score(classifier,train_array,target)
print "Validation score: {0} sd: {1}".format(vscore.mean(), vscore.std())

test = pandas.read_csv("test.csv")
answers = pandas.DataFrame(test["PassengerId"])
test_array = m.transform(test)
predictions = classifier.predict(test_array)
print(classifier.score(train_array,target))
answers['Survived'] = pandas.Series(predictions.astype(int))

answers.to_csv("solution_svm.csv", index=False)