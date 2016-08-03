from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import BernoulliRBM
import pandas
from massage import Massager
import numpy as np


train = pandas.read_csv("train.csv")
target = train["Survived"]

m = Massager()
train_array = m.transform(train,True)

brbm = BernoulliRBM(n_components= 3, learning_rate=0.01)

trantrain = brbm.fit_transform(train_array)
param_grid = dict(C=np.logspace(-10,2,13), gamma=np.logspace(-9,3,13))
grid = GridSearchCV(svm.SVC(), param_grid=param_grid)
grid.fit(trantrain,target)
C = grid.best_params_['C']
gamma = grid.best_params_['gamma']
classifier = svm.SVC(C=C, gamma=gamma)
classifier.fit(trantrain,target)

vscore = cross_val_score(classifier,train_array,target)
print "Validation score: {0} sd: {1}".format(vscore.mean(), vscore.std())

test = pandas.read_csv("test.csv")
answers = pandas.DataFrame(test["PassengerId"])
test_array = m.transform(test)
trantest = brbm.transform(test_array)
predictions = classifier.predict(trantest)
print(classifier.score(trantrain,target))
answers['Survived'] = pandas.Series(predictions.astype(int))

answers.to_csv("solution_rbm_svm.csv", index=False)