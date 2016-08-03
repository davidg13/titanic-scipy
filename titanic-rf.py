from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import pandas
import numpy as np
from massage import Massager


train = pandas.read_csv("train.csv")
target = train["Survived"]

m = Massager()
train_array = m.transform(train, True)
param_grid = dict(n_estimators=[32,64,128,256,512,1024],
                   max_depth=[5,6,7,8,9,10],
                   min_samples_leaf=[8,16,32,64])
grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
grid.fit(train_array,target)
ne = grid.best_params_['n_estimators']
md = grid.best_params_['max_depth']
mss = grid.best_params_['min_samples_leaf']
print ne, md, mss
classifier = RandomForestClassifier(n_estimators=ne,max_depth=md,min_samples_split=mss)
classifier.fit(train_array,target)
print(classifier.score(train_array,target))

vscore = cross_val_score(classifier,train_array,target)
print "Validation score: {0} sd: {1}".format(vscore.mean(), vscore.std())

print classifier.feature_importances_
test = pandas.read_csv("test.csv")
answers = pandas.DataFrame(test["PassengerId"])
test_array = m.transform(test)
predictions = classifier.predict(test_array)
answers['Survived'] = pandas.Series(predictions.astype(int))

answers.to_csv("solution_rf.csv", index=False)