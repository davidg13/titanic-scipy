from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
import pandas
from massage import Massager


train = pandas.read_csv("train.csv")
target = train["Survived"]

m = Massager()
train_array = m.transform(train, True)
dt = DecisionTreeClassifier(max_depth=1)
dt.fit(train_array,target)
param_grid = dict(n_estimators=[5,10,20,40,80,160,320],
                  learning_rate=[1.0,0.9,0.8,0.7])
grid = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid)
grid.fit(train_array,target)
ne = grid.best_params_['n_estimators']
lr = grid.best_params_['learning_rate']
print "ne {0} lr {1}".format(ne, lr)
classifier = AdaBoostClassifier(dt,n_estimators=ne,learning_rate=lr)
classifier.fit(train_array,target)

vscore = cross_val_score(classifier,train_array,target)
print "Validation score: {0} sd: {1}".format(vscore.mean(), vscore.std())

test = pandas.read_csv("test.csv")
answers = pandas.DataFrame(test["PassengerId"])
test_array = m.transform(test)
predictions = classifier.predict(test_array)
answers['Survived'] = pandas.Series(predictions.astype(int))

answers.to_csv("solution_ada.csv", index=False)