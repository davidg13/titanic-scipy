from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas
from massage import Massager


train = pandas.read_csv("train.csv")
target = train["Survived"]

m = Massager()
train_array = m.transform(train, True)
param_grid = dict(min_samples_split=[1,2,3,4,5,6,7],
                  max_depth=[3,4,5,6,7,8,9,10,11,12])
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid)
grid.fit(train_array,target)
mss = grid.best_params_['min_samples_split']
md = grid.best_params_['max_depth']
classifier = DecisionTreeClassifier(max_depth=md, min_samples_leaf=mss)

#pca = PCA(n_components=3)
#classifier = Pipeline([('pca', pca),('dt', dt)])
classifier.fit(train_array,target)

vscore = cross_val_score(classifier,train_array,target)
print "Validation score: {0} sd: {1}".format(vscore.mean(), vscore.std())

test = pandas.read_csv("test.csv")
answers = pandas.DataFrame(test["PassengerId"])
test_array = m.transform(test)
predictions = classifier.predict(test_array)
answers['Survived'] = pandas.Series(predictions.astype(int))

answers.to_csv("solution_dt.csv", index=False)