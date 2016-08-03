from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
import pandas
from massage import Massager


train = pandas.read_csv("train.csv")
target = train["Survived"]

m = Massager()
train_array = m.transform(train, True)
classifier = GaussianNB()
classifier.fit(train_array,target)

vscore = cross_val_score(classifier,train_array,target)
print "Validation score: {0} sd: {1}".format(vscore.mean(), vscore.std())

test = pandas.read_csv("test.csv")
answers = pandas.DataFrame(test["PassengerId"])
test_array = m.transform(test)
predictions = classifier.predict(test_array)
answers['Survived'] = pandas.Series(predictions.astype(int))

answers.to_csv("solution_nb.csv", index=False)