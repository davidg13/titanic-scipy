from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
import pandas
import numpy as np
from massage import Massager


train = pandas.read_csv("train.csv")
target = train["Survived"]

m = Massager()
train_array = m.transform(train, True)
input_len = train_array.shape[1]
net = buildNetwork(input_len, 4, 1)
ds = ClassificationDataSet(input_len, 1)
for i in xrange(train_array.shape[0]):
    ds.addSample(train_array[i, :], target[i])

trainer = BackpropTrainer(net, ds, learningrate=0.02, momentum=0.5, verbose=True)
trainer.trainUntilConvergence(maxEpochs=150,verbose=True)

test = pandas.read_csv("test.csv")
answers = pandas.DataFrame(test["PassengerId"])
test_array = m.transform(test)
test_ds = ClassificationDataSet(train_array.shape[1], 1)
predictions = []
for i in xrange(test_array.shape[0]):
    out = net.activate(test_array[i, :])
    survived = 1 if out[0] >= 0.5 else 0
    predictions.append(survived)
answers['Survived'] = pandas.Series(predictions)

answers.to_csv("solution_nn.csv", index=False)