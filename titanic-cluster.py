from sklearn.cluster import MeanShift
import pandas
import numpy as np
from massage import Massager
import matplotlib.pyplot as plt
from itertools import cycle

train = pandas.read_csv("train.csv")
target = train["Survived"]

m = Massager()
train_array = m.transform(train, True)
ms = MeanShift()
ms.fit(train_array)

labels = ms.labels_
unique_labels = np.unique(labels)
n_clusters = len(np.unique(labels))

print "there are %d unique clusters" % n_clusters

colors = cycle('rgbcmyk')
for k, col in zip(unique_labels,colors):
    members = labels == k
    live_members = train_array[members & target == 1,:]
    dead_members = train_array[members & target == 0,:]

    plt.plot(live_members[5,:], live_members[1,:], col + 'o')
    plt.plot(dead_members[5,:], dead_members[1,:], col + 'x')

plt.title('Titanic clusters')
plt.show()