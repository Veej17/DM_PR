import scipy.io as sio
import sklearn.cluster as cl
from sklearn.cluster import KMeans
from Toolbox.clusterPlot import clusterPlot as cp
import matplotlib.pyplot as plt
import pandas as pd
from Open import data

X= data.loc[:,'IFFABF':].T
y = X.index.values

clf = cl.k_means(X, n_clusters=15)
centroids = clf[0]
labels = clf[1]
cp(X, labels, centroids, y)
# plt.show()

km = KMeans(n_clusters=15).fit(X)

import csv

with open('clustermap.csv', mode='w') as cmap:
    cwriter = csv.writer(cmap, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    cwriter.writerow(X.index.values)
    cwriter.writerow(clf[1])
cmap.close()