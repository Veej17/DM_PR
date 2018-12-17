#Imports and read data
import sklearn.cluster as cl
from sklearn.cluster import KMeans
from Toolbox.clusterPlot import clusterPlot as cp
import matplotlib.pyplot as plt
import csv
import pandas as pd
df = pd.read_csv('Data/SC_expression.csv')

#Set dataframe
df = df.set_index('Genes')
#condition SICBA is deleted, since primary and secundary condition are undifined and the expressions are doubble of the rest.
del df['SICIBA']
# make a colum list
cols = list(df.columns)
treatments = []
# filter the columns (conditions) by the first 3 letters.
for col in cols:
    treatments.append(col[:3])

#transpos dataframe, to cluster per gene and not per comdition
df = df.T
#Delete undefined genes
del df['__alignment_not_unique']
del df['__no_feature']
del df['__ambiguous']

#set X and y values for the cluster
X= df
print(X)
y = treatments

#Cluster with Kmeans and plot the clusters
clf = cl.k_means(X, n_clusters=14)
centroids = clf[0]
labels = clf[1]
cp(X, labels, centroids, y)
plt.show()

#fit the X data and write a file with the clusters.
km = KMeans(n_clusters=14).fit(X)

with open('clustermap.csv', mode='w') as cmap:
    cwriter = csv.writer(cmap, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    cwriter.writerow(X.index.values)
    cwriter.writerow(clf[1])
cmap.close()