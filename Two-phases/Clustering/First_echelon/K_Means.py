import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.spatial import distance


x = pd.read_csv("First_Echelon/Input/demanda_bodegas.csv")


latitude = x["latitude"][0:600]
longitude = x["longitude"][0:600]
demanda = x["demanda"][0:600]
X = np.array(list(zip(latitude, longitude)))
X.shape

# Using the K-KMeans clustering method


def scoreSilouette(X):
    scores = []
    for i in range(2, 11):
        fitx = KMeans(n_clusters=i, init='k-means++', n_init=5).fit(X)
        score = metrics.silhouette_score(X, fitx.labels_)
        scores.append(score)
    return np.array(scores)

# plotting optimal number of n_clusters


valores = np.array(list(zip(range(1, 10), scoreSilouette(X))))
maximum = max(valores[:, 1])  # optimal number of clusters =4


# kmeans algorithm

km = KMeans(n_clusters=4, init='k-means++')
km.fit(X)
predict = km.predict(X)


def plot_solution(latitude, longitude, array):
    plt.scatter(latitude, longitude, c=km.labels_, cmap="rainbow", alpha=0.7)
    plt.scatter(array[:, 0], array[:, 1], marker='x')
    plt.show()


# GROUPING THE DATA TOGHETER
df = pd.DataFrame(list(zip(latitude, longitude, demanda)))
df.columns = ["latitude", "longitude", "demanda"]
df["cluster"] = km.labels_
colores = sns.color_palette()[0:4]
df = df.sort_values("cluster")

# RENAMING TH VALUES
nombres = {"0": "down", "1": "center", "2": "upp_right", "3": "down"}
df["cluster_name"] = [nombres[str(i)] for i in df.cluster]


# check plot_solution
sns.scatterplot(df["latitude"], df["longitude"], hue=df["cluster_name"], palette=colores)
plt.show()


def clusters():
    cluster1 = df[df.cluster == 0]
    cluster2 = df[df.cluster == 1]
    cluster3 = df[df.cluster == 2]
    cluster4 = df[df.cluster == 3]
    k_coords = km.cluster_centers_
    return cluster1, cluster2, cluster3, cluster4, k_coords


c1, c2, c3, c4, center = clusters()

mercados_total = pd.read_csv("Two-phases/Clustering/mercados_fixed.csv")
mercados_coord = mercados_total[["latitude", "longitude"]]


# KNOWING THE DEMANDS FOR ALL THE clusters
demand0 = df.loc[df["cluster"] == 0, "demanda"].sum()
demand1 = df.loc[df["cluster"] == 1, "demanda"].sum()
demand2 = df.loc[df["cluster"] == 2, "demanda"].sum()
demand3 = df.loc[df["cluster"] == 3, "demanda"].sum()


demands = np.array([demand0, demand1, demand2, demand3])
demands
# WHAT ARE THE BEST MARKETS?


def clusterisar(cluster, indice):
    X = [(center[indice, :])]
    coords = []
    for index, row in cluster.iterrows():
        coords.append((row[0], row[1]))
    distancia = distance.cdist(X, coords, "euclidean")
    return distancia


merca1 = c1.iloc[np.argmin(np.min(clusterisar(c1, 0), axis=0))][[
    "latitude", "longitude", "cluster", "cluster_name"]]
merca2 = c2.iloc[np.argmin(np.min(clusterisar(c2, 1), axis=0))][[
    "latitude", "longitude", "cluster", "cluster_name"]]
merca3 = c3.iloc[np.argmin(np.min(clusterisar(c3, 2), axis=0))][[
    "latitude", "longitude", "cluster", "cluster_name"]]
merca4 = c4.iloc[np.argmin(np.min(clusterisar(c4, 3), axis=0))][[
    "latitude", "longitude", "cluster", "cluster_name"]]


#
# merca1 = merca1[["latitude", "longitude"]]
# merca2 = merca2[["latitude", "longitude"]]
# merca3 = merca3[["latitude", "longitude"]]
# merca4 = merca4[["latitude", "longitude"]]
markets = [merca1, merca2, merca3, merca4]
markets = np.array(markets)

markets = np.concatenate((markets, demands[:, None]), axis=1)
markets

plot_solution(latitude, longitude, km.cluster_centers_)

plot_solution(latitude, longitude, markets)

type(markets)
a = pd.DataFrame(markets)
a
a.columns = ["latitude", "longitude", "cluster_index", "cluster_name", "demand"]
a

a.to_csv("Two-phases/Clustering/First_echelon/optimal_markets.csv")
c1.head()
a.iloc[0]
cluster_1 = pd.DataFrame(np.vstack((a.iloc[0], c1)))
cluster_2 = pd.DataFrame(np.vstack((a.iloc[1], c2)))
cluster_3 = pd.DataFrame(np.vstack((a.iloc[2], c3)))
cluster_4 = pd.DataFrame(np.vstack((a.iloc[3], c4)))


cluster_1.columns = ["latitude", "longitude", "cluster_index", "cluster_name", "demand"]
cluster_2.columns = ["latitude", "longitude", "cluster_index", "cluster_name", "demand"]
cluster_3.columns = ["latitude", "longitude", "cluster_index", "cluster_name", "demand"]
cluster_4.columns = ["latitude", "longitude", "cluster_index", "cluster_name", "demand"]

cluster_1.to_csv("Two-phases/Optimization/cluster_1.csv")
cluster_2.to_csv("Two-phases/Optimization/cluster_2.csv")
cluster_3.to_csv("Two-phases/Optimization/cluster_3.csv")
cluster_4.to_csv("Two-phases/Optimization/cluster_4.csv")


# a
#
# minc1=np.argmin(np.min(cluster1(c1,0),axis=0))
# mercado1=c1.iloc[minc1]
#
# def cluster2():
#     X=[(center[1,:])]
#     coords=[]
#     for index,row in c2.iterrows():
#         coords.append((row[0],row[1]))
#     distancia=distance.cdist(X,coords,"euclidean")
#     return distancia
#
# minc2=np.argmin(np.min(cluster2(),axis=0))
# mercado2=c2.iloc[minc2]
#
# def cluster3():
#     X=[(center[1,:])]
#     coords=[]
#     for index,row in c2.iterrows():
#         coords.append((row[0],row[1]))
#     distancia=distance.cdist(X,coords,"euclidean")
#     return distancia
#
# minc3=np.argmin(np.min(cluster3(),axis=0))
# mercado3=c3.iloc[minc3]
