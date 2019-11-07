import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.metrics import silhouette_score

x = pd.read_csv("First_Echelon/Input/demanda_bodegas.csv")

X = np.array(list(zip(x["latitude"], x["longitude"])))
X

plt.scatter(x["latitude"], x["longitude"], alpha=0.7, edgecolors="b")

# Affinity Propagation clustering
af = AffinityPropagation(damping=.85)
clustering = af.fit(X)

# Silouette scores
score = metrics.silhouette_score(X, clustering.labels_)
score

# Plotting


def ploting():
    plt.scatter(x["latitude"], x["longitude"], c=clustering.labels_, cmap="rainbow", alpha=0.7)
    plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], marker="x")
    plt.show()


ploting()
