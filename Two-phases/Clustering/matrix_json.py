import json
import numpy as np
import pandas as pd

points = pd.read_csv(
    "Optimization/cluster_4.csv")[["longitude", "latitude"]].values.tolist()

blocks = []

for i in range(0, 9):
    with open("Optimization/clust_4/osr_matrix_{}.json".format(i), 'r') as f:
        data = json.load(f)
    try:
        duratione = data["durations"]
        array = np.array(duratione)
        blocks.append(array)
    except:
        print("Error en: ".format(i))

[blocks[i].shape for i in range(len(blocks))]

matrix = np.vstack([np.hstack(blocks[i:i+3]) for i in range(0, len(blocks), 3)])

np.diagonal(matrix)
matrix


np.savetxt("duration_cluster4.csv", matrix, delimiter=",")

dist_matrix = pd.read_csv("Optimization/dist_matrix_clust4.csv", header=None).iloc[1:, 1:]

dist_matrix = dist_matrix/1000
dist_matrix = np.array(dist_matrix)
dist_matrix
# CREATE THE TRANSPORTATION COST FUNCTION
cost_matrix = 0.15*(matrix) + 0.73*(dist_matrix) + 7
np.fill_diagonal(cost_matrix, 0)
np.savetxt("Optimization/cost_matrix_clust4.csv", cost_matrix, delimiter=',')


cost_matrix.diagonal()
