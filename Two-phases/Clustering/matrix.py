import numpy as np
import pandas as pd
import requests
import json
from tqdm import tqdm

# DEFINE the array that you will use
cluster1 = pd.read_csv("Optimization/cluster_1.csv")[["longitude", "latitude"]].values.tolist()

cluster2 = pd.read_csv("Optimization/cluster_2.csv")[["longitude", "latitude"]].values.tolist()

cluster3 = pd.read_csv("Optimization/cluster_3.csv")[["longitude", "latitude"]].values.tolist()

cluster4 = pd.read_csv("Optimization/cluster_4.csv")[["longitude", "latitude"]].values.tolist()

# create a np.zero matrix with len(points) x len(points) elements to append later when you use the API
# matrix = np.zeros((len(points), len(points)))

keys = ["5b3ce3597851110001cf62484b5a677deef84813a41b445795184fd2",
        "5b3ce3597851110001cf6248d06de8ed9179499ea2f83309128acbfc",
        "5b3ce3597851110001cf6248528ad3c1233e44d3ae2e1774bfbba309"]


len(cluster4)


def osr(key, array, sources, destinations):
    body = {
        "locations": array,
        "destinations": destinations,
        "id": "matrix_request",
        "metrics": ["duration"],
        "resolve_locations": "false",
        "units": "mi",
        "sources": sources,
    }

    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml,img/png; charset=utf-8',
        'Authorization': key
    }
    response = requests.post('https://api.openrouteservice.org/v2/matrix/driving-car',
                             json=body, headers=headers).json()

    return response


len(cluster4)


a = 0
for ix in (range(0, len(cluster4), 50)):
    for jx in (range(0, len(cluster4), 50)):
        c_keys = keys[a % 3]
        sources = [str(i) for i in range(ix, ix+50)]
        destinations = [str(j) for j in range(jx, jx+50)]
        try:
            query = osr(c_keys, cluster4, sources, destinations)
            with open(f"Optimization/clust_4/osr_matrix_{a}.json", "w") as f:
                json.dump(query, f)
        except:
            print("Error en ", a)
            pass
        a += 1

# FOR THE REMAINING GOOD values
sources = [str(i) for i in range(100, 116)]
destinations = [str(i) for i in range(100, 116)]


query = osr(c_keys, cluster4, sources, destinations)
with open(f"Optimization/clust_4/osr_matrix_8.json", "w") as f:
    json.dump(query, f)

len(cluster4)

a = 6
for ix in (range(0, len(cluster4), 50)):
    c_keys = keys[a % 3]
    sources = [str(i) for i in range(100, 116)]
    destinations = [str(j) for j in range(ix, ix+50)]
    query = osr(c_keys, cluster4, sources, destinations)
    with open(f"Optimization/clust_4/osr_matrix_{a}.json", "w") as f:
        json.dump(query, f)
    a += 1


len(cluster1)
