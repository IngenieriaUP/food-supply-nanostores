#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import random
import os
import codecs
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[32]:


class TravelingSalesmanProblem:
 

    def __init__(self,name,coordenadas):
     

        # initialize instance variables:
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0
        self.coordenadas=coordenadas
    

        # initialize the data:
        self.__initData()



    def __len__(self):
 
        return self.tspSize   #retorna cuantas ciudades hay en el problema

    def __initData(self):
        
        
        #crear data desde cero
        if not self.locations or not self.distances:
            self.__createData()
        
        self.tspSize = len(self.locations)

    def __createData(self):
        
        self.locations=[]
      
    #         self.locations = [[i,j] for i in range (self.tspSize) for j in range (self.tspSize)]
        
        #dataframe
        file=pd.read_csv("/home/diegomatuk/snap/julia/Pruebas/"+self.name+".csv",sep=",",header=None)
        file=file.iloc[0:,0:]
        
        
        #coordenadas
        coordenadas=pd.read_csv("/home/diegomatuk/snap/julia/Pruebas/"+self.coordenadas+".csv")
        coordenadas=coordenadas.iloc[:,4:6]

        coordenadas=coordenadas.iloc[1:]
        
        for fila in range(len(coordenadas.index)):
            self.locations.append(np.asarray(coordenadas.iloc[fila], dtype=np.float32))
        
        #longitud del problema
        self.tspSize=len(self.locations)
        
        #imprimir la data del problem
        print("tamaño={}".format(self.tspSize),"locations={}".format(self.locations))
        
        #crear la matriz con 0 primero
        self.distances = [[0] * self.tspSize for _ in range(self.tspSize)]

        #que la nueva matriz tenga las ya calculadas distancias (solo va a ser el triangulo inferior)
        for i in range(self.tspSize):
            for j in range(i + 1, self.tspSize):
                # poner la distancia del dataframe a la nueva matriz
                distance = file[i][j]
                self.distances[i][j] = distance
                self.distances[j][i] = distance
                print("{}, {}: location1 = {}, location2 = {} => distance = {}".format(i, j, self.locations[i], self.locations[j], distance))
#         print(self.distances)
#             # serialize locations and distances:
#             pickle.dump(self.locations, open(os.path.join("tsp-data", self.name + "-loc.pickle"), "wb"))
#             pickle.dump(self.distances, open(os.path.join("tsp-data", self.name + "-dist.pickle"), "wb"))


    def getTotalDistance(self, ciudad):
        """ciudad: lista de las ciudades que hay en la ruta :)"""
        
        # distancia entre la ultima y primera ciudad
        distance = self.distances[ciudad[-1]][ciudad[0]]

        # add the distance between each pair of consequtive cities:
        for i in range(len(ciudad) - 1):
            distance += self.distances[ciudad[i]][ciudad[i + 1]]

        return distance


    def plotData(self, indices):
        """plots the path described by the given indices of the cities

        :param indices: A list of ordered city indices describing the given path.
        :return: the resulting plot
        """

        # plot the dots representing the cities:
        plt.scatter(*zip(*self.locations), marker='.', color='red')

        # create a list of the corresponding city locations:
        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])

        # plot a line between each pair of consequtive cities:
        plt.plot(*zip(*locs), linestyle='-', color='blue')

        return plt

   
 


# In[33]:


def main():
    tsp=TravelingSalesmanProblem("distance_matrix_vrp_bodegas_sa","demanda_bodegas")
    
    solucionoptima=[]
    print("Soluciòn òptima=",solucionoptima)
    print("Soluciòn òptima=",tsp.getTotalDistance(solucionoptima))

    
    plotear=tsp.plotData(solucionoptima)

    plotear.show()
    


# In[34]:


if __name__=="__main__":
    main()


# In[ ]:





# In[ ]:




