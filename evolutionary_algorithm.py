# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:51:44 2023

@author: LuisMi-ISDEFE
"""

import numpy as np
import random
import pandas as pd
import time

from Fitness_function import Fitness

class EvolutiveClass:
    def __init__(self, 
                 Problem_data=None, 
                 Num_Individuos=200, 
                 Num_Generaciones=10, 
                 Tam_Individuos=10, 
                 Prob_Padres=0.5, 
                 Prob_Mutacion=0.02,
                 Prob_Hard_Mutation=0.01,
                 Prob_Cruce=0.5, 
                 seed=False, 
                 verbose = False):
        self.Problem_data = Problem_data
        
        self.Num_Individuos = Num_Individuos
        self.Num_Generaciones = Num_Generaciones
        self.Tam_Individuos = Tam_Individuos
        self.Num_Max = self.Problem_data["Coords_nodi_island"].shape[0]
        self.Prob_Padres = Prob_Padres
        self.Num_Padres = round(self.Num_Individuos * self.Prob_Padres)
        self.Prob_Mutacion = Prob_Mutacion
        self.Prob_Hard_Mutation = Prob_Hard_Mutation
        self.Prob_Cruce = Prob_Cruce        
        if seed is not False:
            print(f"Seed is set to {seed}")
            np.random.seed(seed)
        self.verbose = verbose
        

    def ImprimirInformacion(self):
        print("The evolutionary algorithm hyper-parameters are:")
        print("Individual number: " + str(self.Num_Individuos))
        print("Generation number: " + str(self.Num_Generaciones))
        print("Surviving proportion: " + str(self.Prob_Padres))
        print("Surviving number of individuals: " + str(self.Num_Padres))
        print("Mutation probability: " + str(self.Prob_Mutacion))
        print("Crossover probability: " + str(self.Prob_Cruce))
    
    def PoblacionInicial(self, Fil=None, Col=None, Num_Max=None):
        if Fil == None:
            Fil = self.Num_Individuos
        if Col == None:
            Col = self.Tam_Individuos
        if Num_Max == None:
            Num_Max = self.Num_Max

        Pob_Ini = np.zeros((Fil,Col),dtype=int)
        
        for i in range(Fil):
            Pob_Ini[i,:] = np.random.choice(range(Num_Max),Col,False)

        # return np.sort(Pob_Ini,axis=1)
        return Pob_Ini

    def Seleccion(self, poblacion_inicial, coste):
        # Minimizar
        index = np.argsort(coste)
        coste_ordenado = np.sort(coste)
        # Maximizar
        # index = np.argsort(coste)[::-1]
        # coste_ordenado = np.sort(coste)[::-1]
        
        # coste_ordenado = coste_ordenado[0:self.Num_Padres]
        poblacion_actual = poblacion_inicial[index,:]
        poblacion_actual = poblacion_actual[0:self.Num_Padres,:]

        return poblacion_actual, coste_ordenado

    def Cruce (self, poblacion, Num_Max = None):
        if Num_Max == None:
            Num_Max = self.Num_Max
        for i in range (self.Num_Individuos - self.Num_Padres):
            Indice_Padres = random.sample(range(self.Num_Padres), 2)            # Se elige aleatoriamente el indice de los padres
            
            # Cruce del individuo en los dispositivos
            Padre1 = poblacion[Indice_Padres[0],:]                              # Se coge el padre 1
            Padre2 = poblacion[Indice_Padres[1],:]                              # Se coge el padre 2
            Hijo = np.copy(Padre1)                                              # El hijo va a ser una copia del padre 1
            vector = 1*(np.random.rand(self.Tam_Individuos) > self.Prob_Cruce)  # Se genera un vector para seleccionar los genes del padre 2
            Hijo[np.where(vector==1)[0]] = Padre2[np.where(vector==1)[0]]       # Los genes seleccionados del padre 2 pasan al hijo
            Hijo = self.Reparacion(Hijo)
            if np.random.rand() < self.Prob_Mutacion:                           # Se comprueba si el hijo va a mutar
            #     if np.random.rand() < 0.5:
                    Hijo = self.Mutacion(Hijo, Num_Max)
            #     else:
            #         Hijo = self.Mutacion_Gaussiana(Hijo, Num_Max/2)
            poblacion = np.insert(poblacion,self.Num_Padres+i,Hijo, axis = 0)   # Se añade a la población una vez que ha mutado y se ha reparado

        return poblacion

    def Mutacion (self, individuo, Num_Max=None):                                
        # aux1 = np.random.randint(0, individuo.shape[0])                         # Se genera número aleatorio para ver la posición que muta
        aux1 = int(np.random.choice(np.where(individuo != 0)[0],1))
        aux2 = np.random.randint(0,Num_Max)                                   # Se genera el número a modificar
        individuo[aux1] = aux2
        # individuo[aux1] = 0
        individuo = self.Reparacion(individuo)

        # A veces se hace una mutación dura, haciendo que el individuo sea nuevo
        if np.random.rand() < self.Prob_Hard_Mutation: # 30% de probabilidad
            individuo = np.random.choice(range(Num_Max),len(individuo),False)  
        return individuo

    def Mutacion_Gaussiana(self, individuo, Media):
        Vec_Mut = Media * np.random.randn(individuo.shape[0])
        
        individuo = np.clip(np.array(Vec_Mut + individuo, dtype = np.int32),0,2*Media)
        return individuo

    def Reparacion(self, individuo):
        # Creamos un conjunto de todos los posibles valores en el rango [0, self.Num_Max]
        todos_valores = set(range(self.Num_Max))
        
        # Inicializamos el resultado y un conjunto para los valores ya utilizados
        resultado = []
        usados = set()
        
        # Reemplazamos los valores duplicados
        for val in individuo:
            if val < 0 or val > self.Num_Max:
                continue
            if val in usados:
                # Encontramos un valor aleatorio disponible en el rango
                valores_disponibles = list(todos_valores - usados)
                if valores_disponibles:
                    nuevo_valor = np.random.choice(valores_disponibles)
                    usados.add(nuevo_valor)
                    resultado.append(nuevo_valor)
            else:
                usados.add(val)
                resultado.append(val)
        
        # Aseguramos que el resultado tenga el mismo tamaño que el vector original
        while len(resultado) < len(individuo):
            valores_disponibles = list(todos_valores - usados)
            if valores_disponibles:
                nuevo_valor = np.random.choice(valores_disponibles)
                usados.add(nuevo_valor)
                resultado.append(nuevo_valor)

        # return np.sort(np.array(resultado))
        return np.array(resultado)
        


    def InicioAlgoritmo(self):

        self.Fitness_Grafica = []
        self.Pob_Ini = self.PoblacionInicial()
        self.Coste_Pob = np.zeros((self.Num_Individuos))

        for indice, individuo in enumerate(self.Pob_Ini):
            # print(indice)
            self.Coste_Pob[indice] = Fitness(individuo, self.Problem_data, flag_route = False)

        self.Pob_Act = np.copy(self.Pob_Ini)


        
        t_inicio = time.process_time()
        for generacion in range(self.Num_Generaciones):
            self.Pob_Act, self.Coste_Pob = self.Seleccion(self.Pob_Act, self.Coste_Pob)
            self.Pob_Act = self.Cruce(self.Pob_Act)
            for indice, individuo in enumerate(self.Pob_Act):
                if indice < self.Num_Padres:
                    continue
                self.Coste_Pob[indice] = Fitness(individuo, self.Problem_data, flag_route = False)

            self.Fitness_Grafica.append(self.Coste_Pob[0])
            t_gen = time.process_time()
            if self.verbose:
                print(f"Tiempo en generación {generacion}: {t_gen-t_inicio}s. Coste = {self.Coste_Pob[0]:.2f}€")
        self.Mejor_Individuo = self.Pob_Act[0,:]
        self.Fitness_Mejor, self.Ruta, self.Costs = Fitness(self.Mejor_Individuo, self.Problem_data, flag_route = True)

if __name__ == "__main__":
    print("Evolutionary algorithm")
    