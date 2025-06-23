import numpy as np
import random
import pandas as pd
import time

from Fitness_function import Fitness

class GRASPClass:
    def __init__(self, 
                Problem_data=None, 
                iter_greedy=100,
                iter_local_search=100,
                Tam_Individuos=10, 
                Prob_Mutacion=0.02,
                Prob_Hard_Mutation=0.01,
                seed=False, 
                verbose = False):
        self.Problem_data = Problem_data
        self.iter_greedy = iter_greedy
        self.iter_local_search = iter_local_search
        self.Tam_Individuos = Tam_Individuos
        self.Prob_Mutacion = Prob_Mutacion
        self.Prob_Hard_Mutation = Prob_Hard_Mutation
        self.Num_Max = self.Problem_data["Coords_nodi_island"].shape[0]

        if seed is not False:
            print(f"Seed is set to {seed}")
            np.random.seed(seed)
        self.verbose = verbose
        
        
    def greedy(self):
        individuo = np.random.choice(range(self.Num_Max),self.Tam_Individuos,False)
        cost = Fitness(individuo, self.Problem_data, flag_route = False)
        return individuo, cost
    
    def local_search(self, individuo):
        # aux1 = np.random.randint(0, individuo.shape[0])                         # Se genera número aleatorio para ver la posición que muta
        aux1 = int(np.random.choice(np.where(individuo != 0)[0],1))
        aux2 = np.random.randint(0,self.Num_Max)                                   # Se genera el número a modificar
        individuo[aux1] = aux2
        # individuo[aux1] = 0
        individuo = self.Reparacion(individuo)

        # A veces se hace una mutación dura, haciendo que el individuo sea nuevo
        if np.random.rand() < self.Prob_Hard_Mutation: # 30% de probabilidad
            individuo = np.random.choice(range(self.Num_Max),len(individuo),False)  
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
        self.cost = np.zeros(self.iter_greedy)
        self.individuos = np.zeros((self.iter_greedy, self.Tam_Individuos), dtype=int)
        for i in range (self.iter_greedy):
            self.individuos[i], self.cost[i] = self.greedy()
            t_inicio = time.process_time()
            for j in range (self.iter_local_search):
                individuo_new = self.local_search(self.individuos[i])
                cost_new = Fitness(individuo_new, self.Problem_data, flag_route = False)
                
                if cost_new < self.cost[i]: # If minimization is taken into account
                    self.cost[i] = cost_new
                    self.individuos[i] = individuo_new
            self.Fitness_Grafica.append(self.cost[i])
            t_gen = time.process_time()
            if self.verbose:
                print(f"Tiempo en generación {i}: {t_gen-t_inicio}s. Coste = {self.cost[i]}€")
        self.Mejor_Individuo = self.individuos[np.argmin(self.cost)]
        self.Fitness_Mejor, self.Ruta, self.Costs = Fitness(self.Mejor_Individuo, self.Problem_data, flag_route = True)
