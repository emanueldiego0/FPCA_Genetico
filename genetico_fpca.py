# -*- coding: utf-8 -*-
"""

TESTE1: HOUDOUT=5, TAM_POP=3, TAXA=0.02, EPOCHS=100

@author: Emanuel
"""
#importacoes
import numpy as np
import glob
import imageio
import cv2
import warnings
import random
import progressbar
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#constantes
M = 644             #numero de features
K = 70              #numero maximo de componentes
HOLDOUT = 5        #rodadas
TAXA = 0.02         #taxa de probabilidade de mutacao
TAM_POP = 3        #qtde de individuos da populacao
EPOCHS = 100        #numero de epocas


def CarregarYaleFaces():
    files = glob.glob("databases/yalefaces/*")
    images_yale = [np.array(imageio.mimread(file))[0] for file in files]
    images_yale_resized = [cv2.resize(image, dsize=(28, 23), interpolation=cv2.INTER_CUBIC) for image in images_yale]
    images_yale_resized = np.array(images_yale_resized)
    images_yale_flatten = [image.flatten() for image in images_yale_resized]
    images_yale_flatten = np.array(images_yale_flatten)
    #print('#Amostras (n): '+str(images_yale_flatten.shape[0]))
    #print('#Features (m): '+str(images_yale_flatten.shape[1]))
    Y = [f.split('.')[0] for f in files]
    return images_yale_flatten, Y

def F_Eigenfaces(X, W, k, R):
        r = 0.01
        n = X.shape[0]
        m = X.shape[1]
        mean = np.mean(X, axis = 0)
        D = np.zeros((n, n))
        for j in range(m):
            a = np.power(X[:,j], r) - np.power(X[:,j].mean(), r)
            a = a.reshape(n,1)
            b = a.T
            D = D + (a * b)
        val, vec = np.linalg.eig(D)
        val = np.abs(val)
        vec_c = 1. / np.power((n * val), 0.5)
        vec_c = vec_c * (np.power(X, R) - np.power(mean, R)).T.dot(vec)
        X_ = vec_c.T.dot((np.power(X, R) - np.power(mean, R)).T)
        X__ = vec_c.T.dot((np.power(W, R) - np.power(mean, R)).T)
        return X_.T[:,:k], X__.T[:,:k]

#ALGORITMO GENETICO
class Individuo():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        cromossomo = np.random.rand(M) / 10   #vetor de tamanho M com valores no intervalo [0,1]
        self.cromossomo = cromossomo.tolist()
        self.nota = 0
    
    
    def fitness(self):
        acc_geral = [] #guarda 15 acuracias. Componentes de 1, 4-70 em intervalos de 5
    
        n_components = 1
        
        for i in range(10):
            
            acc = []
            for j in range(HOLDOUT):
                
                #amostragem em treino e teste
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state= round(time()) + 15 * j)
                
                #aplicacao do FPCA
                X_feig_train, X_feig_test = F_Eigenfaces(X_train, X_test, n_components, self.cromossomo)
                
                #avaliar acuracia usando 1-NN
                clf_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_feig_train, y_train)
                score = clf_1nn.score(X_feig_test, y_test) * 100
                acc.append(score)
    
            acc_geral.append(np.array(acc).mean())
            
            if(n_components == 1):
                n_components += 4
            else:
                n_components += 5
        
        #print(acc_geral)
        self.nota = np.array(acc_geral).mean()
    
    
    def crossover(self, outro_individuo):
        #M = 8
        n = 0
        slices = [0]
        while(n < M-1):
            slice = random.randint(50, 100)
            n += slice
            slices.append(n)
        
        if(slices[-1] >= M-1):
            slices.pop(-1)
        
        slices.append(M-1)
        slices.append(M)
        
        #print(slices)
        
        filho1 = Individuo(self.X, self.Y)
        filho2 = Individuo(self.X, self.Y)
        
        cromossomo_filho1 = []
        turno = 0
        for i in range(len(slices)-1):
            if(turno == 0):
                cromossomo_filho1.extend(self.cromossomo[slices[i]:slices[i+1]])
                turno = 1
            else:
                cromossomo_filho1.extend(outro_individuo.cromossomo[slices[i]:slices[i+1]])
                turno = 0
        
        cromossomo_filho2 = []
        turno = 0
        for i in range(len(slices)-1):
            if(turno == 0):
                cromossomo_filho2.extend(outro_individuo.cromossomo[slices[i]:slices[i+1]])
                turno = 1
            else:
                cromossomo_filho2.extend(self.cromossomo[slices[i]:slices[i+1]])
                turno = 0
                
        filho1.cromossomo = cromossomo_filho1
        filho2.cromossomo = cromossomo_filho2
        return [filho1, filho2]
    
    def mutacao(self, taxa):
        for i in range(len(self.cromossomo)):
            if random.random() < taxa:
                self.cromossomo[i] = np.random.rand()
        return self


class AlgoritmoGenetico():
    def __init__(self, X, Y, tamanho_populacao):
        self.tamanho_populacao = tamanho_populacao
        self.populacao = []
        #self.geracao = 0
        self.melhor_solucao = 0
        self.X = X
        self.Y = Y
        
    def inicializa_populacao(self):
        for i in range(self.tamanho_populacao):
            self.populacao.append(Individuo(X, Y))
        self.melhor_solucao = self.populacao[0]
    
    def ordena_populacao(self):
        self.populacao = sorted(self.populacao,
                                key = lambda populacao: populacao.nota,
                                reverse = True)
        
    def melhor_individuo(self, individuo):
        if individuo.nota > self.melhor_solucao.nota:
            self.melhor_solucao = individuo
    
    #media????
    def soma_fitness(self):
        soma = 0
        for i in self.populacao:
            soma += i.nota
        return soma
    
    def seleciona_pai(self, soma_fitness):
        pai = -1
        valor_sorteado = random.random() * soma_fitness
        soma = 0
        i = 0
        while i < len(self.populacao) and soma < valor_sorteado:
            soma += self.populacao[i].nota
            pai += 1
            i += 1
        return pai
    
    def fitness_medio_populacao(self):
        return self.soma_fitness() / float(self.tamanho_populacao)
    
    
    def executar(self, epochs, taxa_mutacao):
        
        self.inicializa_populacao()
        
        for individuo in self.populacao:
            individuo.fitness()
        
        self.ordena_populacao()
        self.melhor_solucao = self.populacao[0]
        #self.lista_solucoes.append(self.melhor_solucao.nota_avaliacao)
        
        #self.visualiza_geracao()
    
        #for e in progressbar.progressbar(range(epochs)):
        for e in range(epochs):
            #print("loop1")
            for i in progressbar.progressbar(self.populacao):
            #for i in genetico.populacao:
                i.fitness()
                #print("loop2") 
                #print("nota: %s" %(str(i.nota)))
            self.ordena_populacao()
            self.melhor_individuo(self.populacao[0])
            print('# Epoca: %s\tFitness Melhor Cromossomo: %s\tFitness Medio: %s' 
                  %(str(e+1), str(self.melhor_solucao.nota), str(self.fitness_medio_populacao())))
            soma_fitness = self.soma_fitness()
            
            nova_populacao = []
            for novos in range(0, self.tamanho_populacao, 2):
                pai1 = self.seleciona_pai(soma_fitness)
                #print("loop3")    
                while True:
                    pai2 = self.seleciona_pai(soma_fitness)
                    #print('pai1 %s pai2 %s' %(str(pai1), str(pai2)))
                    if(pai1 != pai2):
                        break
                
                filhos = self.populacao[pai1].crossover(self.populacao[pai2])
                nova_populacao.append(filhos[0].mutacao(taxa_mutacao))
                nova_populacao.append(filhos[1].mutacao(taxa_mutacao))
            
            self.populacao = list(nova_populacao)
            
        self.ordena_populacao()
        self.melhor_individuo(self.populacao[0])
        return self.melhor_individuo
        
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    X, Y = CarregarYaleFaces()
    genetico = AlgoritmoGenetico(X, Y, TAM_POP)
    solucao = genetico.executar(EPOCHS, TAXA)