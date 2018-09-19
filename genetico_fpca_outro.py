# -*- coding: utf-8 -*-
"""
Implementacao de algoritmo genetico para encontrar o vetor R, de dimensoes 1xM sendo M o numero de features de um dataset.


@author: Emanuel
"""
#imports necessarios
import os
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
from tqdm import tnrange, tqdm_notebook
from time import sleep, time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#constantes
HOLDOUT = 10


#FUNCOES FPCA -- INICIO
#carregamento das bases
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


#classificador
def AvaliarClassificadores2(X_train, X_test, y_train, y_test):
    clf_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    return clf_1nn.score(X_test, y_test) * 100


#FPCA
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
#FUNCOES FPCA -- FIM

#FUNCOES ALGORITMO GENETICO -- INICIO
def Fitness(R, k):
    X, Y = CarregarYaleFaces()
    
    acc_geral = [] #guarda 70 acuracias para os 1-70 componentes
    
    for c in range(1, k)
        acc = []
        for j in range(HOLDOUT):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state= round(time()) + 15 * j)
            X_feig_train, X_feig_test = F_Eigenfaces2(X_train, X_test, c, R)

            acc.append(AvaliarClassificadores2(X_feig_train, X_feig_test, y_train, y_test))

        acc_geral.append(acc)
    return np.array(acc_geral).mean()

def GerarIndividuo():
    
    R = np.random.rand(644)
    R = R / 10
    
    return R.tolist()

def AvaliarPopulacao(populacao):
    fitness = []
    
    for p in populacao:
        fitness.append(Fitness(p))
    
    return fitness

def EliminarPopulacao(populacao, fitness):
    
    n = len(populacao)
    #print('n: '+str(n))
    populacao_ = [x for _,x in sorted(zip(fitness, populacao))]
    #print(len(populacao_))
    return populacao_[(n//2):]


def NovaPopulacao(populacao):
    
    n = len(populacao)
    x = []
    
    for i in range(n):
        x.append(Crossover(populacao))
        
    return populacao + x

def Crossover(populacao):
    
    tam = len(populacao[0])
    
    indice1 = np.random.randint(len(populacao))
    indice2 = np.random.randint(len(populacao))
    
    a = populacao[indice1]
    b = populacao[indice2]
    
    c = a[:tam//2]
    c.extend(b[tam//2:])
    
    c = Mutacao(c)
    
    return c

def Mutacao(individuo):
    
    x = np.random.random()
    
    if(x < 0.3):
        index = np.random.randint(len(individuo))
        individuo[index] = np.random.rand()
    return individuo


def Genetico(epochs, n_individuos):
    
    populacao = []
    
    for i in range(n_individuos):
        populacao.append(GerarIndividuo())
    
    for i in range(epochs):
        fitness = AvaliarPopulacao(populacao)
        print("# epocha: "+str(i+1)+"\t fitness medio: "+str(np.array(fitness).mean()))
        populacao_ = EliminarPopulacao(populacao, fitness)
        populacao = NovaPopulacao(populacao_)
#FUNCOES ALGORITMO GENETICO -- FIM

if __name__ == "__main__":
    Genetico(50, 20)
    

