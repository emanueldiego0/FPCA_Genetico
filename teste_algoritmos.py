# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:02:26 2018

@author: EMANUEL
"""
import scipy.stats as st
import numpy as np
import math
from operator import itemgetter


def testeSobreposicao(intervalo1, intervalo2):
    p = (intervalo1[0] >= intervalo2[0] and intervalo1[0] <= intervalo2[1])
    q = (intervalo2[0] >= intervalo1[0] and intervalo2[0] <= intervalo1[1])

    return q or p


def testeDiferencasMedias(acuracias1, acuracias2):

    diferencas = []

    for i,j in zip(acuracias1, acuracias2):
        diferencas.append(i - j)

    #media = sum(diferencas)/float(len(diferencas))
    
    intervalo = st.t.interval(0.95, len(diferencas)-1, loc=np.mean(diferencas), scale=st.sem(diferencas))

    return ((intervalo[0] <= 0 and intervalo[1] >= 0), intervalo)

def testeWilcoxon(acc1, acc2):
    linha = []
    tabela = []
    
    for i,j in zip(acc1, acc2):
        linha.append(i)
        linha.append(j)
        linha.append(j - i)
        linha.append(abs(j - i))
        tabela.append(linha)
        linha = []

    tabela.sort(key = itemgetter(3))

    ranks = range(1,len(tabela)+1)

    i = 0
    while(i < len(acc1)):
        j = 1
        media = ranks[i]
        while(i != len(acc1) - 1 and tabela[i][3] == tabela[i+j][3]):
            media += ranks[i+j]
            j += 1
        for k in range(j):
            tabela[i+k].append(media/float(j))
        i += j


    rankNulo = 0
    rankPositivo = 0
    rankNegativo = 0
    
    for t in tabela:
        if(t[2] == 0):
            rankNulo += t[4]
        elif(t[2] > 0):
            rankPositivo += t[4]
        else:
            rankNegativo += t[4]
    
    Rmais = rankPositivo + (0.5 * rankNulo)
    Rmenos = rankNegativo + (0.5 * rankNulo)

    S = min(Rmais, Rmenos)
    N = len(tabela)
    
    Z = S - (0.25 * N * (N - 1))
    Z = Z / math.sqrt(1/24. * N * (N + 1) * (2 * N + 1))
    
    return abs(Z)


if __name__ == "__main__":
    
    datasets = ['ar', 'att', 'faces95', 'georgia_tech', 'sheffield', 'yale_faces']
    
    
    for d in datasets:
        print('\nDATASET: '+d)
        file1 = 'acc\\'+'acc_'+d+'_eigenfaces.csv'
        file2 = 'acc\\'+'acc_'+d+'_eigenfaces_proposed.csv'
        
        classificador1 = ("Eigenfaces",file1)
        classificador2 = ("Eigenfaces Proposto",file2)
    
        acc_classificador1 = np.genfromtxt(classificador1[1], delimiter=',')
        acc_classificador2 = np.genfromtxt(classificador2[1], delimiter=',')
        
        taxa = 1.96
        
        #intervalos de confianca
        int1 = st.t.interval(0.95, len(acc_classificador1)-1, loc=np.mean(acc_classificador1), scale=st.sem(acc_classificador1))
        int2 = st.t.interval(0.95, len(acc_classificador2)-1, loc=np.mean(acc_classificador2), scale=st.sem(acc_classificador2))
    
        
        print('\nTESTES DE HIPOSTESE (95% de confianca):')
        print('[+] Teste de Sobreposicao: ')
        if(testeSobreposicao(int1, int2)):
            print('H0 nao rejeitada. Concluimos que os classificadores tem desempenhos iguais')
        else:
            print('H0 rejeitada. Concluimos que os calssificadores nao sao iguais em desempenho')
            if(acc_classificador1.mean() > acc_classificador2.mean()):
                print('{0} eh melhor para essa base'.format(classificador1[0]))
            else:
                print('{0} eh melhor para essa base'.format(classificador2[0]))
    
        print('[+] Teste de Diferenca das Medias: ')
        if(testeDiferencasMedias(int1, int2)[0]):
            print('H0 nao rejeitada. Concluimos que os classificadores tem desempenhos iguais')
        else:
            print('H0 rejeitada. Concluimos que os calssificadores nao sao iguais em desempenho')
            if(acc_classificador1.mean() > acc_classificador2.mean()):
                print('{0} eh melhor para essa base'.format(classificador1[0]))
            else:
                print('{0} eh melhor para essa base'.format(classificador2[0]))
                
        print('[+] Teste de Wilcoxon: ')
        
        if(testeWilcoxon(acc_classificador1, acc_classificador2) > taxa):
            print('H0 rejeitada. Concluimos que os calssificadores nao sao iguais em desempenho')
            if(acc_classificador1.mean() > acc_classificador2.mean()):
                print('{0} eh melhor para essa base'.format(classificador1[0]))
            else:
                print('{0} eh melhor para essa base'.format(classificador2[0]))
        else:
            print('H0 nao rejeitada. Concluimos que os classificadores tem desempenhos iguais')
            