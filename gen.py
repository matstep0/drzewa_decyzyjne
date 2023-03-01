#!/usr/bin/python3
import numpy as np
from scipy.special import beta
import json
###to trzeba zaimportowac do generatora
def Bd(a : float): #symetric beta distribution
    return np.random.beta(a,a)
def gen_ts(n=10,a=1): 
	# generate time series 
	# length 2**n with 
	#H depend on a parameter (beta symetric distribution)
	
    theoretical_value=-np.log2(beta(a + 2, a)/(beta(a, a)))/2
    ts=np.array([1])
    for i in range(n):
        newts=np.zeros(ts.shape[0]*2)
        for i in range(ts.shape[0]):
            v=Bd(a)
            newts[2*i]=ts[i]*v
            newts[2*i+1]=ts[i]*(1-v)
        ts=newts/np.linalg.norm(newts)
    #ts=ts-np.mean(ts)
    #ts=ts/np.linalg.norm(newts)
    return ts.tolist(),theoretical_value
def makedata(filename,n=12,s=10):
	#n dlugosc szeroegu 2**n
	#ilosc sampli o danym herscie
    aa=np.array([0.0141571, 0.0773232, 0.174672, 0.300634, 0.469525, 0.707107, \
1.06491, 1.66315, 2.86251, 6.46636, 35.318])
    wynik=[]
    for el in aa:
        for i in range(s):
            wynik.append(gen_ts(n=n,a=el))
    #print(wynik)
    with open(filename,'w') as outfile:
        json.dump(wynik,outfile)
        outfile.close()
        print("zrobilem dane")
    return len(wynik)
#makedata("dane.txt",n=4,s=1)
#print("zrobione")
