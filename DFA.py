#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
#szereg=np.cumsum(np.random.random(size=4000))
#np.random.seed(4)
def licz(szereg : list)-> float:
    ts=np.array(szereg)
    tsm=ts-np.mean(ts)
    tscm=np.cumsum(tsm)
    maxn=1
    dn=[2,3,4,5,6,7,8,9,10,11,12]
    #while(maxn*2<ts.shape[0]): 
     #   maxn=maxn*2
      #  dn.append(maxn)
    eFy=[]
    for n in dn:
        ts=np.array_split(tscm,n)

        fn=[]
        for tab in ts:
            F_n=0
            x=np.arange(tab.shape[0])
            y=tab
            p=np.polyfit(x,y,1)
            F_n=np.sqrt(np.mean((y-np.polyval(p,x))**2))
            fn.append(F_n)
        F_n=np.mean(fn)
        eFy.append(F_n)
    eFy=np.array(eFy)
    #plt.plot(dn,eFy)
    Flog=np.log(eFy)
    nlog=np.log(dn)
    #print(nlog,Flog)
    #plt.plot(nlog,Flog)
    H=np.polyfit(nlog,Flog,1)
    wsp=H[0]
    #plt.show()



    return wsp
#print(licz(szereg))