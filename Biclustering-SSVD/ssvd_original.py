#Import necessary packages
import pandas as pd
import numpy as np

#Threshold function
def thresh(X,threshtype, paralambda):
    a = 3.7    
    if threshtype==1:
        tmp= np.multiply(np.sign(X),(abs(X)>=paralambda.astype('int')))
        y = np.multiply(tmp, abs(X)-paralambda)
    elif threshtype==2:
        y = np.multiply(X,(abs(X)>paralambda.astype('int')))
    return y

#ssvd fucntion - unoptimized
def ssvd_works(X,param=None):
    n, d = X.shape
    threu = 1
    threv = 1
    gamu = 0
    gamv = 0
    t1, t2, t3 = np.linalg.svd(X)
    u0 = t1[:,0]
    v0 = t3[:,1]
    merr = 10**-4
    niter = 100
    ud = 1
    vd = 1
    iters = 0
    SST = sum(sum(X**2))
    while (ud > merr or vd > merr):
        iters = iters + 1
        z = np.matmul(X.T,u0)
        winv = abs(z)**gamv
        sigsq = (SST - np.sum(z**2))/(n*d-d)

        #Updating v
        tmp = abs(z**winv)
        tv = np.sort(np.append(tmp,0))
        rv = sum(tv>0)
        Bv = np.ones((d+1,1))*np.Inf
        for i in range(0,rv):
            lvc = tv[d-i]
            para = {'threshtype': threv, 'lambda': lvc/winv[winv!=0]}
            temp2 = thresh(z[winv!=0],para['threshtype'],para['lambda'])
            vc = temp2
            Bv[i] = sum(sum((X - np.multiply(u0[:,np.newaxis],vc[:,np.newaxis].T))**2)/sigsq + i*np.log(n*d))
        Iv = np.argmin(Bv) + 1
        temp = np.sort(np.append(abs(np.multiply(z, winv)),0))
        lv = temp[d-Iv-1]
        para['lambda'] = np.multiply(lv, winv[winv!=0])
        temp2 = thresh(z[winv!=0],para['threshtype'],para['lambda'])
        v1 = temp2
        v1 = v1/np.sqrt(sum(v1**2)) #v_new

        #Updating u
        z = np.matmul(X, v1)
        winu = abs(z)**gamu
        sigsq = (SST - sum(z**2))/(n*d-n)
        tu = np.sort(np.append(abs(np.multiply(z, winu)),0))
        ru = sum((tu>0).astype('int'))
        Bu = np.ones((n+1,1))*np.Inf
        for i in range(0,ru):
            luc = tu[n-i]
            para = {'threshtype': threu, 'lambda': luc/winu[winu!=0]}
            temp2 = thresh(z[winu!=0],para['threshtype'],para['lambda'])
            uc = temp2
            Bu[i] = sum(sum((X - temp2[:,np.newaxis]*v1.T)**2)/sigsq + i*np.log(n*d))
        Iu = np.argmin(Bu)+1
        temp = np.sort(np.append(abs(np.multiply(z, winu)),0))
        lu = temp[n-Iv-1]
        para['lambda'] = lu/winu[winu!=0]
        temp2 = thresh(z[winu!=0],para['threshtype'],para['lambda'])
        u1 = temp2
        u1 = u1/np.sqrt(sum(u1**2)) #u_new

        ud = np.sqrt(np.sum((u0-u1)**2))
        vd = np.sqrt(np.sum((v0-v1)**2))
        if iters > niter:
            print('Fail to converge! Increase the niter!')
            break
        u0 = u1
        v0 = v1
    u = u1
    v = v1
    return u,v,iters