#!/usr/local/bin/python3
"""
this module is a toolbox for atmosphere tide analysis.

Yu-Tsung 2015.11.28 first release "tidaldecompo", "tidalrecompo"
Yu-Tsung 2015.12.13 update and add "tide_sample"
"""

import numpy as np


def tidaldecompo(data,ut,glon):
    """
    ===========================================================================
    tidal decomposition
    A = Tide amplitude, B = Phasa
    C = Zonal and time mean
    
    Yu-Tsung 2015.11.28
    ===========================================================================
    wavefun = n*W*t-s*lambda
    Y = A*cos(n*W*t-s*lambda+B)
    [cos(wavefun)-sin(wavefun)]*[A*cos(theta),A*sin(theta)] = data;
    X*A = Y  ->  X = A\Y
      
    n = period (0,1,2,3)
    W = Angular velocity of the Earth (2*pi/24)
    t = time (hr)
    s = wavenumber (-4,-3,-2,-1,0,1,2,3,4)
    lambda_glon = glon*2*pi/360
    """

    data = data.reshape(data.size,1)
    ut   = ut.reshape(ut.size,1)
    glon = glon.reshape(glon.size,1)

    n = np.array([ 0, 0, 0, 0,\
          1, 1, 1, 1, 1, 1, 1, 1, 1,\
          2, 2, 2, 2, 2, 2, 2, 2, 2,\
          3, 3, 3, 3, 3, 3, 3, 3, 3])
    s = np.array([ 1, 2, 3, 4,\
         -4,-3,-2,-1, 0, 1, 2, 3, 4,\
         -4,-3,-2,-1, 0, 1, 2, 3, 4,\
         -4,-3,-2,-1, 0, 1, 2, 3, 4])
         
    W = 2.*np.pi/24.
    tidefuncc = lambda tm,lambdam:  np.cos(n*W*tm-s*lambdam)
    tidefuncs = lambda tm,lambdam: -np.sin(n*W*tm-s*lambdam)
    glon[glon>180] = glon[glon>180]-360
    lambda_glon = glon*2.*np.pi/360.
    cos_func = tidefuncc(ut,lambda_glon)
    sin_func = tidefuncs(ut,lambda_glon)
    h = np.concatenate(( np.ones((len(data),1)), cos_func, sin_func), axis=1)
    # X = [A1*cos(theta1);A2*cos(theta2);...A1*sin(theta1);...]
    H    = np.asmatrix(h)
    DATA = np.asmatrix(data)
    X = (H.T*H).I*H.T*DATA
    
    C = X[0,0]
    X1 = np.asarray(X[1:32])
    X2 = np.asarray(X[-31:])
    A = (X1**2+X2**2)**0.5
    B = np.arctan(X2/X1)
    index3 = np.asarray([(X1[i][0]<0 and X2[i][0]<0) for i in range(0,31)])
    index2 = np.asarray([(X1[i][0]<0 and X2[i][0]>0) for i in range(0,31)])
    B[index3] = B[index3]+np.pi
    B[index2] = B[index2]-np.pi
    
    return A, B, C


def tidalrecompo(ut,glon,A,B,C,component_name='DW1'):
    """
    ===========================================================================
    tidal recomposition
    A = Tide amplitude, B = Phasa
    C = Zonal and time mean
    component_name (ie. DW1,SW2,TW3...DE3.....)
   
    Yu-Tsung 2015.11.29
    ===========================================================================
    """

    W = 2.*np.pi/24.
    lambda_glon = glon*2.*np.pi/360.
    glon[glon>180] = glon[glon>180]-360
    
    tide_period = {'D':1,'d':1,'S':2,'s':2,'T':3,'t':3}
    tide_direction = {'W':-1,'w':-1,'E':1,'e':1}        
    if len(component_name)==3:
        n = tide_period[component_name[0]]
        s = tide_direction[component_name[1]]*int(component_name[-1])
    elif len(component_name)==2 and component_name[0]=='D':
        n = 0
        s = int(component_name[-1])
    elif len(component_name)==4 and component_name[:3]==('SPW' or 'spw'):
        n = 0
        s = int(component_name[-1])
    
    N = np.array([ 0, 0, 0, 0,\
          1, 1, 1, 1, 1, 1, 1, 1, 1,\
          2, 2, 2, 2, 2, 2, 2, 2, 2,\
          3, 3, 3, 3, 3, 3, 3, 3, 3])
    S = np.array([ 1, 2, 3, 4,\
         -4,-3,-2,-1, 0, 1, 2, 3, 4,\
         -4,-3,-2,-1, 0, 1, 2, 3, 4,\
         -4,-3,-2,-1, 0, 1, 2, 3, 4])
    
    component_index = np.asarray([N[i]==n and S[i]==s for i in range(0,31)])
    A_ns = A[component_index]
    B_ns = B[component_index]
    Y = A_ns*np.cos(n*W*ut-s*lambda_glon+B_ns)+C
    Yt = 0
    for i in range(0,31):
        Each_Y = A[i]*np.cos(N[i]*W*ut-S[i]*lambda_glon+B[i])
        Yt += Each_Y
    Yt += C
    return Yt, Y


if __name__=="__main__":
    ut_tmp = np.arange(0,24,1)
    glon_tmp = np.arange(-180,180,2.5)
    ut,glon = np.meshgrid(ut_tmp,glon_tmp)
    ut = ut.reshape(len(ut_tmp)*len(glon_tmp),1)
    glon = glon.reshape(len(ut_tmp)*len(glon_tmp),1)
    
    W = 2.*np.pi/24.
    lambda_glon = glon*2.*np.pi/360.
    # DW1, SW2, TW3, DE3, SPW1, SPW2
    n    = np.array([ 1, 2, 3, 1, 0, 0])
    s    = np.array([-1,-2,-3, 3, 1, 2])
    theta= np.array([ 3, 3, 3, 2, 1, 0]) * W

    dw1 = 30*np.cos(n[0]*W*ut-s[0]*lambda_glon+theta[0])
    sw2 = 20*np.cos(n[1]*W*ut-s[1]*lambda_glon+theta[1])
    tw3 = 10*np.cos(n[2]*W*ut-s[2]*lambda_glon+theta[2])
    de3 = 25*np.cos(n[3]*W*ut-s[3]*lambda_glon+theta[3])
    spw1= 15*np.cos(n[4]*W*ut-s[4]*lambda_glon+theta[4])
    spw2=  5*np.cos(n[5]*W*ut-s[5]*lambda_glon+theta[5])
    data = dw1+sw2+tw3+de3+spw1+spw2 + 50 # set zonal mean = 50

    amplitude,phase,zonal_mean = tidaldecompo(data,ut,glon)
    
    #print(amplitude)
    #print(phase)
    #print(zonal_mean)
    
    
    
    
