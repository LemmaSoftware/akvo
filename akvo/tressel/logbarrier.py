from __future__ import division
import numpy as np
from scipy.sparse.linalg import iterative  as iter
from scipy.sparse import eye as seye
import pylab 
import pprint 
from scipy.optimize import nnls 

import matplotlib.pyplot as plt

from akvo.tressel.SlidesPlot import * 

def PhiB(mux, minVal, x):
    phib = mux * np.abs( np.sum(np.log( x-minVal)) )
    return phib

def curvaturefd(x, y, t):
    x1 = np.gradient(x,t) 
    x2 = np.gradient(x1,t) 
    y1 = np.gradient(y,t) 
    y2 = np.gradient(y1,t) 
    return np.abs(x1*y2 - y1*x2) / np.power(x1**2 + y1**2, 3./2)

def curvatureg(x, y):
    from scipy.ndimage import gaussian_filter1d
    #first and second derivative
    x1 = gaussian_filter1d(x, sigma=1, order=1)#, mode='constant', cval=x[-1])
    x2 = gaussian_filter1d(x1, sigma=1, order=1)#, mode='constant', cval=y[-1])
    y1 = gaussian_filter1d(y, sigma=1, order=1)#, mode='constant', cval=x1[-1])
    y2 = gaussian_filter1d(y1, sigma=1, order=1)#, mode='constant', cval=y1[-1])
    return np.abs(x1*y2 - y1*x2) / np.power(x1**2 + y1**2, 3./2)

def logBarrier(A, b, T2Bins, lambdastar, x_0=0, xr=0, alpha=10, mu1=10, mu2=10, smooth="Smallest", MAXITER=70, fignum=1000, sigma=1, callback=None):
    """Impliments a log barrier Tikhonov solution to a linear system of equations 
        Ax = b  s.t.  x_min < x < x_max. A log-barrier term is used for the constraint
    """
    # TODO input
    minVal = 0.0
    #maxVal = 1e8

    Wd =    (np.eye(len(b)) / (sigma))        # Wd = eye( sigma )
    WdTWd = (np.eye(len(b)) / (sigma**2))     # Wd = eye( sigma )

    ATWdTWdA = np.dot(A.conj().transpose(), np.dot( WdTWd, A ))     # TODO, implicit calculation instead?
    N = np.shape(A)[1]                        # number of model
    M = np.shape(A)[0]                        # number of data
    SIGMA   = .25 # .25 # lower is more aggresive relaxation of log barrier  
    EPSILON = 1e-25 #1e-35 

    # reference model
    if np.size(xr) == 1:
        xr =  np.zeros(N)     
    
    # initial guess
    if np.size(x_0) == 1:
        x = 1e-10 + np.zeros(N)     
    else:
        x = 1e-10 + x_0
        
    # Construct model constraint base   
    Phim_base = np.zeros( [N , N] ) 
    a1 = .05     # smallest too
    
    # calculate largest term            
    D1 = 1./abs(T2Bins[1]-T2Bins[0])
    D2 = 1./abs(T2Bins[2]-T2Bins[1])
    #a2 = 1. #(1./(2.*D1+D2))    # smooth
    
    if smooth == "Both":
        #print ("Both small and smooth model")
        for ip in range(N):
            D1 = 0.
            D2 = 0.
            if ip > 0:
                #D1 = np.sqrt(1./abs(T2Bins[ip]-T2Bins[ip-1]))**.5
                D1 = (1./abs(T2Bins[ip]-T2Bins[ip-1])) #**2
            if ip < N-1:
                #D2 = np.sqrt(1./abs(T2Bins[ip+1]-T2Bins[ip]))**.5
                D2 = (1./abs(T2Bins[ip+1]-T2Bins[ip])) #**2
            if ip > 0:
                Phim_base[ip,ip-1] =   -(D1)      
            if ip == 0:
                Phim_base[ip,ip  ] = 2.*(D1+D2)  
            elif ip == N-1:
                Phim_base[ip,ip  ] = 2.*(D1+D2) 
            else:
                Phim_base[ip,ip  ] = 2.*(D1+D2)
            if ip < N-1:
                Phim_base[ip,ip+1] =   -(D2)  
        Phim_base /= np.max(Phim_base)            # normalize 
        Phim_base += a1*np.eye(N)

    elif smooth == "Smooth":
        #print ("Smooth model")
        for ip in range(N):
            if ip > 0:
                Phim_base[ip,ip-1] = -1    # smooth in log space
            if ip == 0:
                Phim_base[ip,ip  ] = 2.05   # Encourage a little low model
            elif ip == N-1:
                Phim_base[ip,ip  ] = 2.5   # Penalize long decays
            else:
                Phim_base[ip,ip  ] = 2.1   # Smooth and small
            if ip < N-1:
                Phim_base[ip,ip+1] = -1    # smooth in log space

    elif smooth == "Smallest":
        for ip in range(N):
            Phim_base[ip,ip  ] = 1.
    else: 
        print("non valid model constraint:", smooth)
        exit()
    
    Phi_m =  alpha*Phim_base
    WmTWm = Phim_base # np.dot(Phim_base, Phim_base.T)            
    b_pre = np.dot(A, x)
    phid = np.linalg.norm( np.dot(Wd, (b-b_pre)) )**2
    phim = np.linalg.norm( np.dot(Phim_base, (x-xr)) )**2

    mu2 = phim
    phib = PhiB(mu1, 0, x) 
    mu1 = ((phid + alpha*phim) / phib)

    PHIM = []
    PHID = []
    MOD = []

    ALPHA = []
    ALPHA.append(alpha)
    #ALPHA = np.linspace( alpha, 1, MAXITER  )
    print ("{:^5} {:^15} {:^15} {:^15} {:^15} {:^10} {:^10}".format("iter.",  "lambda", "phi_d", "phi_m","phi","kappa","kappa dist."), flush=True) 
    print ("{:^5} {:>15} {:<15} {:<15} {:<15} {:<10} {:<10}".format("-----", "---------------", "---------------","---------------","---------------","----------","----------"), flush=True) 
    for i in range(MAXITER):
        #alpha = ALPHA[i]

        Phi_m =  alpha*Phim_base
        
        # reset mu1 at each iteration 
        # Calvetti -> No ; Li -> Yes   
        # without this, non monotonic convergence occurs...which is OK if you really trust your noise 
        mu1 = ((phid + alpha*phim) / phib) 

        WmTWm = Phim_base # np.dot(Phim_base, Phim_base.T)            
        phid_old = phid
        inner = 0

        First = True # guarantee entry 

        xp = np.copy(x) # prior step x 

        # quick and dirty solution
        #b2a = np.dot(A.conj().transpose(), np.dot(WdTWd, b-b_pre) ) - alpha*np.dot(WmTWm,(x-xr))
        #xg = nnls(ATWdTWdA + Phi_m, b2a)
        #x = xg[0]

        while ( (phib / (phid+alpha*phim)) > EPSILON  or First==True ):
        #while ( False ): # skip the hard stuff

            First = False
            # Log barrier, keep each element above minVal
            X1 = np.eye(N) * (x-minVal)**-1           
            X2 = np.eye(N) * (x-minVal)**-2         
            
            # Log barrier, keep sum below maxVal TODO normalize by component. Don't want to push all down  
            #Y1 = np.eye(N) * (maxVal - np.sum(x))**-1           
            #Y2 = np.eye(N) * (maxVal - np.sum(x))**-2         
            
            AA = ATWdTWdA + mu1*X2 + Phi_m 
            M = np.eye( N ) * (1./np.diag(ATWdTWdA + mu1*X2 + Phi_m))
            #M = seye( N ).dot(1./np.diag(ATWdTWdA + mu1*X2 + Phi_m))
        
            # Solve system (newton step) (Li)
            b2 = np.dot(A.conj().transpose(), np.dot(WdTWd, b-b_pre) ) + 2.*mu1*np.diag(X1) - alpha*np.dot(WmTWm,(x-xr))
            ztilde = iter.cg(AA, b2, M = M) 
            h = (ztilde[0].real) 
            
            # Solve system (direct solution) (Calvetti) 
            #b2 = np.dot(A.conj().transpose(), np.dot(WdTWd, b)) + 2.*mu1*np.diag(X1) - alpha*np.dot(WmTWm,(x-xr))
            #ztilde = iter.cg(AA, b2, M=M, x0=x) 
            #h = (ztilde[0].real - x) 

            # step size
            d = np.min( (1, 0.95 * np.min(x/np.abs(h+1e-120))) )
            
            ##########################################################
            # Update and fix any over/under stepping
            x += d*h
        
            # Determine mu steps to take
            s1 = mu1 * (np.dot(X2, ztilde[0].real) - 2.*np.diag(X1))
            #s2 = mu2 * (np.dot(Y2, ztilde[0].real) - 2.*np.diag(Y1))

            # determine mu for next step
            mu1 = SIGMA/N * np.abs(np.dot(s1, x))
            #mu2 = SIGMA/N * np.abs(np.dot(s2, x))
            
            b_pre = np.dot(A, x)
            phid = np.linalg.norm(np.dot(Wd, (b-b_pre)))**2
            phim = np.linalg.norm( np.dot(Phim_base, (x-xr)) )**2
            phib = PhiB(mu1, minVal, x)
            inner += 1
 
        PHIM.append(phim)      
        PHID.append(phid)      
        MOD.append(np.copy(x))  

        tphi = phid + alpha*phim

        # determine alpha
        scale = 1.5*(len(b)/phid)
        #alpha *= np.sqrt(scale)
        alpha *= min(scale, .95) # was .85...
        #print("alpha", min(scale, 0.99))
        #alpha *= .99 # was .85...
        ALPHA.append(alpha)
        #alpha = ALPHA[i+1]
            
        #print("inversion progress", i, alpha, np.sqrt(phid/len(b)), phim, flush=True)      
        #print ("{:<8} {:<15} {:<10} {:<10}".format(i, alpha, np.sqrt(phid/len(b)), phim), flush=True) 
        
        if i < 4:        
            print ("{:^5} {:>15.4f} {:>15.4f} {:>15.4f} {:>15.4f}".format(i, alpha, np.sqrt(phid/len(b)), phim, tphi ), flush=True) 

#         if np.sqrt(phid/len(b)) < 0.97: 
#             ibreak = -1
#             print ("------------overshot--------------------", alpha, np.sqrt(phid/len(b)), ibreak)
#             alpha *= 2. #0
#             x -= d*h
#             b_pre = np.dot(A, x)
#             phid = np.linalg.norm( np.dot(Wd, (b-b_pre)))**2
#             phim = np.linalg.norm( np.dot(Phim_base, (x-xr)) )#**2
#             mu1 = ((phid + alpha*phim) / phib)
        if lambdastar == "discrepency": 
            if np.sqrt(phid/len(b)) < 1.00 or alpha < 1e-5: 
                ibreak = 1
                print ("optimal solution found", alpha, np.sqrt(phid/len(b)), ibreak)
                break
        # slow convergence, bail and use L-curve 
        # TI- only use L-curve. Otherwise results for perlin noise are too spurious for paper.  
        if lambdastar == "lcurve":
            if i > 4: 
                kappa = curvaturefd(np.log(np.array(PHIM)), np.log(np.array(PHID)), ALPHA[0:i+1])#ALPHA[0:-1])
                #kappa = curvatureg(np.log(np.array(PHIM)), np.log(np.array(PHID)))
                #print("max kappa", np.argmax(kappa), "distance from", i-np.argmax(kappa)) 
                print ("{:^5} {:>15.4f} {:>15.4f} {:>15.4f} {:>15.4f} {:^10} {:^10}".format(i, alpha, np.sqrt(phid/len(b)), phim, tphi, np.argmax(kappa), i-np.argmax(kappa)), flush=True) 
            if i > 4 and (i-np.argmax(kappa)) > 4: # ((np.sqrt(phid_old/len(b))-np.sqrt(phid/len(b))) < 1e-4) : 
            #if np.sqrt(phid/len(b)) < 3.0 and ((np.sqrt(phid_old/len(b))-np.sqrt(phid/len(b))) < 1e-3): 
                ibreak = 1
                MOD = np.array(MOD)
                print ("################################") #slow convergence", alpha, "phid_old", np.sqrt(phid_old/len(b)), "phid", np.sqrt(phid/len(b)), ibreak)
                print ("Using L-curve criteria") 
                #kappa = curvaturefd(np.log(np.array(PHIM)), np.log(np.array(PHID)), ALPHA[0:-1])
                #kappa2 = curvatureg(np.log(np.array(PHIM)), np.log(np.array(PHID)))
                #kappa = curvature( np.array(PHIM), np.array(PHID))
                x = MOD[ np.argmax(kappa) ]
                alphastar = ALPHA[ np.argmax(kappa) ]
                b_pre = np.dot(A, x)
                phid = np.linalg.norm( np.dot(Wd, (b-b_pre)))**2
                phim = np.linalg.norm( np.dot(Phim_base, (x-xr)) )**2
                mu1 = ((phid + alpha*phim) / phib) 
                print ("L-curve selected: iteration=", np.argmax(kappa)) #, " lambda*=", alpha, "phid_old=", np.sqrt(phid_old/len(b)), "phid=", np.sqrt(phid/len(b)), ibreak)
                print ("################################")
                if np.sqrt(phid/len(b)) <= 1:
                    ibreak=0

                fig = plt.figure( figsize=(pc2in(20.0),pc2in(22.)) )
                ax1 = fig.add_axes( [.2,.15,.6,.7] )
                #plt.plot( (np.array(PHIM)),  np.log(np.array(PHID)/len(b)), '.-')
                #plt.plot(  ((np.array(PHIM))[np.argmax(kappa)]) , np.log( (np.array(PHID)/len(b))[np.argmax(kappa)] ), '.', markersize=12)
                #plt.axhline()
                lns1 = plt.plot( np.log(np.array(PHIM)),  np.log(np.sqrt(np.array(PHID)/len(b))), '.-', label="L curve")
                lns2 = plt.plot( np.log(np.array(PHIM))[np.argmax(kappa)], np.log(np.sqrt(np.array(PHID)/len(b))[np.argmax(kappa)]), '.', markersize=12, label="$\lambda^*$")
                ax2 = plt.twinx()
                lns3 = ax2.plot( np.log(np.array(PHIM)), kappa, color='orange', label="curvature" )

                # Single legend 
                lns = lns1+lns3
                labs = [l.get_label() for l in lns]
                ax2.legend(lns, labs, loc=0)

                ax1.set_xlabel("$\phi_m$")
                ax1.set_ylabel("$\phi_d$")
                
                ax2.set_ylabel("curvature")

                plt.savefig('lcurve.pdf')
                break

    PHIM = np.array(PHIM)
    PHID = np.array(PHID)

    if (i == MAXITER-1 ):
        ibreak = 2
        #print("Reached max iterations!!", alpha, np.sqrt(phid/len(b)), ibreak)
        #kappa = curvaturefd(np.log(np.array(PHIM)), np.log(np.array(PHID)), ALPHA[0:-1])
        x = MOD[-1]
        b_pre = np.dot(A, x)
        phid = np.linalg.norm( np.dot(Wd, (b-b_pre)))**2
        phim = np.linalg.norm( np.dot(Phim_base, (x-xr)) )**2
        mu1 = ((phid + alpha*phim) / phib) 

    if lambdastar == "lcurve":
        #print("Returning L curve result")
        return x, ibreak, np.sqrt(phid/len(b)), PHIM, PHID/len(b), np.argmax(kappa), Wd, Phim_base, alphastar
    else:
        print("Returning max iteration result")
        return x, ibreak, np.sqrt(phid/len(b))



if __name__ == "__main__":
    print("Test")
