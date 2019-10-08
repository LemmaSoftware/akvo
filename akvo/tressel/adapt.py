import numpy as np
from numpy.linalg import lstsq
from numpy.linalg import norm
from numpy import fft 

import pylab

from scipy.signal import correlate

def autocorr(x):
    #result = np.correlate(x, x, mode='full')
    result = correlate(x, x, mode='full')
    return result[result.size/2:]

class AdaptiveFilter:

    def __init__(self, mu):
        self.mu = mu

    def adapt_filt_Ref(self, x, R, M, mu, PCA, lambda2=0.95, H0=0):
        """ Taken from .m file
        This function is written to allow the user to filter a input signal   
        with an adaptive filter that utilizes 2 reference signals instead of  
        the standard method which allows for only 1 reference signal.         
        Author: Rob Clemens              Date: 3/16/06                       
        Modified and ported to Python, now takes arbitray number of reference points 
        Original public domain source 
        https://www.mathworks.com/matlabcentral/fileexchange/10447-noise-canceling-adaptive-filter 
            x = data array 
            R = reference array 
            M = number of taps 
            mu = forgetting factor 
            PCA = Perform PCA 
        """
        #from akvo.tressel import pca 
        import akvo.tressel.pca as pca 

        if np.shape(x) != np.shape(R[0]): # or np.shape(x) != np.shape(rx1):
            print ("Error, non aligned")
            exit(1)
        
        if PCA == "Yes":
            #print("Performing PCA calculation in noise cancellation")
            # PCA decomposition on ref channels so signals are less related
            R, K, means = pca.pca( R )
           
            # test for in loop reference  
            #print("Cull nearly zero terms?", np.shape(x), np.shape(R))     
            #R = R[0:3,:] 
            #R = R[2:4,:] 
            #print("    removed zero terms?", np.shape(x), np.shape(R))     
            #H0 = H0[0:3*np.shape(x)[0]]
            #H0 = H0[0:2*np.shape(x)[0]]

        if all(H0) == 0:
            # corrects for dimensionality issues if a simple 0 is passed
            H = np.zeros( (len(R)*M))
        else:
            H = H0

        Rn = np.ones(len(R)*M) / mu 
        
        r_ = np.zeros( (len(R), M) ) 
        e = np.zeros(len(x))           # error, in our case the desired output
        ilambda = lambda2**-1

        for ix in range(0, len(x)):
            # Only look forwards, to avoid distorting the lates times 
            # (run backwards, if opposite and you don't care about distorting very late time.)
            for ir in range(len(R)):  # number of reference channels 
                if ix < M:
                    r_[ir,0:ix] = R[ir][0:ix]
                    r_[ir,ix:M] = 0 
                else:
                    r_[ir,:] = R[ir][ix-M:ix]

            # reshape            
            r_n = np.reshape(r_, -1) # concatenate the ref channels in to a 1D array 

            K      = (Rn* r_n) / (lambda2 + np.dot(r_n*Rn, r_n))       # Create/update K
            e[ix]  = x[ix] - np.dot(r_n.T, H)                          # e is the filtered signal, input - r(n) * Filter Coefs
            H     += K*e[ix];                                          # Update Filter Coefficients
            Rn     = ilambda*Rn - ilambda*np.dot(np.dot(K, r_n.T), Rn) # Update R(n)
        return e, H

    def transferFunctionFFT(self, D, R, reg=1e-2):
         from akvo.tressel import pca
         """
             Computes the transfer function (H) between a Data channel and 
             a number of Reference channels. The Matrices D and R are 
             expected to be in the frequency domain on input.
             | R1'R1 R1'R2 R1'R3|   |h1|   |R1'D|
             | R2'R1 R2'R2 R2'R3| * |h2| = |R2'D|
             | R3'R1 R3'R2 R3'R3|   |h3|   |R3'D|
 
             Returns the corrected array 
         """
 
         # PCA decomposition on ref channels so signals are less related
         #transMatrix, K, means = pca.pca( np.array([rx0, rx1]))   
         #RR = np.zeros(( np.shape(R[0])[0]*np.shape(R[0])[1], len(R)))
 #         RR = np.zeros(( len(R), np.shape(R[0])[0]*np.shape(R[0])[1] ))
 #         for ir in range(len(R)):
 #             RR[ir,:] = np.reshape(R[ir], -1)
 #         transMatrix, K, means = pca.pca(RR)    
 #         #R rx0 = transMatrix[0,:]
 #         # rx1 = transMatrix[1,:]
 #         for ir in range(len(R)):
 #             R[ir] = transMatrix[ir,0]
 
         import scipy.linalg 
         import akvo.tressel.pca as pca 
         # Compute as many transfer functions as len(R)
         # A*H = B
         nref = len(R)
         H = np.zeros( (np.shape(D)[1], len(R)), dtype=complex )
         for iw in range(np.shape(D)[1]):
             A = np.zeros( (nref, nref), dtype=complex )
             B = np.zeros( (nref) , dtype=complex)
             for ii in range(nref):
                 for jj in range(nref):
                     # build A
                     A[ii,jj] = np.dot(R[ii][:,iw], R[jj][:,iw])                 
 
                 # build B
                 B[ii] = np.dot( R[ii][:,iw], D[:,iw] )
 
             # compute H(iw)
             #linalg.solve(a,b) if a is square
             #print "A", A
             #print "B", B
             # TODO, regularise this solve step? So as to not fit the spurious noise
             #print np.shape(B), np.shape(A) 
             #H[iw, :] = scipy.linalg.solve(A,B)
             H[iw, :] = scipy.linalg.lstsq(A,B,cond=reg)[0]
             #print "lstqt", np.shape(scipy.linalg.lstsq(A,B))
             #print "solve", scipy.linalg.solve(A,B)
             #H[iw,:]  = scipy.linalg.lstsq(A,B) # otherwise 
                 #H = np.zeros( (np.shape(D)[1], )   )
         #print H #A, B
         Error = np.zeros(np.shape(D), dtype=complex)
         for ir in range(nref):
             for q in range( np.shape(D)[0] ):
                 #print "dimcheck", np.shape(H[:,ir]), np.shape(R[ir][q,:] )
                 Error[q,:] += H[:,ir]*R[ir][q,:]
         return D - Error

