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
        """
        #from akvo.tressel import pca 
        import akvo.tressel.pca as pca 

        if np.shape(x) != np.shape(R[0]): # or np.shape(x) != np.shape(rx1):
            print ("Error, non aligned")
            exit(1)
        
        if PCA == "Yes":
            print("Performing PCA calculation in noise cancellation")
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
            H = np.zeros( (len(R)*M))
            #print ("resetting filter")
        else:
            H = H0

        Rn = np.ones(len(R)*M) / mu 
        
        r_ = np.zeros( (len(R), M) ) 
        e = np.zeros(len(x)) # error, desired output
        ilambda = lambda2**-1

        for z in range(0, len(x)):
            # Only look forwards, to avoid distorting the lates times 
            # (run backwards, if opposite and you don't care about distorting very late time.)
            for ir in range(len(R)):
                if z < M:
                    r_[ir,0:z] = R[ir][0:z]
                    r_[ir,z:M] = 0 
                else:
                    # TODO, use np.delete and np.append to speed this up
                    r_[ir,:] = R[ir][z-M:z]
            # reshape            
            r_n = np.reshape(r_, -1) #concatenate((r_v, r_h ))

            #K      = np.dot( np.diag(Rn,0), r_n) / (lambda2 + np.dot(r_n*Rn, r_n))  # Create/update K
            K      = (Rn* r_n) / (lambda2 + np.dot(r_n*Rn, r_n))  # Create/update K
            e[z]   = x[z] - np.dot(r_n.T, H)             # e is the filtered signal, input - r(n) * Filter Coefs
            H     += K*e[z];                             # Update Filter Coefficients
            Rn     = ilambda*Rn - ilambda*np.dot(np.dot(K, r_n.T), Rn)     # Update R(n)
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

    def adapt_filt_tworefFreq(self, x, rx0, rx1, M, lambda2=0.95):
        """ Frequency domain version of above
        """
        from akvo.tressel import pca 

        pylab.figure()
        pylab.plot(rx0)
        pylab.plot(rx1)

        # PCA decomposition on ref channels so signals are less related
        transMatrix, K, means = pca.pca( np.array([rx0, rx1]))    
        rx0 = transMatrix[:,0]
        rx1 = transMatrix[:,1]
        
        pylab.plot(rx0)
        pylab.plot(rx1)
        pylab.show()
        exit()

        if np.shape(x) != np.shape(rx0) or np.shape(x) != np.shape(rx1):
            print ("Error, non aligned")
            exit(1)

        wx = fft.rfft(x)
        wr0 = fft.rfft(rx0)
        wr1 = fft.rfft(rx1)
 
        H = np.zeros( (2*M), dtype=complex ) 
        ident_mat = np.eye((2*M))
        Rn = ident_mat / 0.1
        r_v = np.zeros( (M), dtype=complex ) 
        r_h = np.zeros( (M), dtype=complex ) 
        e = np.zeros(len(x), dtype=complex )
        ilambda = lambda2**-1

        for z in range(0, len(wx)):
            # TODO Padd with Zeros or truncate if M >,< arrays 
            r_v = wr0[::-1][:M] 
            r_h = wr1[::-1][:M] 
            r_n = np.concatenate((r_v, r_h ))
            K      = np.dot(Rn, r_n) / (lambda2 + np.dot(np.dot(r_n.T, Rn), r_n))  # Create/update K
            e[z]   = wx[z] - np.dot(r_n,H)        # e is the filtered signal, input - r(n) * Filter Coefs
            H     += K * e[z];                    # Update Filter Coefficients
            Rn     = ilambda*Rn - ilambda*K*r_n.T*Rn  # Update R(n)
        
        return fft.irfft(e)

    def iter_filt_refFreq(self, x, rx0, Ahat=.05, Bhat=.5, k=0.05):

        X = np.fft.rfft(x)
        X0 = np.copy(X)
        RX0 = np.fft.rfft(rx0)

        # step 0
        Abs2HW = []
        alphai =  k * (np.abs(Ahat)**2 / np.abs(Bhat)**2) 
        betai  =  k * (1. / (np.abs(Bhat)**2) ) 
        Hw     =  ((1.+alphai) * np.abs(X)**2 ) / (np.abs(X)**2 + betai*(np.abs(RX0)**2))
        H      =  np.abs(Hw)**2
        pylab.ion()
        pylab.figure()
        for i in range(10):
            #print "alphai", alphai
            #print "betai", betai
            #print "Hw", Hw
            alphai = k * (np.abs(Ahat)**2 / np.abs(Bhat)**2) * np.product(H, axis=0)
            betai  = k * (1. / np.abs(Bhat)**2) * np.product(H, axis=0)
            # update signal
            Hw   =  ((1.+alphai) * np.abs(X)**2) / (np.abs(X)**2 + betai*np.abs(RX0)**2)
            Hw = np.nan_to_num(Hw)
            X *= Hw
            H = np.vstack( (H, np.abs(Hw)**2) )
            #print "Hw", Hw
            pylab.cla()
            pylab.plot(Hw)
            #pylab.plot(np.abs(X))
            #pylab.plot(np.abs(RX0))
            pylab.draw()
            raw_input("wait")

        pylab.cla()
        pylab.ioff()
        #return np.fft.irfft(X0-X)
        return np.fft.irfft(X)

    def iter_filt_refFreq(self, x, rx0, rx1, Ahat=.1, Bhat=1., k=0.001):

        X = np.fft.rfft(x)
        X0 = np.copy(X)
        RX0 = np.fft.rfft(rx0)
        RX1 = np.fft.rfft(rx1)

        # step 0
        alphai =  k * (np.abs(Ahat)**2 / np.abs(Bhat)**2) 
        betai  =  k * (1. / (np.abs(Bhat)**2) ) 
        #Hw     =  ((1.+alphai) * np.abs(X)**2 ) / (np.abs(X)**2 + betai*(np.abs(RX0)**2))
        H      =  np.ones(len(X)) # abs(Hw)**2
        #pylab.ion()
        #pylab.figure(334)
        for i in range(1000):
            #print "alphai", alphai
            #print "betai", betai
            #print "Hw", Hw
            alphai = k * (np.abs(Ahat)**2 / np.abs(Bhat)**2) * np.product(H, axis=0)
            betai  = k * (1. / np.abs(Bhat)**2) * np.product(H, axis=0)
            # update signal
            Hw   =  ((1.+alphai) * np.abs(X)**2) / (np.abs(X)**2 + betai*np.abs(RX0)**2)
            Hw = np.nan_to_num(Hw)
            X *= Hw #.conjugate
            #H = np.vstack((H, np.abs(Hw)**2) )
            H = np.vstack((H, np.abs(Hw)) )
            #print "Hw", Hw
            #pylab.cla()
            #pylab.plot(Hw)
            #pylab.plot(np.abs(X))
            #pylab.plot(np.abs(RX0))
            #pylab.draw()
            #raw_input("wait")

        #pylab.cla()
        #pylab.ioff()
        return np.fft.irfft(X0-X)
        #return np.fft.irfft(X)

    def Tdomain_DFT(self, desired, input, S):
        """ Lifted from Adaptive filtering toolbox. Modefied to accept more than one input 
            vector
        """
        
        # Initialisation Procedure
        nCoefficients =   S["filterOrderNo"]/2+1
        nIterations   =   len(desired)

        # Pre Allocations
        errorVector  = np.zeros(nIterations, dtype='complex')
        outputVector = np.zeros(nIterations, dtype='complex')
        
        # Initial State
        coefficientVectorDFT =   np.fft.rfft(S["initialCoefficients"])/np.sqrt(float(nCoefficients))
        desiredDFT           =   np.fft.rfft(desired)
        powerVector          =   S["initialPower"]*np.ones(nCoefficients)

        # Improve source code regularity, pad with zeros
        # TODO, confirm zeros(nCoeffics) not nCoeffics-1
        prefixedInput  =   np.concatenate([np.zeros(nCoefficients-1), np.array(input)])

        # Body
        pylab.ion()
        pylab.figure(11)
        for it in range(nIterations): # = 1:nIterations,
            
            regressorDFT = np.fft.rfft(prefixedInput[it:it+nCoefficients][::-1]) /\
                           np.sqrt(float(nCoefficients))

            # Summing two column vectors
            powerVector = S["alpha"] * (regressorDFT*np.conjugate(regressorDFT)) + \
                                  (1.-S["alpha"])*(powerVector)

            pylab.cla()
            #pylab.plot(prefixedInput[::-1], 'b')
            #pylab.plot(prefixedInput[it:it+nCoefficients][::-1], 'g', linewidth=3)
            #pylab.plot(regressorDFT.real)
            #pylab.plot(regressorDFT.imag)
            pylab.plot(powerVector.real)
            pylab.plot(powerVector.imag)
            #pylab.plot(outputVector)
            #pylab.plot(errorVector.real)
            #pylab.plot(errorVector.imag)
            pylab.draw()
            #raw_input("wait")

            outputVector[it] = np.dot(coefficientVectorDFT.T, regressorDFT)

            #errorVector[it] = desired[it] - outputVector[it]
            errorVector[it] = desiredDFT[it] - outputVector[it]

            #print errorVector[it], desired[it], outputVector[it]

            # Vectorized
            coefficientVectorDFT += (S["step"]*np.conjugate(errorVector[it])*regressorDFT) /\
                                    (S['gamma']+powerVector)

        return np.real(np.fft.irfft(errorVector))
        #coefficientVector = ifft(coefficientVectorDFT)*sqrt(nCoefficients);

    def Tdomain_DCT(self, desired, input, S):
        """ Lifted from Adaptive filtering toolbox. Modefied to accept more than one input 
            vector. Uses cosine transform
        """
        from scipy.fftpack import dct
 
        # Initialisation Procedure
        nCoefficients =   S["filterOrderNo"]+1
        nIterations   =   len(desired)

        # Pre Allocations
        errorVector  = np.zeros(nIterations)
        outputVector = np.zeros(nIterations)
        
        # Initial State
        coefficientVectorDCT =   dct(S["initialCoefficients"]) #/np.sqrt(float(nCoefficients))
        desiredDCT           =   dct(desired)
        powerVector          =   S["initialPower"]*np.ones(nCoefficients)

        # Improve source code regularity, pad with zeros
        prefixedInput  =   np.concatenate([np.zeros(nCoefficients-1), np.array(input)])
        
        # Body
        #pylab.figure(11)
        #pylab.ion()
        for it in range(0, nIterations): # = 1:nIterations,
            
            regressorDCT = dct(prefixedInput[it:it+nCoefficients][::-1], type=2) 
            #regressorDCT = dct(prefixedInput[it+nCoefficients:it+nCoefficients*2+1])#[::-1]) 

            # Summing two column vectors
            powerVector = S["alpha"]*(regressorDCT) + (1.-S["alpha"])*(powerVector)
            #pylab.cla()
            #pylab.plot(powerVector)
            #pylab.draw()

            outputVector[it] = np.dot(coefficientVectorDCT.T, regressorDCT)
            #errorVector[it] = desired[it] - outputVector[it]
            errorVector[it] = desiredDCT[it] - outputVector[it]

            # Vectorized
            coefficientVectorDCT += (S["step"]*errorVector[it]*regressorDCT) #/\
                                    #(S['gamma']+powerVector)

        #pylab.plot(errorVector)
        #pylab.show()
        return dct(errorVector, type=3)
        #coefficientVector = ifft(coefficientVectorDCT)*sqrt(nCoefficients);



    def Tdomain_CORR(self, desired, input, S):

        from scipy.linalg import toeplitz
        from scipy.signal import correlate

        # Autocorrelation
        ac = np.correlate(input, input, mode='full')
        ac = ac[ac.size/2:]
        R = toeplitz(ac)
        
        # cross correllation
        r = np.correlate(desired, input, mode='full')
        r = r[r.size/2:]
        
        #r = np.correlate(desired, input, mode='valid')
        print ("R", np.shape(R))
        print ("r", np.shape(r))
        print ("solving")
        #H = np.linalg.solve(R,r)
        H = np.linalg.lstsq(R,r,rcond=.01)[0]
        #return desired - np.dot(H,input)
        print ("done solving")
        pylab.figure()
        pylab.plot(H)
        pylab.title("H")
        #return desired - np.convolve(H, input, mode='valid')
        #return desired - np.convolve(H, input, mode='same')
        #return np.convolve(H, input, mode='same')
        return desired - np.dot(toeplitz(H), input)
        #return np.dot(R, H)

#         T = toeplitz(input)
#         print "shapes", np.shape(T), np.shape(desired)
#         h = np.linalg.lstsq(T, desired)[0]
#         print "shapes", np.shape(h), np.shape(input)
#         #return np.convolve(h, input, mode='same')
#         return desired - np.dot(T,h)
 
    def Fdomain_CORR(self, desired, input, dt, freq):
        
        from scipy.linalg import toeplitz
        
        # Fourier domain
        Input = np.fft.rfft(input)
        Desired = np.fft.rfft(desired)

        T = toeplitz(Input)
        #H = np.linalg.solve(T, Desired)
        H = np.linalg.lstsq(T, Desired)[0]
#         ac = np.correlate(Input, Input, mode='full')
#         ac = ac[ac.size/2:]
#         R = toeplitz(ac)
#         
#         r = np.correlate(Desired, Input, mode='full')
#         r = r[r.size/2:]
#         
#         #r = np.correlate(desired, input, mode='valid')
#         print "R", np.shape(R)
#         print "r", np.shape(r)
#         print "solving"
#         H = np.linalg.solve(R,r)
#         #H = np.linalg.lstsq(R,r)
#         #return desired - np.dot(H,input)
#         print "done solving"
        pylab.figure()
        pylab.plot(H.real)
        pylab.plot(H.imag)
        pylab.plot(Input.real)
        pylab.plot(Input.imag)
        pylab.plot(Desired.real)
        pylab.plot(Desired.imag)
        pylab.legend(["hr","hi","ir","ii","dr","di"])
        pylab.title("H")
        #return desired - np.fft.irfft(Input*H)
        return np.fft.irfft(H*Input)

    def Tdomain_RLS(self, desired, input, S):
        """
            A DFT is first performed on the data. Than a RLS algorithm is carried out 
            for noise cancellation. Related to the RLS_Alt Algoritm 5.3 in  Diniz book.
            The desired and input signals are assummed to be real time series data.
        """

        # Transform data into frequency domain
        Input = np.fft.rfft(input)
        Desired = np.fft.rfft(desired)

        # Initialisation Procedure
        nCoefficients = S["filterOrderNo"]+1
        nIterations   = len(Desired)

        # Pre Allocations
        errorVector  = np.zeros(nIterations, dtype="complex")
        outputVector = np.zeros(nIterations, dtype="complex")
        errorVectorPost  = np.zeros(nIterations, dtype="complex")
        outputVectorPost = np.zeros(nIterations, dtype="complex")
        coefficientVector = np.zeros( (nCoefficients, nIterations+1), dtype="complex" )        

        # Initial State
        coefficientVector[:,1] = S["initialCoefficients"]  
        S_d                    = S["delta"]*np.eye(nCoefficients)

        # Improve source code regularity, pad with zeros
        prefixedInput = np.concatenate([np.zeros(nCoefficients-1, dtype="complex"), 
                                np.array(Input)])
        invLambda = 1./S["lambda"]
        
        # Body
        pylab.ion()
        pylab.figure(11)

        for it in range(nIterations):
            
            regressor = prefixedInput[it:it+nCoefficients][::-1]

            # a priori estimated output
            outputVector[it] = np.dot(coefficientVector[:,it].T, regressor)
       
            # a priori error
            errorVector[it] = Desired[it] - outputVector[it]

            psi             = np.dot(S_d, regressor)
            if np.isnan(psi).any():
                print ("psi", psi)
                exit(1)
            
            pylab.cla()
            #pylab.plot(psi)
            pylab.plot(regressor.real)
            pylab.plot(regressor.imag)
            pylab.plot(coefficientVector[:,it].real)
            pylab.plot(coefficientVector[:,it].imag)
            pylab.legend(["rr","ri", "cr", "ci"])
            pylab.draw()
            raw_input("paws")

            S_d             = invLambda * (S_d - np.dot(psi, psi.T)  /\
                                S["lambda"] + np.dot(psi.T, regressor))

            coefficientVector[:,it+1] = coefficientVector[:,it] + \
                                        np.conjugate(errorVector[it])*np.dot(S_d, regressor)
            # A posteriori estimated output
            outputVectorPost[it]  =  np.dot(coefficientVector[:,it+1].T, regressor)

            # A posteriori error
            errorVectorPost[it] = Desired[it] - outputVectorPost[it]
 
        errorVectorPost = np.nan_to_num(errorVectorPost)

        pylab.figure(11)
        print (np.shape(errorVectorPost))
        pylab.plot(errorVectorPost.real)
        pylab.plot(errorVectorPost.imag)
        pylab.show()
        print(errorVectorPost)
        #return np.fft.irfft(Desired)
        return np.fft.irfft(errorVectorPost)

if __name__ == "__main__":

    def noise(nu, t, phi):
        return np.sin(nu*2.*np.pi*t + phi)

    import matplotlib.pyplot as plt
    print("Test driver for adaptive filtering")
    Filt = AdaptiveFilter(.1)
    t = np.arange(0, .5, 1e-4)
    omega = 2000 * 2.*np.pi
    T2 = .100
    n1 = noise(60, t, .2   )
    n2 = noise(61, t, .514 )
    x = np.sin(omega*t)* np.exp(-t/T2) + 2.3*noise(60, t, .34) + 1.783*noise(31, t, 2.1)
    e = Filt.adapt_filt_tworef(x, n1, n2, 200, .98)
    plt.plot(t,  x)
    plt.plot(t, n1)
    plt.plot(t, n2)
    plt.plot(t,  e)
    plt.show()
