import numpy, array #,rpy2
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import least_squares

#from rpy2.robjects.packages import importr
#import rpy2.robjects as robjects
#import rpy2.robjects.numpy2ri

#import notch
from numpy.fft import fft, fftfreq

# We know/can calculate frequency peak, use this to guess where picks will be.
# maybe have a sliding window that reports peak values.
def peakPicker(data, omega, dt):

    # compute window based on omega and dt
    # make sure you are not aliased, grab every other peak
    window = (2*numpy.pi) / (omega*dt)

    data = numpy.array(data) 
    peaks = []
    troughs = []
    times = []
    times2 = []
    indices = []
    ws = 0
    we = window
    ii = 0
    for i in range((int)(len(data)/window)):
        
        # initially was just returning this I think avg is better
        #times.append( (ws + numpy.abs(data[ws:we]).argmax()) * dt )
    
        peaks.append(numpy.max(data[ws:we]))
        times.append( (ws + data[ws:we].argmax()) * dt )
        indices.append( ii + data[ws:we].argmax() )        

        troughs.append(numpy.min(data[ws:we]))
        times2.append( (ws + (data[ws:we]).argmin()) * dt )
        indices.append( ii + data[ws:we].argmin() )        

        ws += window
        we += window
        ii += (int)(we-ws)
    
    #return numpy.array(peaks), numpy.array(times)
    
    # Averaging peaks does a good job of removing bias in noise
    return (numpy.array(peaks)-numpy.array(troughs))/2., \
        (numpy.array(times)+numpy.array(times2))/2., \
        indices           


def fun(x, t, y):
    """ Cost function for regression, single exponential, no DC term 
        x[0] = A0
        x[1] = zeta 
        x[2] = df
        x[3] = T2
    """
    # concatenated real and imaginary parts  
    return y - np.concatenate((-x[0]*np.sin(2.*np.pi*x[2]*t + x[1])*np.exp(-t/x[3]), \
                               +x[0]*np.cos(2.*np.pi*x[2]*t + x[1])*np.exp(-t/x[3])))  

def fun2(x, t, y):
    """ Cost function for regression, single exponential, no DC term 
        x[0] = A0
        x[1] = zeta 
        x[2] = T2
    """
    # concatenated real and imaginary parts  
    pre =  np.concatenate((x[0]*np.cos(x[1])*np.exp(-t/x[2]), \
                          -x[0]*np.sin(x[1])*np.exp(-t/x[2])))  
    return y-pre


def quadratureDetect2(X, Y, tt, method, loss, x0="None"): 
    """ Pure python quadrature detection using Scipy.  
        X = real part of NMR signal 
        Y = imaginary component of NMR signal 
        tt = time 
    """

    #method = ['trf','dogbox','lm'][method_int]
    #loss = ['linear','soft_l1','cauchy','huber'][loss_int] 
    #print ("method", method, 'loss', loss) 
    if x0=="None":
        if method == 'lm':
            x0 = np.array( [50., 0., 0., .200] ) # A0, zeta, df, T2 
            res_lsq = least_squares(fun, x0, args=(tt, np.concatenate((X, Y))), loss=loss, f_scale=1.0,\
                    method=method 
                    )
        else:
            x0 = np.array( [50., 0., 0., .200] ) # A0, zeta, df, T2 
            res_lsq = least_squares(fun, x0, args=(tt, np.concatenate((X, Y))), loss=loss, f_scale=1.0,\
                    bounds=( [5, -np.pi, -5, .001] , [5000., np.pi, 5, .800] ),
                    method=method 
                    )
        x = res_lsq.x 
        #print ("A0={} zeta={} df={} T2={}".format(x[0],x[1],x[2],x[3]))
    else:
        res_lsq = least_squares(fun, x0, args=(tt, np.concatenate((X, Y))), loss=loss, f_scale=1.0,\
            #bounds=( [1., -np.pi, -5, .005] , [1000., np.pi, 5, .800] ),
            method=method 
            )

        #bounds=( [0., 0, -20, .0] , [1., np.pi, 20, .6] ))

    x = res_lsq.x 
    return res_lsq.success, x[0], x[2], x[1], x[3]
    
    # no df
    #x = np.array( [1., 0., 0.2] )
    #res_lsq = least_squares(fun2, x, args=(tt, np.concatenate((X, Y))), loss='soft_l1', f_scale=0.1)
    #x = res_lsq.x 
    #return conv, E0,df,phi,T2
    #return res_lsq.success, x[0], 0, x[1], x[2]



###################################################################
###################################################################
###################################################################
if __name__ == "__main__":

    dt    = .0001
    T2    = .1
    omega = 2000.*2*numpy.pi
    phi   = .0
    T     = 8.*T2
    
    t = numpy.arange(0, T, dt)

    # Synthetic data, simple single decaying sinusoid 
    # with a single decay parameter and gaussian noise added 
    data = numpy.exp(-t/T2) * numpy.sin(omega * t + phi) + numpy.random.normal(0,.05,len(t)) \
                         + numpy.random.randint(-1,2,len(t))*numpy.random.exponential(.2,len(t)) 
    cdata = numpy.exp(-t/T2) * numpy.sin(omega * t + phi) #+ numpy.random.normal(0,.25,len(t))
    #data = numpy.random.normal(0,.25,len(t))

    sigma2 = numpy.std(data[::-len(data)/4])
    #sigma2 = numpy.var(data[::-len(data)/4])
    print("sigma2", sigma2)    
    
    [peaks,times,indices] = peakPicker(data, omega, dt)
    
    [b1,b2,rT2] = regressCurve(peaks,times)
    print("rT2 nonweighted", rT2)
    
    [b1,b2,rT2] = regressCurve(peaks,times,sigma2)
    print("rT2 weighted", rT2)

    envelope   =  numpy.exp(-t/T2)
    renvelope  =  numpy.exp(-t/rT2)

    #outf = file('regress.txt','w')
    #for i in range(len(times)):
    #    outf.write(str(times[i]) + "   " +  str(peaks[i]) + "\n")  
    #outf.close()

    plt.plot(t,data, 'b')
    plt.plot(t,cdata, 'g', linewidth=1)
    plt.plot(t,envelope, color='violet', linewidth=4)
    plt.plot(t,renvelope, 'r', linewidth=4)
    plt.plot(times, numpy.array(peaks), 'bo', markersize=8, alpha=.25)
    plt.legend(['noisy data','clean data','real envelope','regressed env','picks'])
    plt.savefig("regression.pdf")


    # FFT check
    fourier = fft(data)
    plt.figure()
    freq = fftfreq(len(data), d=dt)
    plt.plot(freq, (fourier.real))
    
    plt.show()

    # TODO do a bunch in batch mode to see if T2 estimate is better with or without 
    # weighting and which model is best.

    # TODO try with real data

    # TODO test filters (median, FFT, notch)

    # It looks like weighting is good for relatively low sigma, but for noisy data
    # it hurts us. Check
