import numpy, array #,rpy2
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from rpy2.robjects.packages import importr

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

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


#################################################
# Regress for T2 using rpy2 interface
def regressCurve(peaks,times,sigma2=1,intercept=True):

    # TODO, if regression fails, it might be because there is no exponential
    # term, maybe do a second regression then on a linear model. 
    b1  = 0                  # Bias
    b2  = 0                  # Linear 
    rT2 = 0.3                # T2 regressed
    r   = robjects.r         

    # Variable shared between R and Python
    robjects.globalenv['b1'] = b1
    robjects.globalenv['b2'] = b2
    robjects.globalenv['rT2'] = rT2
    robjects.globalenv['sigma2'] = sigma2
    value = robjects.FloatVector(peaks)
    times = robjects.FloatVector(numpy.array(times))
    
#    my_weights = robjects.RVector(value/sigma2)
#    robjects.globalenv['my_weigts'] = my_weights

#    if sigma2 != 0:
#        print "weighting"
#        tw = numpy.array(peaks)/sigma2 
#        my_weights = robjects.RVector( tw/numpy.max(tw) )
#    else:
#        my_weights = robjects.RVector(numpy.ones(len(peaks))) 

#    robjects.globalenv['my_weights'] = my_weights
    
    if (intercept):
        my_list = robjects.r('list(b1=50, b2=1e2, rT2=0.03)')
        my_lower = robjects.r('list(b1=0, b2=0, rT2=.005)')
        my_upper = robjects.r('list(b1=20000, b2=2000, rT2=.700)')
    else:
        my_list = robjects.r('list(b2=1e2, rT2=0.3)')
        my_lower = robjects.r('list(b2=0, rT2=.005)')
        my_upper = robjects.r('list(b2=2000, rT2=.700)')

    my_cont = robjects.r('nls.control(maxiter=1000, warnOnly=TRUE, printEval=FALSE)')

    
    if (intercept):
        #fmla = robjects.RFormula('value ~ b1 + exp(-times/rT2)')
        fmla = robjects.Formula('value ~ b1 + b2*exp(-times/rT2)')
        #fmla = robjects.RFormula('value ~ b1 + b2*times + exp(-times/rT2)')
    else:
        fmla = robjects.Formula('value ~ b2*exp(-times/rT2)')

    env = fmla.getenvironment()
    env['value'] = value
    env['times'] = times
    
    # ugly, but I get errors with everything else I've tried
    my_weights = robjects.r('rep(1,length(value))')
    for ii in range(len(my_weights)):
        my_weights[ii] *= peaks[ii]/sigma2
    Error = False
    #fit = robjects.r.nls(fmla,start=my_list,control=my_cont,weights=my_weights)
    if (sigma2 != 1):
        print("SIGMA 2")
        #fit = robjects.r.tryCatch(robjects.r.suppressWarnings(robjects.r.nls(fmla,start=my_list,control=my_cont,algorithm="port", \
        #                     weights=my_weights)), 'silent=TRUE')
        fit = robjects.r.tryCatch(robjects.r.nls(fmla,start=my_list,control=my_cont))#, \
                            # weights=my_weights))
    else:
        try:
            fit = robjects.r.tryCatch(robjects.r.nls(fmla,start=my_list,control=my_cont,algorithm="port"))#,lower=my_lower,upper=my_upper))
        except:
            print("regression issue pass")
            Error = True
    # If failure fall back on zero regression values   
    if not Error:
        #Error = fit[3][0]
        report =  r.summary(fit)
    b1 = 0
    b2 = 0 
    rT2 = 1
    if (intercept):
        if not Error:
            b1  =  r['$'](report,'par')[0]
            b2  =  r['$'](report,'par')[1]
            rT2 =  r['$'](report,'par')[2]
            #print  report
            #print  r['$'](report,'convergence')
            #print  r['convergence'] #(report,'convergence')
            #print  r['$'](report,'par')[13]
            #print  r['$'](report,'par')[14]
        else:
            print("ERROR DETECTED, regressed values set to default")
            b1 = 1e1
            b2 = 1e-2
            rT2 = 1e-2
            #print r['$'](report,'par')[0]
            #print r['$'](report,'par')[1]
            #print r['$'](report,'par')[2]
        return [b1,b2,rT2] 
    else:
        if not Error:
            rT2 =  r['$'](report,'par')[1]
            b2  =  r['$'](report,'par')[0]
        else:
            print("ERROR DETECTED, regressed values set to default")
        return [b2, rT2] 

#################################################
# Regress for T2 using rpy2 interface
def regressCurve2(peaks,times,sigma2=[None],intercept=True):

    if sigma2[0] != None:
        my_weights = robjects.FloatVector( sigma2 )

    # TODO, if regression fails, it might be because there is no exponential
    # term, maybe do a second regression then on a linear model. 
    b1  = 0                  # Bias
    b2  = 0                  # Linear 
    bb2  = 0                 # Linear 
    rT2 = 0.3                # T2 regressed
    rrT2 = 1.3               # T2 regressed
    r   = robjects.r         

    # Variable shared between R and Python
    robjects.globalenv['b1'] = b1
    robjects.globalenv['b2'] = b2
    robjects.globalenv['rT2'] = rT2
    
    robjects.globalenv['bb2'] = b2
    robjects.globalenv['rrT2'] = rT2
    
    #robjects.globalenv['sigma2'] = sigma2
    value = robjects.FloatVector(peaks)
    times = robjects.FloatVector(numpy.array(times))
    
    
    if (intercept):
        my_list = robjects.r('list(b1=.50, b2=1e2, rT2=0.03, bb2=1e1, rrT2=1.3)')
        my_lower = robjects.r('list(b1=0, b2=0, rT2=.005, bb2=0, rrT2=.005 )')
        my_upper = robjects.r('list(b1=2000, b2=2000, rT2=.700, bb2=2000, rrT2=1.3 )')
    else:
        my_list  = robjects.r('list(b2=.5, rT2=0.3,  bb2=.5, rrT2=1.3)')
        my_lower = robjects.r('list(b2=0,  rT2=.005, bb2=0,  rrT2=.005)')
        my_upper = robjects.r('list(b2=1,  rT2=2.6,    bb2=1,  rrT2=2.6)')

    my_cont = robjects.r('nls.control(maxiter=1000, warnOnly=TRUE, printEval=FALSE)')

    
    if (intercept):
        #fmla = robjects.RFormula('value ~ b1 + exp(-times/rT2)')
        fmla = robjects.Formula('value ~ b1 + b2*exp(-times/rT2) + bb2*exp(-times/rrT2)')
        #fmla = robjects.RFormula('value ~ b1 + b2*times + exp(-times/rT2)')
    else:
        fmla = robjects.Formula('value ~ b2*exp(-times/rT2) + bb2*exp(-times/rrT2)')

    env = fmla.getenvironment()
    env['value'] = value
    env['times'] = times
    
    # ugly, but I get errors with everything else I've tried
    Error = False
    #fit = robjects.r.nls(fmla,start=my_list,control=my_cont,weights=my_weights)
    if (sigma2[0] != None):
        #print("SIGMA 2")
        #fit = robjects.r.tryCatch(robjects.r.suppressWarnings(robjects.r.nls(fmla,start=my_list,control=my_cont,algorithm="port", \
        #                     weights=my_weights)), 'silent=TRUE')
        fit = robjects.r.tryCatch(robjects.r.nls(fmla,start=my_list,control=my_cont,algorithm='port',weights=my_weights,lower=my_lower,upper=my_upper))#, \
                            # weights=my_weights))
    else:
        try:
            fit = robjects.r.tryCatch(robjects.r.nls(fmla,start=my_list,control=my_cont,algorithm="port"))#,lower=my_lower,upper=my_upper))
        except:
            print("regression issue pass")
            Error = True
    # If failure fall back on zero regression values   
    if not Error:
        #Error = fit[3][0]
        report =  r.summary(fit)
    b1 = 0
    b2 = 0 
    rT2 = 1
    if (intercept):
        if not Error:
            b1  =  r['$'](report,'par')[0]
            b2  =  r['$'](report,'par')[1]
            rT2 =  r['$'](report,'par')[2]
            #print  report
            #print  r['$'](report,'convergence')
            #print  r['convergence'] #(report,'convergence')
            #print  r['$'](report,'par')[13]
            #print  r['$'](report,'par')[14]
        else:
            print("ERROR DETECTED, regressed values set to default")
            b1 = 1e1
            b2 = 1e-2
            rT2 = 1e-2
            #print r['$'](report,'par')[0]
            #print r['$'](report,'par')[1]
            #print r['$'](report,'par')[2]
        return [b1,b2,rT2, bb2, rrT2] 
    else:
        if not Error:
            rT2 =  r['$'](report,'par')[1]
            b2  =  r['$'](report,'par')[0]
            rrT2 =  r['$'](report,'par')[3]
            bb2  =  r['$'](report,'par')[2]
        else:
            print("ERROR DETECTED, regressed values set to default")
        return [b2, rT2, bb2, rrT2] 

def fun(x, t, y):
    """ Cost function for regression, single exponential, no DC term 
        x[0] = A0
        x[1] = zeta 
        x[2] = df
        x[3] = T2
    """
    # concatenated real and imaginary parts  
    pre =  np.concatenate((-x[0]*np.sin(2.*np.pi*x[2]*t + x[1])*np.exp(-t/x[3]), \
                           +x[0]*np.cos(2.*np.pi*x[2]*t + x[1])*np.exp(-t/x[3])))  
    return y-pre

def fun2(x, t, y):
    """ Cost function for regression, single exponential, no DC term 
        x[0] = A0
        x[1] = zeta 
        x[2] = T2
    """
    # concatenated real and imaginary parts  
    pre =  np.concatenate((x[0]*np.cos(x[1])*np.exp(-t/x[2]), \
                       -1.*x[0]*np.sin(x[1])*np.exp(-t/x[2])))  
    return y-pre


def quadratureDetect2(X, Y, tt, x0="None"): 
    """ Pure python quadrature detection using Scipy.  
        X = real part of NMR signal 
        Y = imaginary component of NMR signal 
        tt = time 
    """
    print("Pure Python Quad Det", "TODO look at loss functions and method")
    # Loss functions, linear, soft_l1, huber, cauchy, arctan 
    # df
    loss = 'cauchy'  #  'soft_l1'
    method = 'trf'   # trf, dogbox, lm 
    if x0=="None":
        x0 = np.array( [1., 0., 0., .2] ) # A0, zeta, df, T2 
        res_lsq = least_squares(fun, x0, args=(tt, np.concatenate((X, Y))), loss=loss, f_scale=1.0,\
            bounds=( [1., -np.pi, -5, .005] , [1000., np.pi, 5, .800] ),
            method=method 
            )
        x = res_lsq.x 
        print ("df", x[0], x[1], x[2], x[3])
    else:
        res_lsq = least_squares(fun, x0, args=(tt, np.concatenate((X, Y))), loss=loss, f_scale=1.0,\
            bounds=( [1., -np.pi, -5, .005] , [1000., np.pi, 5, .800] ),
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

def quadratureDetect(X, Y, tt, CorrectFreq=False, BiExp=False, CorrectDC=False):
 
    r   = robjects.r        

    if CorrectDC:
        robjects.r(''' 
             Xc1 <- function(E01, df, tt, phi, T2_1, DC) {
	                DC + E01*cos(2*pi*df*tt + phi) * exp(-tt/T2_1)
            }
    
            Yc1 <- function(E01, df, tt, phi, T2_1, DC) {
	                DC - E01*sin(2*pi*df*tt + phi) * exp(-tt/T2_1)
            } 
            ''')
    else:   
        robjects.r(''' 
             Xc1 <- function(E01, df, tt, phi, T2_1) {
	                E01*cos(2*pi*df*tt + phi) * exp(-tt/T2_1)
            }
    
            Yc1 <- function(E01, df, tt, phi, T2_1) {
	                -E01*sin(2*pi*df*tt + phi) * exp(-tt/T2_1)
            } 
            ''')

    # bi-exponential 
    if CorrectDC:
        robjects.r(''' 
             Xc2 <- function(E01, E02, df, tt, phi, T2_1, T2_2, DC) {
	               DC + E01*cos(2*pi*df*tt + phi) * exp(-tt/T2_1) + 
	                DC + E02*cos(2*pi*df*tt + phi) * exp(-tt/T2_2)
            }

            Yc2 <- function(E01, E02, df, tt, phi, T2_1, T2_2, DC) {
	                DC - E01*sin(2*pi*df*tt + phi) * exp(-tt/T2_1) + 
	                DC - E02*sin(2*pi*df*tt + phi) * exp(-tt/T2_2)
            } 
            ''')
    else:   
        robjects.r(''' 
             Xc2 <- function(E01, E02, df, tt, phi, T2_1, T2_2) {
	               E01*cos(2*pi*df*tt + phi) * exp(-tt/T2_1) + 
	               E02*cos(2*pi*df*tt + phi) * exp(-tt/T2_2)
            }

            Yc2 <- function(E01, E02, df, tt, phi, T2_1, T2_2) {
	                -E01*sin(2*pi*df*tt + phi) * exp(-tt/T2_1) + 
	                -E02*sin(2*pi*df*tt + phi) * exp(-tt/T2_2)
            } 
            ''')

    # Make 0 vector 
    Zero = robjects.FloatVector(numpy.zeros(len(X)))
    
    # Fitted Parameters
    E01 = 0.
    E02 = 0.
    df = 0.
    phi = 0.
    T2_1 = 0.
    T2_2 = 0.
    DC = 0.
    robjects.globalenv['DC'] = DC
    robjects.globalenv['E01'] = E01
    robjects.globalenv['E02'] = E02
    robjects.globalenv['df'] = df
    robjects.globalenv['phi'] = phi
    robjects.globalenv['T2_1'] = T2_1
    robjects.globalenv['T2_2'] = T2_2
    XY = robjects.FloatVector(numpy.concatenate((X,Y)))
    
    # Arrays
    tt = robjects.FloatVector(numpy.array(tt))
    X = robjects.FloatVector(numpy.array(X))
    Y = robjects.FloatVector(numpy.array(Y))
    Zero = robjects.FloatVector(numpy.array(Zero))

    

    if BiExp:
        if CorrectDC:
            fmla = robjects.Formula('XY ~ c(Xc2( E01, E02, df, tt, phi, T2_1, T2_2, DC ), Yc2( E01, E02, df, tt, phi, T2_1, T2_2, DC ))')
            if CorrectFreq:    
                start = robjects.r('list(E01=.100, E02=.01,   df=0,    phi=0.    ,  T2_1=.100, T2_2=.01, DC=0.0)')
                lower = robjects.r('list(E01=1e-6, E02=1e-6,  df=-50,  phi=-3.14 ,  T2_1=.001, T2_2=.001, DC=0.0)')
                upper = robjects.r('list(E01=1.00, E02=1.0,   df=50,   phi=3.14  ,  T2_1=.800, T2_2=.8, DC=0.5)')
            else:
                start = robjects.r('list(E01=.100, E02=.01,   phi=0.9   ,  T2_1=.100, T2_2=.01,  DC=0.0)')
                lower = robjects.r('list(E01=1e-6, E02=1e-6,  phi=-3.14 ,  T2_1=.001, T2_2=.001, DC=0.0)')
                upper = robjects.r('list(E01=1.00, E02=1.0,   phi=3.14  ,  T2_1=.800, T2_2=.8,   DC=0.5)')
        else:
            fmla = robjects.Formula('XY ~ c(Xc2( E01, E02, df, tt, phi, T2_1, T2_2 ), Yc2( E01, E02, df, tt, phi, T2_1, T2_2))')
            if CorrectFreq:    
                start = robjects.r('list(E01=.100, E02=.01,   df=0,    phi=0.    ,  T2_1=.100, T2_2=.01)')
                lower = robjects.r('list(E01=1e-6, E02=1e-6,  df=-50,  phi=-3.14 ,  T2_1=.001, T2_2=.001)')
                upper = robjects.r('list(E01=1.00, E02=1.0,   df=50,   phi=3.14  ,  T2_1=.800, T2_2=.8)')
            else:
                start = robjects.r('list(E01=.100, E02=.01,   phi=0.9   ,  T2_1=.100, T2_2=.01)')
                lower = robjects.r('list(E01=1e-6, E02=1e-6,  phi=-3.14 ,  T2_1=.001, T2_2=.001)')
                upper = robjects.r('list(E01=1.00, E02=1.0,   phi=3.14  ,  T2_1=.800, T2_2=.8)')
    else: 
        if CorrectDC:
            fmla = robjects.Formula('XY ~ c(Xc1( E01, df, tt, phi, T2_1, DC), Yc1( E01, df, tt, phi, T2_1,DC))')
            if CorrectFreq:    
                start = robjects.r('list(E01=.100, df=0   , phi=0.   , T2_1=.100, DC=0.0)')
                lower = robjects.r('list(E01=1e-6, df=-50., phi=-3.14, T2_1=.001, DC=0.0)')
                upper = robjects.r('list(E01=1.00, df=50. , phi=3.14 , T2_1=.800, DC=0.5)')
            else:
                start = robjects.r('list(E01=.100, phi= 0.  , T2_1=.100, DC=0.0)')
                lower = robjects.r('list(E01=1e-6, phi=-3.13, T2_1=.001, DC=0.0)')
                upper = robjects.r('list(E01=1.00, phi= 3.13, T2_1=.800, DC=0.5)')
        else:
            fmla = robjects.Formula('XY ~ c(Xc1( E01, df, tt, phi, T2_1), Yc1( E01, df, tt, phi, T2_1))')
            if CorrectFreq:    
                start = robjects.r('list(E01=.100, df=0     , phi=0.   ,  T2_1=.100)')
                lower = robjects.r('list(E01=1e-6, df=-50. , phi=-3.14 ,  T2_1=.001)')
                upper = robjects.r('list(E01=1.00, df=50.  , phi=3.14  ,  T2_1=.800)')
            else:
                start = robjects.r('list(E01=.100, phi= 0.  , T2_1=.100)')
                lower = robjects.r('list(E01=1e-6, phi=-3.13, T2_1=.001)')
                upper = robjects.r('list(E01=1.00, phi= 3.13, T2_1=.800)')

    env = fmla.getenvironment()
    env['Zero'] = Zero
    env['X'] = X
    env['Y'] = Y
    env['XY'] = XY 
    env['tt'] = tt

    cont = robjects.r('nls.control(maxiter=10000, warnOnly=TRUE, printEval=FALSE)')
    
    fit = robjects.r.tryCatch(robjects.r.nls(fmla, start=start, control=cont, lower=lower, upper=upper, algorithm='port')) #, \
    #fit = robjects.r.tryCatch(robjects.r.nls(fmla, start=start, control=cont)) #, \
    report =  r.summary(fit)

    conv = r['$'](fit,'convergence')[0]
    #if conv:
    #    print (report)
    #    print ("conv", conv)
    print ("Conv",  r['$'](fit,'convergence'))  # T2
    print (report)
    
    if BiExp:
        if CorrectFreq:    
            E0   =  r['$'](report,'par')[0]   # E01
            E0  +=  r['$'](report,'par')[1]   # E02
            df  =  r['$'](report,'par')[2]   # offset
            phi =  r['$'](report,'par')[3]   # phase 
            T2  =  r['$'](report,'par')[4]   # T2
        else:
            E0   =  r['$'](report,'par')[0]   # E01
            E0  +=  r['$'](report,'par')[1]   # E02
            phi =  r['$'](report,'par')[2]   # phase 
            T2  =  r['$'](report,'par')[3]   # T2
    else:
        if CorrectFreq:    
            E0   =  r['$'](report,'par')[0]   # E01
            df  =  r['$'](report,'par')[1]   # offset
            phi =  r['$'](report,'par')[2]   # phase 
            T2  =  r['$'](report,'par')[3]   # T2
        else:
            E0   =  r['$'](report,'par')[0]   # E01
            phi =  r['$'](report,'par')[1]   # phase 
            T2  =  r['$'](report,'par')[2]   # T2
    #phi = 0.907655876627
    #phi = 0
    #print ("df", df)# = 0
    return conv, E0,df,phi,T2
    

#################################################
# Regress for T2 using rpy2 interface
def regressSpec(w, wL, X): #,sigma2=1,intercept=True):

    # compute s
    s = -1j*w

    # TODO, if regression fails, it might be because there is no exponential
    # term, maybe do a second regression then on a linear model. 
    a   = 0                  # Linear 
    rT2 = 0.1                # T2 regressed
    r   = robjects.r         

    # Variable shared between R and Python
    robjects.globalenv['a'] = a
    robjects.globalenv['rT2'] = rT2
    robjects.globalenv['wL'] = wL
    robjects.globalenv['nb'] = 0

    s = robjects.ComplexVector(numpy.array(s))
    XX = robjects.ComplexVector(X)
    Xr = robjects.FloatVector(numpy.real(X))
    Xi = robjects.FloatVector(numpy.imag(X))
    Xa = robjects.FloatVector(numpy.abs(X))
    Xri = robjects.FloatVector(numpy.concatenate((Xr,Xi)))
    
    #my_lower = robjects.r('list(a=.001, rT2=.001, nb=.0001)')
    my_lower = robjects.r('list(a=.001, rT2=.001)')
    #my_upper = robjects.r('list(a=1.5, rT2=.300, nb =100.)')
    my_upper = robjects.r('list(a=1.5, rT2=.300)')
     
    #my_list = robjects.r('list(a=.2, rT2=0.03, nb=.1)')
    my_list = robjects.r('list(a=.2, rT2=0.03)')
    my_cont = robjects.r('nls.control(maxiter=5000, warnOnly=TRUE, printEval=FALSE)')
    
    #fmla = robjects.Formula('Xri ~ c(a*Re((wL) / (wL^2+(s+1/rT2)^2 )), a*Im((wL)/(wL^2 + (s+1/rT2)^2 )))') # envelope
    ##fmla = robjects.Formula('Xri ~ c(a*Re((wL) / (wL^2+(s+1/rT2)^2 )), a*Im((wL)/(wL^2 + (s+1/rT2)^2 )))') # envelope
    #fmla = robjects.Formula('XX ~ a*(wL) / (wL^2 + (s+1/rT2)^2 )') # complex
    #fmla = robjects.Formula('Xa ~ abs(a*(wL) / (wL^2 + (s+1/rT2)^2 )) + nb') # complex
    fmla = robjects.Formula('Xa ~ abs(a*(wL) / (wL^2 + (s+1/rT2)^2 ))') # complex
 
    env = fmla.getenvironment()
    env['s'] = s
    env['Xr'] = Xr
    env['Xa'] = Xa
    env['Xi'] = Xi
    env['Xri'] = Xri
    env['XX'] = XX
     
    #fit = robjects.r.tryCatch(robjects.r.nls(fmla,start=my_list, control=my_cont)) #, lower=my_lower, algorithm='port')) #, \
    fit = robjects.r.tryCatch(robjects.r.nls(fmla, start=my_list, control=my_cont, lower=my_lower, upper=my_upper, algorithm='port')) #, \
    report =  r.summary(fit)
    #print report 
    #print  r.warnings()
 
    a  =  r['$'](report,'par')[0]
    rT2 =  r['$'](report,'par')[1]
    nb =  r['$'](report,'par')[2]
    
    return a, rT2, nb

#################################################
# Regress for T2 using rpy2 interface
def regressSpecComplex(w, wL, X): #,sigma2=1,intercept=True):

    # compute s
    s = -1j*w

    # TODO, if regression fails, it might be because there is no exponential
    # term, maybe do a second regression then on a linear model. 
    a   = 1                  # Linear 
    rT2 = 0.1                # T2 regressed
    r   = robjects.r         
    phi2 = 0                 # phase
    wL2 = wL

    # Variable shared between R and Python
    robjects.globalenv['a'] = a
    robjects.globalenv['rT2'] = rT2
    robjects.globalenv['wL'] = wL
    robjects.globalenv['wL2'] = 0
    robjects.globalenv['nb'] = 0
    robjects.globalenv['phi2'] = phi2

    s = robjects.ComplexVector(numpy.array(s))
    XX = robjects.ComplexVector(X)
    Xr = robjects.FloatVector(numpy.real(X))
    Xi = robjects.FloatVector(numpy.imag(X))
    Xa = robjects.FloatVector(numpy.abs(X))
    Xri = robjects.FloatVector(numpy.concatenate((X.real,X.imag)))

    robjects.r(''' 
        source('kernel.r')
    ''')   
    #Kw = robjects.globalenv['Kwri']
     
    #print (numpy.shape(X))
    
    #my_lower = robjects.r('list(a=.001, rT2=.001, nb=.0001)')
    #my_lower = robjects.r('list(a=.001, rT2=.001)') # Working
    my_lower = robjects.r('list(a=.001, rT2=.001, phi2=-3.14, wL2=wL-5)')
    #my_upper = robjects.r('list(a=1.5, rT2=.300, nb =100.)')
    my_upper = robjects.r('list(a=3.5, rT2=.300, phi2=3.14, wL2=wL+5)')
     
    #my_list = robjects.r('list(a=.2, rT2=0.03, nb=.1)')
    my_list = robjects.r('list(a=.2, rT2=0.03, phi2=0, wL2=wL)')
    my_cont = robjects.r('nls.control(maxiter=5000, warnOnly=TRUE, printEval=FALSE)')
    
    #fmla = robjects.Formula('Xri ~ c(a*Re((wL) / (wL^2+(s+1/rT2)^2 )), a*Im((wL)/(wL^2 + (s+1/rT2)^2 )))') # envelope
    #fmla = robjects.Formula('Xi   ~   Im(a*(sin(phi2)*s + ((1/rT2)*sin(phi2)) + wL*cos(phi2)) / (wL^2+(s+1/rT2)^2 ))') # envelope
    #fmla = robjects.Formula('Xri ~ c(Re(a*(sin(phi2)*s + ((1/rT2)*sin(phi2)) + wL*cos(phi2)) / (wL^2+(s+1/rT2)^2 )), Im(a*(sin(phi2)*s + ((1/rT2)*sin(phi2)) + wL*cos(phi2)) / (wL^2+(s+1/rT2)^2 )))') # envelope
    
    #fmlar = robjects.Formula('Xr ~ (Kwr(a, phi2, s, rT2, wL)) ') # envelope
    #fmlai = robjects.Formula('Xi ~ (Kwi(a, phi2, s, rT2, wL)) ') # envelope
    fmla = robjects.Formula('Xri ~ c(Kwr(a, phi2, s, rT2, wL2), Kwi(a, phi2, s, rT2, wL2) ) ') # envelope
    #fmla = robjects.Formula('Xri ~ (Kwri(a, phi2, s, rT2, wL)) ') # envelope
    
    #fmla = robjects.Formula('Xa ~ (abs(a*(sin(phi2)*s + ((1/rT2)*sin(phi2)) + wL*cos(phi2)) / (wL^2+(s+1/rT2)^2 )))') # envelope
    #fmla = robjects.Formula('XX ~ a*(wL) / (wL^2 + (s+1/rT2)^2 )') # complex
    #fmla = robjects.Formula('Xa ~ abs(a*(wL) / (wL^2 + (s+1/rT2)^2 )) + nb') # complex
    
    #fmla = robjects.Formula('Xri ~ c(a*Re((wL) / (wL^2+(s+1/rT2)^2 )), a*Im((wL)/(wL^2 + (s+1/rT2)^2 )))') # envelope
    
    #        self.Gw[iw, iT2] = ((np.sin(phi2) *  (alpha + 1j*self.w[iw]) + self.wL*np.cos(phi2)) / \
    #                               (self.wL**2 + (alpha+1.j*self.w[iw])**2 ))
    #        self.Gw[iw, iT2] = ds * self.sc*((np.sin(phi2)*( alpha + 1j*self.w[iw]) + self.wL*np.cos(phi2)) / \
    #                               (self.wL**2 + (alpha+1.j*self.w[iw])**2 ))
    
    # Works Amplitude Only!
    #fmla = robjects.Formula('Xa ~ abs(a*(wL) / (wL^2 + (s+1/rT2)^2 ))') # complex
 
    env = fmla.getenvironment()
    env['s'] = s
    env['Xr'] = Xr
    env['Xa'] = Xa
    env['Xi'] = Xi
    env['Xri'] = Xri
    env['XX'] = XX
     
    fit = robjects.r.tryCatch(robjects.r.nls(fmla,start=my_list, control=my_cont)) #, lower=my_lower, algorithm='port')) #, \
    #fitr = robjects.r.tryCatch(robjects.r.nls(fmlar, start=my_list, control=my_cont, lower=my_lower, upper=my_upper, algorithm='port')) #, \
    
    #env = fmlai.getenvironment()
    #fiti = robjects.r.tryCatch(robjects.r.nls(fmlai, start=my_list, control=my_cont, lower=my_lower, upper=my_upper, algorithm='port')) #, \
    
    #reportr =  r.summary(fitr)
    #reporti =  r.summary(fiti)
    report =  r.summary(fit)
    #print( report )
    #exit()
    #print( reportr )
    #print( reporti  )
    #exit()
    #print  r.warnings()
 
    #a   =  (r['$'](reportr,'par')[0] + r['$'](reporti,'par')[0]) / 2.
    #rT2 =  (r['$'](reportr,'par')[1] + r['$'](reporti,'par')[1]) / 2.
    #nb  =  (r['$'](reportr,'par')[2] + r['$'](reporti,'par')[2]) / 2.
    a   =  r['$'](report,'par')[0] 
    rT2 =  r['$'](report,'par')[1] 
    nb  =  r['$'](report,'par')[2] #phi2 

    print ("Python wL2", r['$'](report,'par')[3] )   
    print ("Python zeta", r['$'](report,'par')[2] )   
 
    return a, rT2, nb



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
