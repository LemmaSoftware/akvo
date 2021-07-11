import numpy as np
import scipy.optimize as so 

def phid(Wd, K, m, d_obs):
    """
        Wd = data weighting matrix 
        K = complex valued forward model kernel 
        m = model 
        d_obs = observed data 
    """
    #print("phid=", np.linalg.norm(np.dot(Wd, np.abs(np.dot(K,m)) - d_obs))**2 / len(d_obs) )
    return np.linalg.norm(np.dot(Wd, np.abs(np.dot(K,m)) - d_obs))**2

def phim(Wm, m):
    """
        Wm = model weighting matrix 
        x = model 
    """
    return np.linalg.norm(np.dot(Wm, m))**2

def PHI(m, Wd, K, d_obs, Wm, alphastar):
    """
        Global objective function 
        x = model to be fit 
        Wd = data weighting matrix 
        K = complex forward modelling kernel 
        d_obs = observed data (modulus)
        Wm = model weighting matrix 
        alphastar = regularisation to use
    """
    return phid(Wd, K, m, d_obs) + alphastar*phim(Wm, m)

# main 
def nonlinearinversion( x0, Wd, K, d_obs, Wm, alphastar):
    print("Performing non-linear inversion")
    args = (Wd, K, d_obs, Wm, alphastar)
    #return so.minimize(PHI, np.zeros(len(x0)), args, 'Nelder-Mead')
    bounds = np.zeros((len(x0),2))
    bounds[:,0] = x0*0.75
    bounds[:,1] = x0*1.25
    #bounds[:,0] = 0
    #bounds[:,1] = np.max(x0)*1.25
    return so.minimize(PHI, x0, args, 'L-BFGS-B', bounds=bounds)     # Works well 
    #return so.minimize(PHI, x0, args, 'Powell', bounds=bounds)       # Slow but works 
    #return so.minimize(PHI, x0, args, 'trust-constr', bounds=bounds) # very Slow 
    #return so.minimize(PHI, x0, args, 'TNC', bounds=bounds)          # slow 
    #return so.minimize(PHI, x0, args, 'SLSQP', bounds=bounds)        # slow 
