##########################################################
#
#
#
#
#
#from scipy import mean
import numpy

def pca(A, remove=[]):
    ''' The input matrix A should be a 2D numpy array in column-major 
        order. Each column is a dataset and PCA will be applied across
        columns
    '''
    #nrow = len(A[:,0])
    #ncol = len(A[0,:])
    nrow,ncol = numpy.shape(A)

    # Cast into matrix type for easier math
    A = numpy.matrix(A)

    # Allocate Covariance Matrix
    covMatrix = numpy.matrix(numpy.zeros((nrow, nrow)))
    
    # Compute Means of each row. Each column must be normalised
    meanArray = []
    for i in range(nrow):
        meanArray.append(numpy.mean(A[i].tolist()[0]) ) 
        A[i] -= meanArray[i]
    meanArray = numpy.array(meanArray) 
    
    # Generate Covariance Matrix
    covMatrix = numpy.cov(A)

    # Compute Eigen Values, Eigen Vectors     
    eigs = numpy.linalg.eig(covMatrix)
    K = eigs[1].T
    
    #print K
    #print "Eigen Values"
    #print eigs[0]
    
    # Zero requested components
    for i in remove:
        K[i] = numpy.zeros(len(K))        

    # Make Transform Matrices
    transMatrix = K*A
    
    # Return Necssary Stuff
    return numpy.array(transMatrix), K, meanArray
    #return K, A, meanArray 

#def invpca(K, A, means):
def invpca(transMatrix, K, means):
    '''Converts a PCA rotated dataset back to normal. Input parameters
    are the components to discard in re-creation.
    '''
    K = numpy.matrix(K)
    # Transform and Untransform the data
    #transMatrix = K*A
    untransMatrix = K.T*transMatrix
    
    # Correct for normalisation
    for i in range(len(transMatrix)):
        untransMatrix[i] += means[i]

    return untransMatrix    
