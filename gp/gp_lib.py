import numpy as np
from scipy.optimize import minimize
import time

"""
Python Gaussian Process module.
This code is supposed to accompany the tutorial 'An Introduction
to Gaussian Processes'

P.L.Green
University of Liverpool
22/05/19
"""


def Kernel(squared_distance, L):
    '''  Kernel function (squared exponential, with length scale L)

    '''

    return np.exp(-1/(2*L**2)*squared_distance)


def FindGramMatrix(X, L, Sigma, N, squared_distances):
    ''' Function that creates and inverts gram matrix for a squared
        exponential kernel with length-scale L.

    '''

    # squared_distances = FindSquaredDistances(X)
    K = Kernel(squared_distances, L)
    C = K + Sigma**2*np.identity(N)
    InvC = np.linalg.inv(C)
    return K, C, InvC


def FindSquaredDistances(X):
    ''' Function that finds the squared distances between
        inputs points (efficiently).

    '''

    if np.size(X[0]) == 1:
        Xsq = X**2
        temp = (Xsq[:, None] + Xsq[None, :])
        squared_distances = -2*np.outer(X, X) + temp
    else:
        squared_distances = np.empty((X.shape[0], X.shape[0]))
        # for ii in range(X.shape[0]):
        #     for jj in range(X.shape[0]):
        #         squared_distances[ii, jj] = sqeuclidean(X[ii, :], X[jj, :])
        #         print(ii, jj)
        # Xsq = np.sum(X**2, 1)
        for ii in range(X.shape[0]):
            print(ii)
            squared_distances[ii, :] = np.sum((X[ii, :] - X) ** 2, 1)
            # squared_distances[ii, :] = -2. * np.dot(X[ii, :], X.T) + (Xsq[ii] + Xsq)

        # squared_distances = -2.*np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    return squared_distances


def NegLogLikelihoodFun(Theta, a):
    ''' Returns negative log-likelihood function in a form
        suitable for scipy.optimize.fmin_bfgs.

    '''

    X = a[0]
    Y = a[1]
    N = a[2]
    squared_distances = a[3]
    L = Theta[0]
    Sigma = Theta[1]
    K, C, InvC = FindGramMatrix(X, L, Sigma, N, squared_distances)
    (Sign, LogDetC) = np.linalg.slogdet(C)
    LogDetC = Sign*LogDetC
    return 0.5*LogDetC + 0.5*np.dot(Y, np.dot(InvC, Y))


def dC_dL_Fun(X, L, K, N, squared_distances):
    ''' Derivative of matrix C w.r.t L (length scale).

    '''

    # squared_distances = FindSquaredDistances(X)
    return np.power(L, -3) * np.multiply(squared_distances, K)


def dC_dSigma_Fun(Sigma, K, N):
    ''' Derivative of matrix C w.r.t sigma (noise std).

    '''
    return 2*Sigma*np.eye(N)


def dNLL_dTheta(Theta, a):
    ''' Derivative of the negative log-likelihood w.r.t hyperparameters in a
        form suitable for scipy.optimize.fmin_bfgs.

    '''

    X = a[0]
    Y = a[1]
    N = a[2]
    squared_distances = a[3]
    L = Theta[0]
    Sigma = Theta[1]
    K, C, InvC = FindGramMatrix(X, L, Sigma, N, squared_distances)

    # Find dC / dL
    dC_dL = dC_dL_Fun(X, L, K, N, squared_distances)

    # Find dC / dSigma
    dC_dSigma = dC_dSigma_Fun(Sigma, K, N)

    # Find dlogp / dL
    dLogL_dL = (0.5 * np.trace(np.dot(InvC, dC_dL)) -
                0.5 * np.dot(Y, np.dot(InvC, np.dot(dC_dL, np.dot(InvC,
                                                                  Y)))))

    # Find dlogp / dSigma
    dLogL_dSigma = (0.5*np.trace(np.dot(InvC, dC_dSigma)) -
                    0.5*np.dot(Y, np.dot(InvC, np.dot(dC_dSigma, np.dot(InvC,
                                                                        Y)))))

    # Gradient vector
    gradient = np.array([dLogL_dL, dLogL_dSigma])

    return gradient


def Train(L0, Sigma0, X, Y, N):
    ''' Update hyperparameters using scipy.minimize

    '''

    Theta0 = [L0, Sigma0]
    squared_distances = FindSquaredDistances(X)
    a = (X, Y, N, squared_distances)
    b1 = (1e-6, 4)
    b2 = (1e-6, 1)
    bnds = (b1, b2)
    start_time = time.time()

    sol = minimize(NegLogLikelihoodFun, x0=Theta0, args=(a, ),
                   method='SLSQP', jac=dNLL_dTheta, bounds=bnds)

    elapsed_time = time.time() - start_time
    ThetaOpt = sol.x
    L = ThetaOpt[0]
    Sigma = ThetaOpt[1]
    K, C, InvC = FindGramMatrix(X, L, Sigma, N, squared_distances)
    return L, Sigma, K, C, InvC, elapsed_time


def callbackF(Theta):
    ''' Callback function for the fmin_bfgs optimisation routine

    '''

    print(Theta[0], Theta[1])


def Predict(X, xStar, L, InvCxY, InvC, c):
    ''' GP prediction

    '''
    diff = X - xStar
    squared_distances = np.einsum('ij,ij->j', diff, diff)
    k = Kernel(squared_distances, L)

    yStarMean = np.einsum('i,i->', k, InvCxY)
    yStarStd = np.sqrt(c - np.dot(k, np.dot(InvC, k)))

    return yStarMean, yStarStd
