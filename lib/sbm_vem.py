import scipy.sparse
import scipy.linalg

import numpy as np

from scipy.misc import logsumexp
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans

"""
SBM
Variational EM with Mean field approx.
"""

def clipping(x,minval=1.0e-40):
    x[x<minval]=minval
    x[x>1-minval]=1-minval
    return x

class SBM_VEM:
    def __init__(self,A,k,eps,verbose=False):
        self.A = A
        self.k = k
        self.verbose = verbose
        self.eps = eps

    def fit(self):
        self.n_ = self.A.shape[0]
        self.m_ = self.A.sum()

        self.z = self._initialize_z()
        self.params = self._initialize_params()
        niter = 1
        while True:
            """ M-step """
            self._do_mstep()
            """ E-step """
            old_z = self.z
            self._do_estep()
            if self.verbose: self._print_intermediate_results(niter)
            if self._err(self.z,old_z) < self.eps: break
            niter += 1
        return self

    def predict(self):
        return self.predict_proba().argmax(axis=1)

    def predict_proba(self):
        return self.z

    def _print_intermediate_results(self,niter):
        print("-------------------------------------------")
        print("# Iter: %s" % niter)
        print("Gamma = \n%s" % self.params['r'])
        print("Pi = \n%s" % self.params['pi'])

    def _do_mstep(self):
        self.params['r'] = self._update_r()
        self.params['pi'] = self._update_pi()

    def _do_estep(self):
        self.z = self._update_z()

    def _initialize_params(self):
        params = {}
        params['r'] = np.zeros(self.k)
        params['pi'] = np.zeros((self.k,self.k))
        return params

    def _initialize_z(self):
        """ Spectral Clustering """
        A = self.A.dot(self.A)
        D = scipy.sparse.diags(1.0/np.sqrt(np.maximum(1,A.sum(axis=0).A[0])))
        Laplacian = scipy.sparse.identity(A.shape[0]) - D.dot(A.dot(D))
        _,U = scipy.sparse.linalg.eigsh(Laplacian,k=self.k,which='SA')
        U = whiten(U)
        m = KMeans(n_clusters=self.k)
        res = m.fit_predict(U)
        z = np.zeros((self.n_,self.k))
        z[np.arange(z.shape[0]),res] = 1
        return clipping(z)

    def _update_z(self):
        z = self.z.copy()
        niter = 0
        logr = np.log(self.params['r'])
        logonenpi = np.log(1-self.params['pi'])
        logpidonenpi = np.log(self.params['pi']) - logonenpi
        logdiagpi = np.log(np.diag(self.params['pi']))
        logdiagonenpi = np.log(np.diag(1-self.params['pi']))
        while True:
            old_z = z.copy()
            zk = z.sum(axis=0)
            global_term = 2*logonenpi.dot(zk)
            for i in np.random.permutation(self.n_):
                neighbor_z = self.A[i].dot(z)[0]
                neighbor_term = 2*logpidonenpi.dot(neighbor_z)
                global_term -= 2*logonenpi.dot(z[i])
                self_term = self._calc_self_term(i)
                self_term += self.A[i,i]*logdiagpi + (1-self.A[i,i])*logdiagonenpi
                z[i] = logr + global_term + neighbor_term + self_term
                z[i] = clipping(np.exp(z[i] - logsumexp(z[i])))
                global_term += 2*logonenpi.dot(z[i])
            err = self._err(z,old_z)
            if err < self.eps: break
            niter += 1
        return z

    def _calc_self_term(self,i):
        return 0

    def _update_r(self):
        return clipping(self.z.sum(axis=0) / self.n_)

    def _update_pi(self):
        nk = self.z.sum(axis=0)
        den = np.outer(nk,nk) - self.z.T.dot(self.z) + np.diag(nk)
        num = self.z.T.dot(self.A.dot(self.z))
        return clipping(num/den)

    def _err(self,a,b):
        return abs(a-b).sum() / self.n_
