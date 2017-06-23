import sys,os
sys.path.append(os.path.dirname(__file__))

import scipy.sparse

import numpy as np

from sbm_vem import SBM_VEM
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten


def clipping(x,minval=1.0e-40):
    x[x<minval]=minval
    x[x>1-minval]=1-minval
    return x


class PLSBM_VEM(SBM_VEM):
    def __init__(self,A,y,eps,verbose=False):
        self.A = A
        self.y = y
        self.eps = eps
        self.verbose = verbose
        self.labeled = self.y>-1
        self.k = self.y.max()+1

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
        z[self.labeled] = 0
        z[self.labeled,self.y[self.labeled]] = 1
        return clipping(z)

    def _do_mstep(self):
        self.params['r'] = self._update_r()
        self.params['pi'] = self._update_pi()
        self.params['alpha'] = self._update_alpha()

    def _calc_self_term(self,i):
        ret = np.zeros(self.k)
        if self.labeled[i]:
            beta = (1-self.params['alpha']) / (self.k - 1)
            ret[:] = np.log(beta)
            ret[self.y[i]] = np.log(self.params['alpha'])
        return ret

    def _initialize_params(self):
        params = {}
        params['r'] = np.zeros(self.k)
        params['pi'] = np.zeros((self.k,self.k))
        params['alpha'] = 0
        return params

    def _update_alpha(self,minval=0.001,maxval=0.999):
        alpha = self.z[self.labeled,self.y[self.labeled]].sum() / float(self.labeled.sum())
        if alpha > maxval:
            alpha = maxval
        elif alpha < minval:
            alpha = minval
        return alpha

    def _print_intermediate_results(self,niter):
        print("-------------------------------------------")
        print("# Iter: %s" % niter)
        print("Gamma = \n%s" % self.params['r'])
        print("Pi = \n%s" % self.params['pi'])
        print("Alpha = %s" % self.params['alpha'])
        print(self.z.sum(axis=0))
