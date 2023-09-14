# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:02:38 2023

@author: kathryn.newhart
"""

import numpy as np
from scipy import signal
from sklearn.utils import check_X_y
from sklearn.preprocessing import StandardScaler
from multiprocessing import cpu_count
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
import bottleneck as bn
#from . import mi

"""
Methods for calculating Mutual Information in an embarrassingly parallel way.

Author: Daniel Homola <dani.homola@gmail.com>
License: BSD 3 clause
"""

import numpy as np
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

def get_mi_vector(MI_FS, F, s):
    """
    Calculates the Mututal Information between each feature in F and s.

    This function is for when |S| > 1. s is the previously selected feature.
    We exploite the fact that this step is embarrassingly parallel.
    """

    MIs = Parallel(n_jobs=MI_FS.n_jobs)(delayed(_get_mi)(f, s, MI_FS)
                                        for f in F)
    return MIs


def _get_mi(f, s, MI_FS):
    n, p = MI_FS.X.shape
    if MI_FS.method in ['JMI', 'JMIM']:
        # JMI & JMIM
        joint = MI_FS.X[:, (s, f)]
        if MI_FS.categorical:
            MI = _mi_dc(joint, MI_FS.y, MI_FS.k)
        else:
            vars = (joint, MI_FS.y)
            MI = _mi_cc(vars, MI_FS.k)
    else:
        # MRMR
        vars = (MI_FS.X[:, s].reshape(n, 1), MI_FS.X[:, f].reshape(n, 1))
        MI = _mi_cc(vars, MI_FS.k)

    # MI must be non-negative
    if MI > 0:
        return MI
    else:
        return np.nan


def get_first_mi_vector(MI_FS, k):
    """
    Calculates the Mututal Information between each feature in X and y.

    This function is for when |S| = 0. We select the first feautre in S.
    """
    n, p = MI_FS.X.shape
    MIs = Parallel(n_jobs=MI_FS.n_jobs)(delayed(_get_first_mi)(i, k, MI_FS)
                                        for i in range(p))
    return MIs


def _get_first_mi(i, k, MI_FS):
    n, p = MI_FS.X.shape
    if MI_FS.categorical:
        x = MI_FS.X[:, i].reshape((n, 1))
        MI = _mi_dc(x, MI_FS.y, k)
    else:
        vars = (MI_FS.X[:, i].reshape((n, 1)), MI_FS.y)
        MI = _mi_cc(vars, k)

    # MI must be non-negative
    if MI > 0:
        return MI
    else:
        return np.nan


def _mi_dc(x, y, k):
    """
    Calculates the mututal information between a continuous vector x and a
    disrete class vector y.

    This implementation can calculate the MI between the joint distribution of
    one or more continuous variables (X[:, 1:3]) with a discrete variable (y).

    Thanks to Adam Pocock, the author of the FEAST package for the idea.

    Brian C. Ross, 2014, PLOS ONE
    Mutual Information between Discrete and Continuous Data Sets
    """

    y = y.flatten()
    n = x.shape[0]
    classes = np.unique(y)
    knn = NearestNeighbors(n_neighbors=k)
    # distance to kth in-class neighbour
    d2k = np.empty(n)
    # number of points within each point's class
    Nx = []
    for yi in y:
        Nx.append(np.sum(y == yi))

    # find the distance of the kth in-class point
    for c in classes:
        mask = np.where(y == c)[0]
        knn.fit(x[mask, :])
        d2k[mask] = knn.kneighbors()[0][:, -1]

    # find the number of points within the distance of the kth in-class point
    knn.fit(x)
    m = knn.radius_neighbors(radius=d2k, return_distance=False)
    m = [i.shape[0] for i in m]

    # calculate MI based on Equation 2 in Ross 2014
    MI = psi(n) - np.mean(psi(Nx)) + psi(k) - np.mean(psi(m))
    return MI


def _mi_cc(variables, k=1):
    """
    Returns the mutual information between any number of variables.

    Here it is used to estimate MI between continuous X(s) and y.
    Written by Gael Varoquaux:
    https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
    """

    all_vars = np.hstack(variables)
    return (sum([_entropy(X, k=k) for X in variables]) -
            _entropy(all_vars, k=k))


def _nearest_distances(X, k=1):
    """
    Returns the distance to the kth nearest neighbor for every point in X
    """

    knn = NearestNeighbors(n_neighbors=k, metric='chebyshev')
    knn.fit(X)
    # the first nearest neighbor is itself
    d, _ = knn.kneighbors(X)
    # returns the distance to the kth nearest neighbor
    return d[:, -1]


def _entropy(X, k=1):
    """
    Returns the entropy of the X.

    Written by Gael Varoquaux:
    https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation

    References
    ----------
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:

    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.

    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)

    """

    # Distance to kth nearest neighbor
    r = _nearest_distances(X, k)
    n, d = X.shape
    volume_unit_ball = (np.pi ** (.5 * d)) / gamma(.5 * d + 1)
    return (d * np.mean(np.log(r + np.finfo(X.dtype).eps)) +
            np.log(volume_unit_ball) + psi(n) - psi(k))


"""
Parallelized Mutual Information based Feature Selection module.

Author: Daniel Homola <dani.homola@gmail.com>
License: BSD 3 clause
"""


NUM_CORES = cpu_count()


class MutualInformationFeatureSelector(BaseEstimator, SelectorMixin):
    """
    MI_FS stands for Mutual Information based Feature Selection.
    This class contains routines for selecting features using both
    continuous and discrete y variables. Three selection algorithms are
    implemented: JMI, JMIM and MRMR.

    This implementation tries to mimic the scikit-learn interface, so use fit,
    transform or fit_transform, to run the feature selection.

    Parameters
    ----------

    method : string, default = 'JMI'
        Which mutual information based feature selection method to use:
        - 'JMI' : Joint Mutual Information [1]
        - 'JMIM' : Joint Mutual Information Maximisation [2]
        - 'MRMR' : Max-Relevance Min-Redundancy [3]

    k : int, default = 5
        Sets the number of samples to use for the kernel density estimation
        with the kNN method. Kraskov et al. recommend a small integer between
        3 and 10.

    n_features : int or string, default = 'auto'
        If int, it sets the number of features that has to be selected from X.
        If 'auto' this is determined automatically based on the amount of
        mutual information the previously selected features share with y.

    categorical : Boolean, default = True
        If True, y is assumed to be a categorical class label. If False, y is
        treated as a continuous. Consequently this parameter determines the
        method of estimation of the MI between the predictors in X and y.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    verbose : int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays selected features
        - 2: displays selected features and mutual information

    Attributes
    ----------

    n_features_ : int
        The number of selected features.

    support_ : array of length X.shape[1]
        The mask array of selected features.

    ranking_ : array of shape n_features
        The feature ranking of the selected features, with the first being
        the first feature selected with largest marginal MI with y, followed by
        the others with decreasing MI.

    mi_ : array of shape n_features
        The JMIM of the selected features. Usually this a monotone decreasing
        array of numbers converging to 0. One can use this to estimate the
        number of features to select. In fact this is what n_features='auto''
        tries to do heuristically.

    Examples
    --------

    import pandas as pd
    import mifs

    # load X and y
    X = pd.read_csv('my_X_table.csv', index_col=0).values
    y = pd.read_csv('my_y_vector.csv', index_col=0).values

    # define MI_FS feature selection method
    feat_selector = mifs.MutualInformationFeatureSelector()

    # find all relevant features
    feat_selector.fit(X, y)

    # check selected features
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

    References
    ----------

    [1] H. Yang and J. Moody, "Data Visualization and Feature Selection: New
        Algorithms for Nongaussian Data"
        NIPS 1999
    [2] Bennasar M., Hicks Y., Setchi R., "Feature selection using Joint Mutual
        Information Maximisation"
        Expert Systems with Applications, Vol. 42, Issue 22, Dec 2015
    [3] H. Peng, Fulmi Long, C. Ding, "Feature selection based on mutual
        information criteria of max-dependency, max-relevance,
        and min-redundancy"
        Pattern Analysis & Machine Intelligence 2005
    """

    def __init__(self, method='JMI', k=5, n_features='auto', categorical=True,
                 n_jobs=1, verbose=0):
        self.method = method
        self.k = k
        self.n_features = n_features
        self.categorical = categorical
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._support_mask = None

    def _get_support_mask(self):
        if self._support_mask is None:
            raise ValueError('mRMR has not been fitted yet!')
        return self._support_mask

    def fit(self, X, y):
        """
        Fits the MI_FS feature selection with the chosen MI_FS method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """

        # Check if n_jobs is negative
        if self.n_jobs < 0:
            self.n_jobs = NUM_CORES - self.n_jobs

        self.X, y = self._check_params(X, y)
        n, p = X.shape
        self.y = y.reshape((n, 1))

        # list of selected features
        S = []
        # list of all features
        F = list(range(p))

        if self.n_features != 'auto':
            feature_mi_matrix = np.zeros((self.n_features, p))
        else:
            feature_mi_matrix = np.zeros((n, p))
        feature_mi_matrix[:] = np.nan
        S_mi = []

        # ---------------------------------------------------------------------
        # FIND FIRST FEATURE
        # ---------------------------------------------------------------------

        xy_MI = np.array(mi.get_first_mi_vector(self, self.k))

        # choose the best, add it to S, remove it from F
        S, F = self._add_remove(S, F, bn.nanargmax(xy_MI))
        S_mi.append(bn.nanmax(xy_MI))

        # notify user
        if self.verbose > 0:
            self._print_results(S, S_mi)

        # ---------------------------------------------------------------------
        # FIND SUBSEQUENT FEATURES
        # ---------------------------------------------------------------------
        if self.n_features == 'auto': n_features = np.inf
        else: n_features = self.n_features
         
        while len(S) < n_features:
            # loop through the remaining unselected features and calculate MI
            s = len(S) - 1
            feature_mi_matrix[s, F] = mi.get_mi_vector(self, F, S[-1])

            # make decision based on the chosen FS algorithm
            fmm = feature_mi_matrix[:len(S), F]
            if self.method == 'JMI':
                selected = F[bn.nanargmax(bn.nansum(fmm, axis=0))]
            elif self.method == 'JMIM':
                if bn.allnan(bn.nanmin(fmm, axis = 0)):
                    break
                selected = F[bn.nanargmax(bn.nanmin(fmm, axis=0))]
            elif self.method == 'MRMR':
                if bn.allnan(bn.nanmean(fmm, axis = 0)):
                    break
                MRMR = xy_MI[F] - bn.nanmean(fmm, axis=0)
                selected = F[bn.nanargmax(MRMR)]
                S_mi.append(bn.nanmax(MRMR))

            # record the JMIM of the newly selected feature and add it to S
            if self.method != 'MRMR':
                S_mi.append(bn.nanmax(bn.nanmin(fmm, axis=0)))
            S, F = self._add_remove(S, F, selected)

            # notify user
            if self.verbose > 0:
                self._print_results(S, S_mi)

            # if n_features == 'auto', let's check the S_mi to stop
            if self.n_features == 'auto' and len(S) > 10:
                # smooth the 1st derivative of the MI values of previously sel
                MI_dd = signal.savgol_filter(S_mi[1:], 9, 2, 1)
                # does the mean of the last 5 converge to 0?
                if np.abs(np.mean(MI_dd[-5:])) < 1e-3:
                    break

        # ---------------------------------------------------------------------
        # SAVE RESULTS
        # ---------------------------------------------------------------------

        self.n_features_ = len(S)
        self._support_mask = np.zeros(p, dtype=np.bool)
        self._support_mask[S] = True
        self.ranking_ = S
        self.mi_ = S_mi

        return self

    def _isinteger(self, x):
        return np.all(np.equal(np.mod(x, 1), 0))

    def _check_params(self, X, y):
        # checking input data and scaling it if y is continuous
        X, y = check_X_y(X, y)

        if not self.categorical:
            ss = StandardScaler()
            X = ss.fit_transform(X)
            y = ss.fit_transform(y.reshape(-1, 1))

        # sanity checks
        methods = ['JMI', 'JMIM', 'MRMR']
        if self.method not in methods:
            raise ValueError('Please choose one of the following methods:\n' +
                             '\n'.join(methods))

        if not isinstance(self.k, int):
            raise ValueError("k must be an integer.")
        if self.k < 1:
            raise ValueError('k must be larger than 0.')
        if self.categorical and np.any(self.k > np.bincount(y)):
            raise ValueError('k must be smaller than your smallest class.')

        if not isinstance(self.categorical, bool):
            raise ValueError('Categorical must be Boolean.')
        if self.categorical and np.unique(y).shape[0] > 5:
            print ('Are you sure y is categorical? It has more than 5 levels.')
        if not self.categorical and self._isinteger(y):
            print ('Are you sure y is continuous? It seems to be discrete.')
        if self._isinteger(X):
            print ('The values of X seem to be discrete. MI_FS will treat them'
                   'as continuous.')
        return X, y

    def _add_remove(self, S, F, i):
        """
        Helper function: removes ith element from F and adds it to S.
        """

        S.append(i)
        F.remove(i)
        return S, F

    def _print_results(self, S, MIs):
        out = ''
        if self.n_features == 'auto':
            out += 'Auto selected feature #' + str(len(S)) + ' : ' + str(S[-1])
        else:
            out += ('Selected feature #' + str(len(S)) + ' / ' +
                    str(self.n_features) + ' : ' + str(S[-1]))

        if self.verbose > 1:
            out += ', ' + self.method + ' : ' + str(MIs[-1])
        print (out)
        
        
# Use
      
#import pandas as pd

# define MI_FS feature selection method
#feat_selector = mifs.MutualInformationFeatureSelector()

# find all relevant features
#feat_selector.fit(X, y)

# check selected features
#feat_selector._support_mask

# check ranking of features
#feat_selector.ranking_

# call transform() on X to filter it down to selected features
#X_filtered = feat_selector.transform(X)