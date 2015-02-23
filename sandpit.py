__author__ = 'James Robert Lloyd'
__description__ = 'Scraps of code before module structure becomes apparent'

from util import callback_1d

import pybo
from pybo.functions.functions import _cleanup, GOModel

import numpy as np

from sklearn.datasets import load_iris
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

import os
import sys
sys.path.append(os.path.dirname(__file__))

@_cleanup
class Sinusoidal(GOModel):
    """
    Simple sinusoidal function bounded in [0, 2pi] given by cos(x)+sin(3x).
    """
    bounds = [[0, 2*np.pi]]
    xmax = 3.61439678

    @staticmethod
    def _f(x):
        return -np.ravel(np.cos(x) + np.sin(3*x))

@_cleanup
class CV_RF(GOModel):
    """
    Cross validated random forest
    """
    bounds = [[1, 25]]
    xmax = 10  # FIXME - should this not be optional?

    @staticmethod
    def _f(x):
        # iris = load_iris()
        X, y = X, y = make_hastie_10_2(random_state=0)
        x = np.ravel(x)
        f = np.zeros(x.shape)
        for i in range(f.size):
            clf = RandomForestClassifier(n_estimators=1, min_samples_leaf=int(np.round(x[i])), random_state=0)
            # scores = cross_val_score(clf, iris.data, iris.target)
            scores = cross_val_score(clf, X, y, cv=5)
            f[i] = -scores.mean()
        return f.ravel()

if __name__ == '__main__':
    objective = CV_RF()

    info = pybo.solve_bayesopt(
        objective,
        objective.bounds,
        niter=25,
        noisefree=False,
        rng=0,
        init='uniform',
        callback=callback_1d)

    print('Finished')

    raw_input('Press enter to finish')
