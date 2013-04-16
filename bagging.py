from collections import Counter
import random

import numpy as np
from sklearn.ensemble.base import BaseEnsemble
from sklearn.ensemble.forest import ForestClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def get_asym_task(X, y):
    # assume only 2 classes
    large_class, small_class = [pair[0] for pair in Counter(y).most_common()]

    X_small = np.array([X[i] for (i, cls) in enumerate(y)
                        if cls == small_class])
    X_large = np.array([X[i] for (i, cls) in enumerate(y)
                        if cls == large_class])

    y_new = np.array(([small_class] * len(X_small)) +
                     ([large_class] * len(X_small)))

    return X_small, X_large, y_new


class BalanceForcedDecisionTreeClassifier(DecisionTreeClassifier):
    def fit(self, X, y,
            sample_mask=None, X_argsorted=None,
            check_input=True, sample_weight=None):

        if X_argsorted is not None:
            print 'unsupported: X_argsorted is not None'

        X_small, X_large, y_new = get_asym_task(X, y)

        large_subset = np.array(random.sample(X_large, len(X_small)))
        X_new = np.vstack((X_small, large_subset))

        # shuffle data
        zipped = zip(X_new, y_new)
        random.shuffle(zipped)
        X_new, y_new = zip(*zipped)

        return super(BalanceForcedDecisionTreeClassifier, self).fit(
            X_new, y_new,
            sample_mask, X_argsorted,
            check_input, sample_weight)


class BalanceForcedRandomForestClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_density=0.1,
                 max_features="auto",
                 bootstrap=True,
                 compute_importances=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(BalanceForcedRandomForestClassifier, self).__init__(
            base_estimator=BalanceForcedDecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_density",
                              "max_features", "random_state"),
            bootstrap=bootstrap,
            compute_importances=compute_importances,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_density = min_density
        self.max_features = max_features


class AsymBaggingRFCs(BaseEnsemble):
    """Addresses class imbalance by training an ensemble of RFCs
    on all of the small class, and a random subset of the large class."""
    def __init__(self,
                 asym_estimators=5,

                 # for RFC
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_density=0.1,
                 max_features="auto",
                 bootstrap=True,
                 compute_importances=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):

        super(AsymBaggingRFCs, self).__init__(
            base_estimator=BalanceForcedRandomForestClassifier(),
            n_estimators=n_estimators,
            estimator_params=("n_estimators", "criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_density", "bootstrap", "compute_importances",
                              "oob_score", "n_jobs", "verbose",
                              "max_features", "random_state"),
        )

        self.asym_estimators = asym_estimators

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_density = min_density
        self.bootstrap = bootstrap
        self.compute_importances = compute_importances
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_features = max_features
        self.random_state = random_state

        # hack to make us look like a classifier
        self.estimator = RandomForestClassifier()

    def fit(self, X, y):
        # remove from clone
        self.estimators_ = []

        for i in range(self.asym_estimators):
            self._make_estimator()

        for clf in self.estimators_:
            clf.fit(X, y)

    def predict_proba(self, X):
        #return self.estimators_[0].predict_proba(X)

        all_proba = [clf.predict_proba(X) for clf in self.estimators_]

        # reduce by average
        proba = all_proba[0]

        for j in xrange(1, len(all_proba)):
            for k in xrange(2):
                proba[k] += all_proba[j][k]

        for k in xrange(2):
            proba[k] /= self.asym_estimators

        return proba

    def predict(self, X):
        #return self.estimators_[0].predict(X)

        proba = self.predict_proba(X)

        return np.array([max(enumerate(a),
                             key=lambda x: x[1])[0]
                         for a in proba])


"""
class OldAsymBaggingRFCs(BaseEnsemble):
    def __init__(self,
                 asym_estimators=5,

                 # for RFC
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_density=0.1,
                 max_features="auto",
                 bootstrap=True,
                 compute_importances=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):

        super(AsymBaggingRFCs, self).__init__(
            base_estimator=RandomForestClassifier(),
            n_estimators=n_estimators,
            estimator_params=("n_estimators", "criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_density", "bootstrap", "compute_importances",
                              "oob_score", "n_jobs", "verbose",
                              "max_features", "random_state"),
        )

        self.asym_estimators = asym_estimators

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_density = min_density
        self.bootstrap = bootstrap
        self.compute_importances = compute_importances
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_features = max_features
        self.random_state = random_state

        # hack to make us look like a classifier
        self.estimator = RandomForestClassifier()

    def fit(self, X, y):
        # remove from clone
        self.estimators_ = []

        for i in range(self.asym_estimators):
            self._make_estimator()

        X_small, X_large, y_new = get_asym_task(X, y)

        for clf in self.estimators_:
            # sample without replacement
            large_subset = np.array(random.sample(X_large, len(X_small)))
            X_new = np.vstack((X_small, large_subset))

            # shuffle data
            #zipped = zip(X_new, y_new[:])
            #random.shuffle(zipped)
            #X_new, y_new = zip(*zipped)

            clf.fit(X_new, y_new)


    def predict_proba(self, X):
        return self.estimators_[0].predict_proba(X)

        #all_proba = [clf.predict_proba(X) for clf in self.estimators_]

        ## reduce by average
        #proba = all_proba[0]

        #for j in xrange(1, len(all_proba)):
        #    for k in xrange(2):
        #        proba[k] += all_proba[j][k]

        #for k in xrange(2):
        #    proba[k] /= self.asym_estimators

        #return proba

    def predict(self, X):
        return self.estimators_[0].predict(X)

        #proba = self.predict_proba(X)

        #return np.array([max(enumerate(a),
        #                     key=lambda x: x[1])[0]
        #                 for a in proba])
"""
