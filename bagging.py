from collections import Counter
import random

import numpy as np
from sklearn.ensemble.base import BaseEnsemble
from sklearn.ensemble import RandomForestClassifier


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

        if asym_estimators % 2 == 0:
            # disallow even ensembles; majority voting is performed
            asym_estimators += 1

        # will not work with eg grid search, would need to create inside fit
        for i in range(asym_estimators):
            self._make_estimator()

        # hack to make us look like a classifier
        self.estimator = self.estimators_[0]

    def fit(self, X, y):
        # assume only 2 classes
        large_class, small_class = [pair[0] for pair in Counter(y).most_common()]

        X_small = np.array([X[i] for (i, cls) in enumerate(y)
                            if cls == small_class])
        X_large = np.array([X[i] for (i, cls) in enumerate(y)
                            if cls == large_class])

        y_new = np.array(([small_class] * len(X_small)) +
                         ([large_class] * len(X_small)))

        for clf in self.estimators_:
            # sample with replacement
            large_subset = np.array(random.sample(X_large, len(X_small)))
            X_new = np.vstack((X_small, large_subset))

            clf.fit(X_new, y_new)

    def predict_proba(self, X):
        all_proba = [clf.predict_proba(X) for clf in self.estimators_]

        # reduce
        proba = all_proba[0]

        for j in xrange(1, len(all_proba)):
            for k in xrange(2):
                proba[k] += all_proba[j][k]

        for k in xrange(2):
            proba[k] /= self.asym_estimators

        return proba

    def predict(self, X):
        proba = self.predict_proba(X)

        return np.array([max(enumerate(a),
                             key=lambda x: x[1])[0]
                         for a in proba])
