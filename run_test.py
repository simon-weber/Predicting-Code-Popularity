"""This script handles training and evaluation."""

import functools
from itertools import combinations, chain

import numpy as np
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import sklearn.feature_selection
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import LinearSVC
import sklearn.decomposition

import classes
from features import all_features
from models import Repo
import utils

NGRAM_MIN = 1
NGRAM_MAX = 3  # not inclusive

sorted_stdlib_names = sorted(list(utils.stdlib_module_names()))


def ngrams(mods):
    iters = []
    for i in range(NGRAM_MIN, NGRAM_MAX):
        iters.append(combinations(mods, i))

    return chain(*iters)


def get_classifier():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=1,
        max_features=None,
        #random_state=0,  # random seed is static for comparison
        compute_importances=True,
        n_jobs=-1,  # run on all cores
    )


def _score_func(id_to_class):
    f = functools.partial(
        metrics.classification_report,
        target_names=[id_to_class[i] for i in
                      range(len(classes.classes))])
    return f


def _mod_feature_name(mods):
    return 'imports: ' + ' '.join(mods)


def _select_features(X, y):
    """Return a new instance of the classification source X."""
    estimator = LinearSVC()
    selector = RFECV(estimator, step=.1)
    selector.fit(X, y)
    return selector.transform(X)


def select_kbest_features(X, y):
    selector = sklearn.feature_selection.SelectKBest(
        sklearn.feature_selection.chi2, 15)
    return selector.fit_transform(X, y)


def select_features_stat(X, y):
    selector = sklearn.feature_selection.SelectFpr(
        sklearn.feature_selection.chi2, .0001)
    return selector.fit_transform(X, y)


def select_l1_features(X, y):
    return LinearSVC(C=0.00005, loss="l2", dual=True).fit_transform(X, y)


def select_features(X, y):
    """Return a new instance of the classification source X."""
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X, y)
    return selector.transform(X)

def select_by_pca(X, y):
    return sklearn.decomposition.RandomizedPCA(n_components=15).fit_transform(X, y)


def classify(X, y, id_to_class, vec):
    """Run the given classification task."""
    clf = get_classifier()
    clf.fit(X, y)

    scores = cross_val_score(clf, X, y, score_func=_score_func(id_to_class))
    confusions = cross_val_score(clf, X, y,
                                 score_func=metrics.confusion_matrix)
    confusion = np.apply_over_axes(np.sum, confusions, [0, 0])[0]

    importances = clf.feature_importances_

    if X.shape[1] > 1:
        print 'number of samples:', len(X)
        print
        print
        print "Feature ranking:"
        f_ranks = np.argsort(importances)[::-1]
        id_to_feature = vec.get_feature_names()

        col_offset = max(len(f) for f in id_to_feature) + 1

        for fid in f_ranks:
            name = id_to_feature[fid].ljust(col_offset)
            val = importances[fid]

            if val == 0:
                continue

            val = "%f" % val
            print name, val

    print
    print scores[0]
    print confusion


def _run(repos, features):
    """Train and run a classifier using features from these repos.
    Current classes are used.

    :param repos: a list of Repos
    :param features: a list of strings of feature names
    """
    class_to_id, id_to_class = utils.create_bimap(classes.classes)
    y = np.array([class_to_id[classes.classify(r)] for r in repos])

    # all features except imports are numerical;
    # imports is transformed into n_modules discrete features
    use_imports = False
    if 'imported_stdlib_modules' in features:
        use_imports = True
        mod_feature_dict = {_mod_feature_name(mods): False
                            for mods in ngrams(sorted_stdlib_names)}
        features = [f for f in features if f != 'imported_stdlib_modules']

    dict_repos = []
    for r in repos:
        d = {}

        if use_imports:
            d = mod_feature_dict.copy()

            for mods in ngrams(sorted(r.imported_stdlib_modules)):
                d[_mod_feature_name(mods)] = True

        for fname in features:
            d[fname] = getattr(r, fname)

        dict_repos.append(d)

    vec = DictVectorizer()
    X = vec.fit_transform(dict_repos)
    X = X.todense()

    # classify(X, y, id_to_class, vec)
    classify(select_by_pca(X, y), y, id_to_class, vec)


if __name__ == '__main__':
    ignore = ['imported_modules']
    #ignore += ['imported_stdlib_modules']

    features = [f for f in all_features if f not in ignore]
    # features = ['imported_stdlib_modules']

    _run(Repo.load_sample(), features)
