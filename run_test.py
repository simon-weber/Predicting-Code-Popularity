"""This script handles training and evaluation."""

import functools
from itertools import combinations, chain, product

import numpy as np
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import sklearn.feature_selection
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import LinearSVC
from sklearn import svm
import sklearn.decomposition

from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid

from bagging import AsymBaggingRFCs
import classes
from features import all_features
from models import Repo
import utils

NGRAM_MIN = 1
NGRAM_MAX = 2  # not inclusive

sorted_stdlib_names = sorted(list(utils.stdlib_module_names()))


def ngrams(mods):
    iters = []
    for i in range(NGRAM_MIN, NGRAM_MAX):
        iters.append(combinations(mods, i))

    return chain(*iters)


def RandomForest():
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


def select_features_rfecv(X, y):
    """Return a new instance of the classification source X."""
    estimator = LinearSVC()
    selector = RFECV(estimator, step=10)
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


def select_percentile_best_features(X, y):
    """Return a new instance of the classification source X."""
    selector = SelectPercentile(f_classif, percentile=1)
    return selector.fit_transform(X, y)


def select_by_pca(X, y):
    return sklearn.decomposition.RandomizedPCA(n_components=15).fit_transform(X, y)


def summarize(a, funcs):
    result = [f(a, axis=0) for f in funcs]
    return np.array(result)


def format_summary(a):
    """a is a vector of (median, min, max, std)."""
    return "{0:<6.3g} {3:<6.3g}  ({1:.3g} - {2:.3g})".format(*a)


def benchmark(clf, X, y, feature_names, cv_runs=5):
    """Run a classification task and output performance information."""
    clf.fit(X, y)

    scores = cross_val_score(clf, X, y, cv=cv_runs,
                             score_func=metrics.precision_recall_fscore_support)

    labels = product(classes.classes, 'prec recall fscore support'.split())

    print "over %s runs:" % cv_runs

    funcs = (np.median, np.min, np.max, np.std)
    summaries = summarize(scores, funcs).transpose().reshape(8, len(funcs))

    for (label, summary) in zip(labels, summaries):
        print '/'.join(label)
        print (' ' * 13) + format_summary(summary)
        print

    if hasattr(clf, 'feature_importances_') and X.shape[1] > 1:
        importances = clf.feature_importances_
        print 'number of samples:', len(X)
        print
        print
        print "Feature ranking:"
        f_ranks = np.argsort(importances)[-20:][::-1]
        id_to_feature = feature_names

        col_offset = max(len(f) for f in id_to_feature) + 1

        for fid in f_ranks:
            name = id_to_feature[fid].ljust(col_offset)
            val = importances[fid]

            if val == 0:
                continue

            val = "%f" % val
            print name, val

    # elif hasattr(clf, 'coef_') and X.shape[1] > 1:
    #     for i, category in enumerate(classes.classes.keys()):
    #         top = np.argsort(clf.coef_[i])[-20:]
    #         print(trim("%s: %s"
    #               % (category, " ".join(feature_names[top]))))


#def classify(X, y, id_to_class, vec):
#    """Run the given classification task."""
#    clf = get_classifier()
#    clf.fit(X, y)
#
#    scores = cross_val_score(clf, X, y, score_func=_score_func(id_to_class), cv=5)
#    confusions = cross_val_score(clf, X, y,
#                                 score_func=metrics.confusion_matrix, cv=5)
#    confusion = np.apply_over_axes(np.sum, confusions, [0, 0])[0]
#
#    if hasattr(clf, 'feature_importances_') and X.shape[1] > 1:
#        importances = clf.feature_importances_
#        print 'number of samples:', len(X)
#        print
#        print
#        print "Feature ranking:"
#        f_ranks = np.argsort(importances)[-20:]
#        id_to_feature = vec.get_feature_names()
#
#        col_offset = max(len(f) for f in id_to_feature) + 1
#
#        for fid in f_ranks:
#            name = id_to_feature[fid].ljust(col_offset)
#            val = importances[fid]
#
#            if val == 0:
#                continue
#
#            val = "%f" % val
#            print name, val
#
#
#    print
#    print scores
#    print confusion


# def benchmark(clf, X, y):
#     print 80 * '_'
#     print "Training: "
#     print clf
#     t0 = time()
#     clf.fit(X_train, y_train)
#     train_time = time() - t0
#     print "train time: %0.3fs" % train_time
#
#     t0 = time()
#     pred = clf.predict(X_test)
#     test_time = time() - t0
#     print "test time:  %0.3fs" % test_time
#
#     score = metrics.f1_score(y_test, pred)
#     print "f1-score:   %0.3f" % score
#
#     if hasattr(clf, 'coef_'):
#         print "dimensionality: %d" % clf.coef_.shape[1]
#         print "density: %f" % density(clf.coef_)
#
#         if opts.print_top10 and feature_names is not None:
#             print "top 10 keywords per class:"
#             for i, category in enumerate(categories):
#                 top10 = np.argsort(clf.coef_[i])[-10:]
#                 print trim("%s: %s" % (
#                     category, " ".join(feature_names[top10])))
#         print
#
#     if opts.print_report:
#         print "classification report:"
#         print metrics.classification_report(y_test, pred,
#                                             target_names=categories)
#
#     if opts.print_cm:
#         print "confusion matrix:"
#         print metrics.confusion_matrix(y_test, pred)
#
#     print
#     clf_descr = str(clf).split('(')[0]
#     return clf_descr, score, train_time, test_time


def _run(repos, features):
    """Train and run a classifier using features from these repos.
    Current classes are used.

    :param repos: a list of Repos
    :param features: a list of strings of feature names
    """
    class_to_id, id_to_class = utils.create_bimap(classes.classes)
    y = np.array([class_to_id[classes.classify(r)] for r in repos])

    # all features except imports are numerical;
    # imports become one-hot boolean ngrams
    use_imports = False
    if 'imported_stdlib_modules' in features:
        use_imports = True
        # mod_feature_dict = {_mod_feature_name(mods): False
        #                     for mods in ngrams(sorted_stdlib_names)}
        features = [f for f in features if f != 'imported_stdlib_modules']

    dict_repos = []
    for r in repos:
        d = {}

        if use_imports:
            # d = mod_feature_dict.copy()

            mods = [m for m in r.imported_stdlib_modules if m in
                    set(['hashlib', '__future__', 'functools', 'threading', 'warnings', 'base64',
                         'traceback', 'socket', 'urlparse', 'subprocess', 'tempfile', 'json',
                         'unittest', 'errno', 'StringIO', 're', 'glob', 'signal', 'inspect',
                         'operator'])]

            for mods in ngrams(mods):
                d[_mod_feature_name(mods)] = True

        for fname in features:
            d[fname] = getattr(r, fname)

        dict_repos.append(d)

    vec = DictVectorizer()
    X = vec.fit_transform(dict_repos)
    #X = X.todense()

    feature_names = vec.get_feature_names()

    dense_X = X.toarray()

    # print 'RFC'
    # benchmark(RandomForest(), dense_X, y, feature_names)

    print 'asym bagging RF'
    benchmark(AsymBaggingRFCs(5,
                              n_estimators=100,
                              max_depth=None,
                              min_samples_split=1,
                              max_features=None,
                              #random_state=0,  # random seed is static for comparison
                              compute_importances=True,
                              n_jobs=-1,  # run on all cores
                              ),
              dense_X, y, feature_names
              )

    # print 'AdaBoost'
    # benchmark(AdaBoostClassifier(n_estimators=100), dense_X, y, feature_names)

    #print 'ridge'
    #benchmark(RidgeClassifier(tol=1e-2, solver="lsqr"), X, y, feature_names)

    #print 'NN'
    #benchmark(Perceptron(n_iter=50), X, y, feature_names)

    #print 'passive aggressive'
    #benchmark(PassiveAggressiveClassifier(n_iter=50), X, y, feature_names)

    #print 'kNN'
    #benchmark(KNeighborsClassifier(n_neighbors=10), X, y, feature_names)

    #print 'SGD'
    #benchmark(SGDClassifier(n_jobs=-1, alpha=.0001, n_iter=np.ceil(10**3), penalty="elasticnet", shuffle=True),
    #          X, y, feature_names)

    #print 'nearest centroid'
    #benchmark(NearestCentroid(), X, y, feature_names)

    #print 'naive bayes'
    #benchmark(MultinomialNB(alpha=.01), X, y, feature_names)

    #print 'naive bayes (bernoulli)'
    #benchmark(BernoulliNB(alpha=.01), X, y, feature_names)

    #classify(X, y, id_to_class, vec)
    # classify(select_by_pca(X, y), y, id_to_class, vec)


if __name__ == '__main__':
    ignore = ['imported_modules']
    #ignore += ['imported_stdlib_modules']

    features = [f for f in all_features if f not in ignore]
    #features = ['imported_stdlib_modules']

    _run(Repo.load_sample(), features)
