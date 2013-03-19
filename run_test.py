"""This script handles training and evaluation."""

import functools

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics

import classes
from features import all_features
from models import Repo
import utils


def get_classifier():
    return RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=1,
        #random_state=0,  # random seed is static for comparison
        compute_importances=True,
    )


def _score_func(id_to_class):
    f = functools.partial(
        metrics.classification_report,
        target_names=[id_to_class[i] for i in
                      range(len(classes.classes))])
    return f


def _mod_feature_name(mod):
    return 'imports-' + mod


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
        features = [f for f in features if f != 'imported_stdlib_modules']

    dict_repos = []
    for r in repos:
        d = {}

        if use_imports:
            d.update({_mod_feature_name(mod): False
                      for mod in utils.stdlib_module_names()})
            for mod in r.imported_stdlib_modules:
                d[_mod_feature_name(mod)] = True

        for fname in features:
            d[fname] = getattr(r, fname)

        dict_repos.append(d)

    vec = DictVectorizer()
    X = vec.fit_transform(dict_repos)
    X = X.todense()

    clf = get_classifier()
    clf.fit(X, y)

    scores = cross_val_score(clf, X, y, score_func=_score_func(id_to_class))
    confusions = cross_val_score(clf, X, y,
                                 score_func=metrics.confusion_matrix)
    confusion = np.apply_over_axes(np.sum, confusions, [0, 0])[0]

    importances = clf.feature_importances_

    if len(features) > 1:
        print 'number of samples:', len(X)
        print
        print
        print "Feature ranking:"
        f_ranks = np.argsort(importances)[::-1]
        id_to_feature = vec.get_feature_names()

        col_offset = max(len(f) for f in features) + 1

        for i in xrange(len(features)):
            name = id_to_feature[f_ranks[i]].ljust(col_offset)
            val = "%f" % importances[f_ranks[i]]

            print name, val

    print
    print scores[0]
    print confusion


if __name__ == '__main__':
    ignore = ['imported_modules']
    features = [f for f in all_features
                if f not in ignore]

    _run(Repo.load_sample(), features)
