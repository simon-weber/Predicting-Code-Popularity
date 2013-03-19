import functools

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics

import classes
from models import Repo
import utils


def get_classifier(X, y):
    return RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=1,
        random_state=0,  # random seed is static for comparison
        compute_importances=True,
    )


if __name__ == '__main__':
    repos = Repo.load_sample()

    class_to_id, id_to_class = utils.create_bimap(classes.classes)

    dict_repos = []
    for r in repos:
        d = {mod: False for mod in utils.stdlib_module_names()}

        for mod in r.imported_stdlib_modules:
            d[mod] = True
        dict_repos.append(d)

    vectorizer = DictVectorizer(sparse=False)

    y = np.array([class_to_id[classes.classify(r)] for r in repos])
    X = vectorizer.fit_transform(dict_repos)

    clf = get_classifier(X, y)
    clf.fit(X, y)

    #function to use when evaluating results
    score_func = functools.partial(metrics.classification_report,
                                   target_names=[id_to_class[i] for i in
                                                 range(len(id_to_class))])

    scores = cross_val_score(clf, X, y, score_func=score_func)
    confusion_func = metrics.confusion_matrix
    confusions = cross_val_score(clf, X, y,
                                 score_func=confusion_func)
    confusion = np.apply_over_axes(np.sum, confusions, [0, 0])[0]

    importances = clf.feature_importances_

    """
    if num_features > 1:
        print
        print
        print
        print 'number of samples:', len(X)
        print
        print 'class distribution:'
        for class_name, samples in classes.items():
            print "%s: %s" % (class_name, len(samples))

        print
        print "Feature ranking:"
        f_ranks = np.argsort(importances)[::-1]
        for i in xrange(num_features):
            print "%s \t %f" % (id_to_feature[f_ranks[i]],
                                importances[f_ranks[i]])
    """

    print
    print scores[0]
    print confusion
    import code
    code.interact(local=locals())
