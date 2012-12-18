"""This script handle training and evaluation."""

import functools

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from features import all_features
import utils
from sample import classes, repo_class_names, repos


#populate sample and supporting structures
f_dicts = utils.load_f_dicts(merge_new=False)

class_to_id, id_to_class = utils.create_bimap(classes.keys())

#function to use when evaluating results
score_func = functools.partial(metrics.classification_report,
                               target_names=[id_to_class[i] for i in
                                             range(len(classes))])

confusion_func = metrics.confusion_matrix


def create_samples(id_to_feature, num_features):
    """Return a numpy array representation of all repos, using features from
    f_dicts."""
    samples = []
    for repo in repos:
        #the repo will not be there if we failed to download it
        if repo in f_dicts:
            f_dict = f_dicts[repo]
            sample = [f_dict[id_to_feature[i]] for i in xrange(num_features)]
            samples.append(sample)
        else:
            print "not found: %s" % repo

    return np.array(samples)


def create_training_data(id_to_feature, num_features):
    """Return a tuple (X, y):
        X: array-like of [n_samples, n_features]
        y: array-like of [n_samples], values are ints that map to classes
    """

    X = create_samples(id_to_feature, num_features)

    class_names = (repo_class_names[repo] for repo in repos if repo in f_dicts)
    y = np.array([class_to_id[class_name] for class_name in class_names])

    return (X, y)


def get_classifier(X, y):
    return RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=1,
        random_state=0,  # random seed is static for comparison
        compute_importances=True,
    )


def run(id_to_feature, num_features):
    X, y = create_training_data(id_to_feature, num_features)
    clf = get_classifier(X, y)

    clf.fit(X, y)
    scores = cross_val_score(clf, X, y, score_func=score_func)
    confusions = cross_val_score(clf, X, y,
                                score_func=confusion_func)
    confusion = np.apply_over_axes(np.sum, confusions, [0, 0])[0]

    importances = clf.feature_importances_

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

    print
    print scores[0]
    print confusion


if __name__ == '__main__':
    feature_to_id, id_to_feature = utils.create_bimap(all_features.keys())
    num_features = len(id_to_feature)

    run(id_to_feature, num_features)

    #Calculate individual feature reports
    #for feature in id_to_feature.values():
    #    print
    #    print feature
    #    run({0: feature}, 1)
