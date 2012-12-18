"""A script to compare the feature data collected."""

import numpy as np

from features import all_features
import utils
from sample import classes


f_dicts = utils.load_f_dicts(merge_new=False)
features = all_features.keys()


def process(repos):
    feature_data = {}  # eg 'ReadmeSize' -> np.array(results)

    for feature in features:
        f_data = []
        for repo in repos:
            d = f_dicts.get(repo)
            if d:
                f_data.append(d[feature])

        feature_data[feature] = np.array(f_data)

    for feature, ar in feature_data.items():
        print '\t'.join(str(x) for x in (
            feature,
            np.amin(ar),
            np.amax(ar),
            np.median(ar),
            np.mean(ar),
            np.std(ar)
        ))


if __name__ == '__main__':
    for cls in classes:
        print
        print "=====>", cls
        print
        process(classes[cls])
