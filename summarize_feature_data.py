"""A script to compare the feature data collected."""

from collections import defaultdict, Counter

import numpy as np

from classes import classify
from features import all_features
from models import Repo


def summarize_imports(class_map):
    counter_map = defaultdict(Counter)
    for clsname, repos in class_map.items():
        for r in repos:
            counter_map[clsname].update(r.imported_stdlib_modules)

    # normalize to percent occurance
    for clsname, counter in counter_map.items():
        num_repos = float(len(class_map[clsname]))
        for mod in counter:
            counter[mod] /= num_repos

    print 'top imports'
    for clsname, counter in counter_map.items():
        print clsname
        for mod, occurs in counter.most_common(20):
            print "    {} {:.2g}".format(mod, occurs)
        print
    print

    if len(class_map) == 2:
        print 'biggest differences'
        a_name, b_name = counter_map.keys()
        print "(left is %s)" % a_name
        print

        differences = Counter()
        differences.update(counter_map[a_name])
        differences.subtract(counter_map[b_name])

        for mod in differences:
            differences[mod] = abs(differences[mod])

        for mod, delta in differences.most_common(20):
            print " {} {:.2g}".format(mod, delta)
            print "    {:.2g} {:.2g}".format(
                counter_map[a_name][mod], counter_map[b_name][mod])


# def process(repos):
#     feature_data = {}  # eg 'ReadmeSize' -> np.array(results)
#
#     for feature in features:
#         f_data = []
#         for repo in repos:
#             d = f_dicts.get(repo)
#             if d:
#                 f_data.append(d[feature])
#
#         feature_data[feature] = np.array(f_data)
#
#     for feature, ar in feature_data.items():
#         print '\t'.join(str(x) for x in (
#             feature,
#             np.amin(ar),
#             np.amax(ar),
#             np.median(ar),
#             np.mean(ar),
#             np.std(ar)
#         ))


if __name__ == '__main__':
    class_map = defaultdict(list)
    for r in Repo.load_sample():
        class_map[classify(r)].append(r)

    summarize_imports(class_map)

    # for cls in classes:
    #     print
    #     print "=====>", cls
    #     print
    #     process(classes[cls])
