"""A script to compare the feature data collected."""

from collections import defaultdict, Counter

import numpy as np

from classes import classes
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
        print 'largest absolute differences'
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

    print

    if len(class_map) == 2:
        print 'relative differences'
        print '(only modules in both classes are included)'
        a_name, b_name = counter_map.keys()
        a_cnt, b_cnt = counter_map[a_name], counter_map[b_name]
        print "(left is %s)" % a_name
        print

        deltas = []  # (import, factor, more_frequent_class)
        for mod in (counter_map[a_name].viewkeys() &
                    counter_map[b_name].viewkeys()):
            # ugh
            vals = zip((a_cnt[mod], b_cnt[mod]), (a_name, b_name))
            big_val, big_class_name = max(vals, key=lambda p: p[0])
            small_val, small_class_name = min(vals, key=lambda p: p[0])

            deltas.append((mod, 1.0 * small_val / big_val, big_class_name))

        deltas.sort(key=lambda t: t[1])

        def output(deltas):
            for mod, rel_delta, big_class_name in deltas:
                print " {} {:.2g}".format(mod, rel_delta)
                print "    {:.2g} {:.2g}".format(
                    counter_map[a_name][mod], counter_map[b_name][mod])

        print 'smallest'
        output(deltas[-20:])
        print

        print 'largest'
        output(deltas[:20])


def summarize_features(class_map, feature_names, class_names,
                       funcs=[np.amin, np.amax, np.median, np.mean, np.std]):
    feature_summaries = {}  # eg 'ReadmeSize' -> np.array(num_classes, num_funcs)

    for feature_name in feature_names:
        class_summaries = []
        for class_name in class_names:
            repos = class_map[class_name]
            feature_data = [getattr(repo, feature_name) for repo in repos]

            class_summaries.append([f(feature_data) for f in funcs])

        feature_summaries[feature_name] = np.array(class_summaries)

    for feature_name, summary in feature_summaries.items():
        print feature_name
        for i, class_name in enumerate(class_names):
            print '  ', class_name
            print '    ', '\t'.join(str(e) for e in summary[i])

        print '-----'
        print


if __name__ == '__main__':
    class_map = Repo.load_sample(separate=True)

    #summarize_features(class_map, ['readme_size'], sorted(classes))

    summarize_imports(class_map)
