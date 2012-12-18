"""This script loads the current features from the pickle, adds in undownloaded
repos, then downloads and calcuates any missing features. Since this can take
some time, the pickle is written after every 50 repos are processed."""

import logging
logging.basicConfig(filename='calcfeatures.log', level=logging.DEBUG)
import sys

from features import all_features
import utils


def progress_bar(processed, total):
    pct_done = int(100.0 * processed / total)

    bar = '#' * (pct_done / 5)
    bar = bar.ljust(20)

    sys.stdout.write("\r[{}] {}%".format(bar, pct_done))
    sys.stdout.flush()


if __name__ == '__main__':
    f_dicts = utils.load_f_dicts()

    seen = 0
    total = len(f_dicts)
    failures = []

    #download/calculate uncalculated features
    for user_repo, f_dict in f_dicts.iteritems():
        seen += 1

        success = utils.download(user_repo)
        if not success:
            failures.append(user_repo)
            continue

        for f_name, feature in all_features.iteritems():
            if f_name not in f_dict:
                feature.calculate(user_repo, f_dict)

        progress_bar(seen, total)
        if seen % 50 == 0 and not failures:
            utils.persist_f_dicts(f_dicts)

    #remove failures. another attempt will be made on the next run
    for bad_repo in failures:
        del f_dicts[bad_repo]

    print

    if failures:
        print "%s failures" % len(failures)
        for f in failures:
            print "  %s" % f

    print 'writing out...'
    utils.persist_f_dicts(f_dicts)
