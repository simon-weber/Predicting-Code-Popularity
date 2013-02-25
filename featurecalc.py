"""This script loads the current features from the pickle, adds in undownloaded
repos, then downloads and calcuates any missing features. Since this can take
some time, the pickle is written after every 50 repos are processed."""

import argparse
import code
import logging
import sys

from models import Repo
import utils

logging.basicConfig(filename='calcfeatures.log', level=logging.DEBUG)


def progress_bar(processed, total):
    pct_done = int(100.0 * processed / total)

    bar = '#' * (pct_done / 5)
    bar = bar.ljust(20)

    sys.stdout.write("\r[{}] {}%".format(bar, pct_done))
    sys.stdout.flush()


def calculate(f_to_calc, f_to_overwrite, console):
    """Calculate a list of features."""

    #if feature_names is None:
    #    feature_names = all_features.keys()

    print 'loading...'
    repos = Repo.load_sample()

    seen = 0
    total = len(repos)
    dl_failures = []

    #download/calculate
    for repo in repos:
        seen += 1

        success = utils.download(repo)
        if not success:
            dl_failures.append(repo)
            continue

        if f_to_calc:
            repo.calculate_features(f_to_calc)

        if f_to_overwrite:
            repo.calculate_features(f_to_overwrite, overwrite=True)

        progress_bar(seen, total)
        #if seen % 50 == 0 and not dl_failures:
        #    utils.persist_f_dicts(f_dicts)

    print  # from progress bar line

    if dl_failures:
        print "%s failed to download:" % len(dl_failures)
        for f in dl_failures:
            print "  %s" % f
        print

    if console:
        code.interact(local=locals())

    print 'writing out...'
    Repo.write_update(repos)


def main():
    parser = argparse.ArgumentParser(description='Calculate features for the current sample.')

    parser.add_argument('--calc', nargs='*', metavar='feature',
                        help='calculate the given features, but do not overwrite')

    parser.add_argument('--overwrite', nargs='*', metavar='feature',
                        help='calculate the given features, overwriting any current value')

    parser.add_argument('--console', action='store_true',
                        help=('after calculation and before write-out, open a repl.'
                              ' the list of Repos is available as `repos`.'
                              ' call `exit()` to abort before writing out,'
                              ' otherwise use EOF to continue.'))

    args = parser.parse_args()

    if not args.console and not (args.calc or args.overwrite):
        parser.print_help()
        return

    calculate(args.calc, args.overwrite, args.console)

if __name__ == '__main__':
    main()
