"""This script loads the current features from the pickle, adds in undownloaded
repos, then downloads and calcuates any missing features. Since this can take
some time, the pickle is written after every 50 repos are processed."""

import argparse
import code
import logging
import sys

from features import all_features
from models import Repo
import utils

logging.basicConfig(filename='calcfeatures.log', level=logging.DEBUG)


def progress_bar(processed, total):
    pct_done = int(100.0 * processed / total)

    bar = '#' * (pct_done / 5)
    bar = bar.ljust(20)

    sys.stdout.write("\r[{}] {}%".format(bar, pct_done))
    sys.stdout.flush()


def calculate(f_to_calc, f_to_overwrite, console, download):
    """Calculate a list of features."""

    print 'loading...',
    sys.stdout.flush()
    repos = Repo.load_sample()
    print 'done'

    seen = 0
    total = len(repos)
    dl_failures = []

    if f_to_calc or f_to_overwrite or download:
        for repo in repos:
            seen += 1
            success = True

            if download:
                success = utils.download(repo)

            if not success:
                dl_failures.append(repo)
                continue

            if f_to_calc:
                repo.calculate_features(f_to_calc)

            if f_to_overwrite:
                repo.calculate_features(f_to_overwrite, overwrite=True)

            repo._clear_support_features()  # we're done with this repo now

            progress_bar(seen, total)

            # periodically persist calculations
            if seen % 50 == 0 and f_to_calc or f_to_overwrite:
                Repo.write_update(repos)

    print  # from progress bar line

    if dl_failures:
        print "%s failed to download:" % len(dl_failures)
        for f in dl_failures:
            print "  %s" % f
        print

    if console:
        message = ('`repos` contains results;\n'
                   'use ^d to write out or `exit()` to cancel')
        code.interact(message, local=locals())

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

    parser.add_argument('--nodownload', action='store_true',
                        help="do not download code if it's missing")

    args = parser.parse_args()

    if not args.console and not (args.calc or args.overwrite):
        parser.print_help()
        return

    if args.calc == ['all']:
        args.calc = all_features.keys()

    calculate(args.calc, args.overwrite, args.console, not args.nodownload)

if __name__ == '__main__':
    main()
