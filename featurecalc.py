"""This script loads the current features from the pickle, adds in undownloaded
repos, then downloads and calcuates any missing features. Since this can take
some time, the pickle is written after every 50 repos are processed."""

import argparse
import code
import datetime
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

    flush_right = ' ' * 20  # make sure to overwrite status messages

    sys.stdout.write("\rcalculating [{}] {}%{}".format(bar, pct_done, flush_right))
    sys.stdout.flush()


def calculate(f_to_calc, f_to_overwrite, console, download):
    """Calculate a list of features."""

    sys.stdout.write('loading')
    sys.stdout.flush()
    repos = Repo.load_sample()

    seen = 0
    total = len(repos)
    dl_failures = []
    calc_failures = []
    last_write = datetime.datetime.now()

    if f_to_calc or f_to_overwrite or download:
        for repo in repos:
            seen += 1
            success = True

            if download:
                success = utils.clone(repo)

            if not success:
                dl_failures.append(repo)
                continue

            try:
                if f_to_calc:
                    logging.info("calc: %s", repo)
                    repo.calculate_features(f_to_calc)

                if f_to_overwrite:
                    logging.info("calc: %s", repo)
                    repo.calculate_features(f_to_overwrite, overwrite=True)

                repo._clear_support_features()  # we're done with this repo now
            except:
                print  # from status line
                logging.exception("!problem: %s", repo)
                calc_failures.append(repo)
                print

            progress_bar(seen, total)

            since_write = datetime.datetime.now() - last_write

            if since_write > datetime.timedelta(minutes=5):
                sys.stdout.write("\r(writing results)")
                sys.stdout.flush()
                Repo.write_update(repos)

                last_write = datetime.datetime.now()

    print  # from progress bar line

    if dl_failures:
        print "%s failed to download:" % len(dl_failures)
        for f in dl_failures:
            print "  %s" % f
        print

    if calc_failures:
        print "%s failed during calc:" % len(calc_failures)
        for f in calc_failures:
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

    if args.overwrite == ['all']:
        args.overwrite = all_features.keys()

    calculate(args.calc, args.overwrite, args.console, not args.nodownload)

if __name__ == '__main__':
    main()
