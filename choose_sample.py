"""This script filters valid repos, then selects a random sample of some
specified size. The sample is written as Python code to the classes.py file."""

from collections import defaultdict
import datetime
import json
import os
import random

from config import config
import classes
from models import Repo

sample_sizes = {'high': 260,
                'low': 1000,
                }


def get_samples():
    """Return a {'class': [reponames]}."""

    repos = Repo.load()
    fetch_dates = [datetime.datetime(*(r.fetch_ymd)) for r in repos]

    print 'number of repos:', len(repos)

    latest_fetch = max(fetch_dates)
    print 'fetched between %s and %s' % (min(fetch_dates), latest_fetch)
    print

    filtered = [r for r in repos if
                30720 > r.size > 0 and  # not foolproof to avoid big repos
                r.stars > 1 and
                not r.fork and
                not 'dotfile' in r.name.lower() and
                not 'sublime' in r.name.lower()  # avoid SublimeText config
                ]
    print 'after noise filter:', len(filtered)

    filtered = [r for r in filtered if
                ((latest_fetch - r.creation_date) >
                 datetime.timedelta(30))
                ]
    print 'exluding very new:', len(filtered)

    filtered = [r for r in filtered if
                r.stars > 5 and
                classes.score(r) > (1 / 30)
                ]
    print 'exluding very unpopular:', len(filtered)

    score_pairs = [(classes.score(r), r) for r in filtered]
    score_pairs.sort(key=lambda x: x[0])

    # top 1k, bottom 1k.
    return {'high': [r.name for (score, r) in score_pairs[-1000:]],
            'low': [r.name for (score, r) in score_pairs[:1000]],
            }

#def get_samples(class_map):
#    sample_map = defaultdict(list)
#
#    # take the top of top, bottom of bottom
#    select_map = {'high': lambda repos, k: sorted(repos, reverse=True)[:k],
#                  'low': lambda repos, k: sorted(repos)[:k]}
#
#    for cls_name, num_samples in sample_sizes.items():
#        repos_in_class = class_map[cls_name]
#        select = select_map[cls_name]
#
#        print "%s: get %s/%s" % (cls_name, num_samples, len(repos_in_class))
#        sample_map[cls_name].extend(select(repos_in_class, num_samples))
#
#    return sample_map


def write_samples(output):
    sample_path = os.path.join(config['current_snapshot'], config['current_sample'])

    with open(sample_path, 'wb') as f:
        json.dump(output, f)


if __name__ == '__main__':
    samples = get_samples()
    print
    write_samples(samples)

    print 'to write:'
    for cls, sample in samples.items():
        print "%s: %s  (eg %s)" % (cls, len(sample),
                                   [name for name in (random.sample(sample, 10))])
