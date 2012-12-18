"""A script to interactively examine the distribution of repos."""

import cPickle as pickle
import datetime
import numpy as np
import random

from config import feature_pickle_name
from models import ERepo
import utils


cutoff = datetime.datetime.now() - datetime.timedelta(days=6 * 30)

frepos = [r for r in ERepo.select().where(ERepo.created_at < cutoff)]
frepos = [r for r in frepos if
          r.size > 0 and
          r._stars > 2 and
          r.fork == False and
          r.master_branch == 'master' and
          not (r.size > 30720) #and r._stars < 100)  # avoid huge repos, but keep pop
          ]

print 'filtered:', len(frepos)


def divy(ranges):
    res = []
    for l, h in ranges:
        res.append([x for x in frepos if l <= x._stars <= h])
    return res


def calc(chunks):
    for repos in chunks:
        size_ar = np.array([r.size for r in repos])

        print 'num', len(repos)
        print (100.0 * len(repos) / len(frepos)), '%'
        print 'gigs:', (np.sum(size_ar) / 1048576.0)
        print 'md K:', np.median(size_ar)
        print

    print '\n--\n'


def samples(chunks):
    return [random.sample(repos, 20) for repos in chunks]


def add_to_pickle(samples):
    try:
        with open(feature_pickle_name, 'rb') as f:
            f_dicts = pickle.load(f)
    except IOError:
        print 'could not open pickle'

    for repos in samples:
        for repo in repos:
            if repo not in f_dicts:
                f_dicts[repo._user_repo] = {}

    with utils.FaultTolerantFile(feature_pickle_name) as f:
        pickle.dump(f_dicts, f, pickle.HIGHEST_PROTOCOL)


d = divy([(3, 10), (11, 100), (101, 350), (351, 100000)])
#s = samples(d)

print 'd = divy([(3, 10), (11, 100), (101, 500), (501, 100000)])'
#print 's = samples(d)'
print calc(d)


import code
code.interact(local=locals())
