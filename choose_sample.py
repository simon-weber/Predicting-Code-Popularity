"""This script filters valid repos, then selects a random sample of some
specified size. The sample is written as Python code to the classes.py file."""

import datetime
import os
import random
import shutil

from models import ERepo

sample_sizes = {'high': 150,
                'med': 150,
                'low': 150,
                'un': 150}

class_ranges = {'un': (3, 10),
                'low': (11, 100),
                'med': (101, 350),
                'high': (351, 100000)}


def populate_classes():
    cutoff = datetime.datetime.now() - datetime.timedelta(days=6 * 30)

    frepos = [r for r in ERepo.select().where(ERepo.created_at < cutoff)]
    frepos = [r for r in frepos if
              r.size > 0 and
              r._stars > 2 and
              r.fork == False and
              r.master_branch == 'master' and
              not r.size > 30720 and  # avoid huge repos
              not 'dotfile' in r._user_repo.lower() and
              not 'sublime' in r._user_repo.lower()  # avoid SublimeText config
              ]

    print 'total filtered:', len(frepos)

    classes = {name: [] for name in class_ranges}
    for cls_name, (l, h) in class_ranges.items():
        classes[cls_name].extend(x for x in frepos if l <= x._stars <= h)

    return classes


def get_samples(classes):
    samples = {name: [] for name in classes}
    for cls_name, num_samples in sample_sizes.items():
        repos_in_class = classes[cls_name]
        print "%s: get %s/%s" % (cls_name, num_samples, len(repos_in_class))
        samples[cls_name].extend(random.sample(repos_in_class, num_samples))

    return samples


def write_samples(samples):
    #convert lists to tuples
    output = {}
    for cls_name, sample in samples.items():
        output[cls_name] = tuple(sample)

    #move old file
    if os.path.exists('classes.py'):
        shutil.move('classes.py', 'classes.py.old')

    #write new file
    with open('classes.py', 'w') as f:
        f.write('classes = ')
        f.write(repr(output))


if __name__ == '__main__':
    classes = populate_classes()
    samples = get_samples(classes)
    write_samples(samples)

    print 'wrote out:'
    for cls, sample in samples.items():
        print "%s: %s  (eg %s)" % (cls, len(sample), sample[0])
