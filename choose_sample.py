"""This script filters valid repos, then selects a random sample of some
specified size. The sample is written as Python code to the classes.py file."""

import json
import os
import random

from config import config
from models import Repo

sample_sizes = {'high': 150,
                #'med': 150,
                #'low': 150,
                'un': 150}

class_ranges = {'un': (3, 10),
                # 'low': (11, 100),
                # 'med': (101, 350),
                'high': (351, 100000)}


def populate_classes():
    repos = Repo.load()
    print 'number of repos:', len(repos)

    filtered = [r for r in repos if
                r.size > 0 and
                r.stars > 2 and
                r.size < 30720 and  # avoid huge repos
                not r.fork and
                not 'dotfile' in r.name and
                not 'sublime' in r.name  # avoid SublimeText config
                ]

    print 'total filtered:', len(filtered)

    classes = {name: [] for name in class_ranges}
    for cls_name, (l, h) in class_ranges.items():
        classes[cls_name].extend(x for x in filtered if l <= x.stars <= h)

    return classes


def get_samples(classes):
    samples = {name: [] for name in classes}
    for cls_name, num_samples in sample_sizes.items():
        repos_in_class = classes[cls_name]
        print "%s: get %s/%s" % (cls_name, num_samples, len(repos_in_class))
        samples[cls_name].extend(random.sample(repos_in_class, num_samples))

    return samples


def write_samples(samples):
    # sample is now one big list
    output = [r.name for c in samples.values() for r in c]

    sample_path = os.path.join(config['current_snapshot'], config['current_sample'])

    with open(sample_path, 'wb') as f:
        json.dump(output, f)


if __name__ == '__main__':
    classes = populate_classes()
    samples = get_samples(classes)
    write_samples(samples)

    print 'wrote out:'
    for cls, sample in samples.items():
        print "%s: %s  (eg %s)" % (cls, len(sample), sample[0])
