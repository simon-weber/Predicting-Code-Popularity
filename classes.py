from datetime import datetime

classes = ['high', 'low']
cutoff = .5


def score(repo):
    stars_per_day = float(repo.stars) / (datetime(*repo.fetch_ymd)
                                         - repo.creation_date).days
    return stars_per_day


def classify(repo):
    stars_per_day = score(repo)

    if stars_per_day > cutoff:
        return 'high'
    else:
        return 'low'


# old, absolute-star classes
#def classify(repo):
#    """Return a class name for this repo."""
#    if repo.stars > 10:
#        return 'high'
#    else:
#        return 'un'
