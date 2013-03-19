classes = ['high', 'un']


def classify(repo):
    """Return a class name for this repo."""
    if repo.stars > 10:
        return 'high'
    else:
        return 'un'
