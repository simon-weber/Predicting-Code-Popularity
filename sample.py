"""The names of repos in the sample, along with numpy-representations of all
data."""

import classes

classes = classes.classes

#del classes['un']
#del classes['low']
#del classes['med']
#del classes['high']

repo_class_names = {repo: class_name
                    for class_name, repos in classes.items()
                    for repo in repos}

repos = [r for repos in classes.values() for r in repos]
