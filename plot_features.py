import code

import numpy as np
import matplotlib.pyplot as plt

from classes import classes
from models import Repo

_sample = None  # global memoized Repo.load_sample(separate=True)


class FeatureChart(object):
    feature_name = None  # provide in subclass

    def __init__(self, sample_dict=None, class_names=None):
        global _sample

        if self.feature_name is None:
            raise Exception('Provide feature_name field in subclass.')

        if sample_dict is None:
            if _sample is None:
                _sample = Repo.load_sample(separate=True)
            sample_dict = _sample

        if class_names is None:
            class_names = sorted(classes)

        self.class_names = class_names
        self.figure, self.ax = plt.subplots()

        hist_data = []
        for clsname in class_names:
            hist_data.append([getattr(repo, self.feature_name) for repo in sample_dict[clsname]])

        self.hist_data = np.transpose(np.array(hist_data))

    def prepare(self):
        """Return a Figure ready to show().

        Subclasses probably want to override this."""
        self.ax.set_title(self.feature_name)
        self.ax.hist(self.hist_data)

        return self.figure


class NumAllFilesChart(FeatureChart):
    # low clusters at low
    feature_name = 'num_all_files'

    def prepare(self):
        self.ax.set_title('Number of files')
        self.ax.hist(self.hist_data,
                     bins=range(int(np.percentile(self.hist_data, 20)),
                                int(np.percentile(self.hist_data, 80)),
                                5),
                     label=self.class_names,
                     normed=True
                     )

        self.ax.legend()

        return self.figure


class TotalFilesizeChart(FeatureChart):
    feature_name = 'size_all_files'

    def prepare(self):
        self.ax.set_title('Sum of filesizes')
        self.ax.hist(self.hist_data,
                     bins=range(int(np.percentile(self.hist_data, 25)),
                                int(np.percentile(self.hist_data, 75)),
                                2**14),
                     label=self.class_names,
                     )
        self.ax.legend()

        return self.figure


class EmptyReadmeChart(FeatureChart):
    # high more likely to have nonempty readme

    feature_name = 'readme_size'

    def prepare(self):
        self.ax.set_title('Ratio of repos with a nonempty README file')

        # ar(num_classes, num_repos)
        nonempty = np.apply_along_axis(lambda x: x > 0, 0, self.hist_data).transpose()

        width = .35

        bars = []
        for i, class_name in enumerate(self.class_names):
            bars.append(self.ax.bar(i*width,
                                    1.0 * np.sum(nonempty, axis=1)[i] / len(nonempty[i]),
                                    width,
                                    color=self.ax._get_lines.color_cycle.next(),
                                    ))

        self.ax.legend([bar[0] for bar in bars], self.class_names)

        return self.figure


class ReadmeSizeChart(FeatureChart):
    feature_name = 'readme_size'

    def prepare(self):
        self.ax.set_title('Size of README file')
        self.ax.hist(self.hist_data,
                     #bins=range(int(np.percentile(self.hist_data, 25)),
                     #           int(np.percentile(self.hist_data, 75)),
                     #           2**14),
                     label=self.class_names,
                     )
        self.ax.legend()

        return self.figure


class SourceRatioChart(FeatureChart):
    # high clustered at very low, low clustered at very high
    # in middle, high is usually greater

    feature_name = 'ratio_vol_src_files'

    def prepare(self):
        self.ax.set_title('Ratio of source to other files')
        self.ax.hist(self.hist_data,
                     bins=100,
                     label=self.class_names,
                     normed=True,
                     )
        self.ax.legend()

        return self.figure


if __name__ == '__main__':
    for c in [EmptyReadmeChart]:
        f = c().prepare()
        f.show()

    plt.draw()

    code.interact(local=locals())


### old
def process(sample_dict, class_names, feature_names):
    # sample_dict: {classname: [repos]}

    # eg 'ReadmeSize' -> histogram data (ie ar(num_repos_in_class, num_classes))
    feature_data = {}
    for feature_name in feature_names:
        data = []
        for clsname in class_names:
            data.append([getattr(repo, feature_name) for repo in sample_dict[clsname]])

        feature_data[feature_name] = np.transpose(np.array(data))

    for feature_name, hist_data in feature_data.items():
        P.figure()
        P.title(feature_name)
        P.hist(hist_data, bins=range(50, 500, 25), label=class_names)
        P.legend()
        P.show()

    #for feature_name, ar in feature_data.items():
    #    print '\t'.join(str(x) for x in (
    #        feature_name,
    #        np.amin(ar),
    #        np.amax(ar),
    #        np.median(ar),
    #        np.mean(ar),
    #        np.std(ar)
    #    ))
