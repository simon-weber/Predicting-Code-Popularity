import code

import numpy as np
import matplotlib.pyplot as plt

from classes import classes
from models import Repo

_sample = None  # global memoized Repo.load_sample(separate=True)


class FeatureChart(object):
    feature_name = None  # provide in subclass
    in_class_percentiles = (5, 95)  # set to None if not desired

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
            # ie 'class feature data'
            cfd = np.array([getattr(repo, self.feature_name) for repo in sample_dict[clsname]])

            if self.in_class_percentiles is not None:
                min_val, max_val = [np.percentile(cfd, i) for i in self.in_class_percentiles]
                cfd = np.array([e for e in cfd if min_val < e < max_val])

            hist_data.append(cfd)

        self.hist_data = hist_data

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
        self.ax.set_title('Number of files (low end)')
        self.ax.hist(self.hist_data,
                     bins=100,
                     range=(0, 100),  # long tail of high
                     label=self.class_names,
                     normed=True,
                     histtype='step',
                     fill=False,
                     )

        self.ax.legend()

        return self.figure


class TotalFilesizeChartBottom(FeatureChart):
    #TODO
    feature_name = 'size_all_files'
    in_class_percentiles = None

    def prepare(self):
        self.ax.set_title('Sum of filesizes (low end)')
        self.ax.hist(self.hist_data,
                     bins=100,
                     #range=(0, 2**16),
                     label=self.class_names,
                     normed=False,
                     histtype='step',
                     #fill=True,
                     )
        self.ax.legend()

        return self.figure


class EmptyReadmeChart(FeatureChart):
    # high more likely to have nonempty readme

    feature_name = 'readme_size'

    def prepare(self):
        self.ax.set_title('Ratio of repos with a nonempty README file')

        # jagged ar(num_classes, num_repos)
        nonempty = np.array([np.array([e > 0 for e in row]) for row in self.hist_data])

        width = .35

        bars = []
        for i, class_name in enumerate(self.class_names):
            bars.append(self.ax.bar(i*width,
                                    1.0 * np.sum(nonempty[i], axis=0) / len(nonempty[i]),
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
                     bins=100,
                     label=self.class_names,
                     normed=False,
                     histtype='step',
                     )
        self.ax.legend()

        return self.figure


class SourceRatioChart(FeatureChart):
    # high clustered at very low, low clustered at very high
    # in middle, high is usually greater

    feature_name = 'ratio_src_files'
    in_class_percentiles = None

    def prepare(self):
        self.ax.set_title('Ratio of source to other files')
        self.ax.hist(self.hist_data,
                     bins=100,
                     label=self.class_names,
                     normed=True,
                     histtype='step',
                     )
        self.ax.legend()

        return self.figure


class CommentRatioChart(FeatureChart):
    feature_name = 'comment_ratio'

    def prepare(self):
        self.ax.set_title('Occurence of comments')
        self.ax.hist(self.hist_data,
                     bins=100,
                     label=self.class_names,
                     histtype='step',
                     )
        self.ax.legend()

        return self.figure


class GlobalUsageChart(FeatureChart):
    feature_name = 'global_usage'

    def prepare(self):
        self.ax.set_title('Global variable usage (percent of AST nodes)')
        self.ax.hist(self.hist_data,
                     bins=100,
                     label=self.class_names,
                     histtype='stepfilled',
                     alpha=.5,
                     )
        self.ax.legend()

        return self.figure


class GlobalUsageChart(FeatureChart):
    feature_name = 'global_usage'

    def prepare(self):
        self.ax.set_title('Global variable usage (percent of AST nodes)')
        self.ax.hist(self.hist_data,
                     bins=100,
                     label=self.class_names,
                     histtype='stepfilled',
                     alpha=.5,
                     )
        self.ax.legend()

        return self.figure


if __name__ == '__main__':
    for c in [
        #NumAllFilesChart,
        #TotalFilesizeChartBottom,
        #EmptyReadmeChart,
        #ReadmeSizeChart,
        #SourceRatioChart,
        #CommentRatioChart,
        GlobalUsageChart,
    ]:
        f = c().prepare()
        f.show()

    plt.draw()

    code.interact(local=locals())


### old
#def process(sample_dict, class_names, feature_names):
#    # sample_dict: {classname: [repos]}
#
#    # eg 'ReadmeSize' -> histogram data (ie ar(num_repos_in_class, num_classes))
#    feature_data = {}
#    for feature_name in feature_names:
#        data = []
#        for clsname in class_names:
#            data.append([getattr(repo, feature_name) for repo in sample_dict[clsname]])
#
#        feature_data[feature_name] = np.transpose(np.array(data))
#
#    for feature_name, hist_data in feature_data.items():
#        P.figure()
#        P.title(feature_name)
#        P.hist(hist_data, bins=range(50, 500, 25), label=class_names)
#        P.legend()
#        P.show()

    #for feature_name, ar in feature_data.items():
    #    print '\t'.join(str(x) for x in (
    #        feature_name,
    #        np.amin(ar),
    #        np.amax(ar),
    #        np.median(ar),
    #        np.mean(ar),
    #        np.std(ar)
    #    ))
