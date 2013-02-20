"""This file contains the definition of the different features. It's not
terribly elegant, since features grew in complexity as the project went on.
Basically, each repo has a feature dictionary; it is passed to
Feature._calculate, which writes its result in under the key for its name.

There are also SupportOnlyFeatures, which are basically just used to memoize
intermediate work common to multiple features. They are not persisted, and are
calculated lazily."""

import ast as pyast
from collections import Counter
from glob import glob
import os
import logging
import tokenize
from StringIO import StringIO

from config import repo_dir
import utils


# {'feature name': Feature} for outside the module.
# serialization assumes that features will not be removed once added
all_features = {}

_support_features = {}


class _RegisterMeta(type):
    """This metaclass registers concrete features in all_features."""

    def __new__(cls, name, bases, dct):
        c = super(_RegisterMeta, cls).__new__(cls, name, bases, dct)

        # don't register abcs and support features
        base_names = [base.__name__ for base in bases]
        if 'Feature' in base_names and name != 'SupportOnlyFeature':
            all_features[name] = c
        elif 'SupportOnlyFeature' in base_names:
            _support_features[name] = c

        return c


class Feature(object):
    """ABC for a feature. Features define a calculation on a repo.

    Features are callable, and mutate a given f_dict to include their value.
    """
    __metaclass__ = _RegisterMeta

    calculated_on = None  # the last repo we have been calculated on

    @classmethod
    def _get_val(cls, user_repo, features):
        """Return the value for this Feature, calculating it if needed."""
        if cls.calculated_on != user_repo:
            cls.calculate(user_repo, features)

        return cls._pull_val(features)

    @classmethod
    def _pull_val(cls, features):
        return features[cls.__name__]

    @classmethod
    def _set_val(cls, user_repo, features, val):
        cls.calculated_on = user_repo
        cls._store_val(features, val)

    @classmethod
    def _store_val(cls, features, val):
        features[cls.__name__] = val

    @classmethod
    def calculate(cls, user_repo, features):
        """Set our feature value on *features*."""
        #also factors out boilerplate from actual calculation

        logging.info('fcalc: %s(%s)', cls.__name__,  user_repo)

        try:
            repo_path = os.path.join(repo_dir, *user_repo.split('/'))

            if os.getcwd().endswith(repo_path):
                # we're already in the directory from another feature
                repo_path = '.'

            with utils.cd(repo_path):
                retval = cls._calculate(user_repo, features)
                cls._set_val(user_repo, features, retval)
                #logging.info('found: %s', retval)
        except:
            #logging.exception('exception during fcalc')
            raise

    @classmethod
    def _calculate(cls, user_repo, features):
        """Perform the actual calculation of the Feature, possibly using other
        features to support our calculation."""
        raise NotImplementedError  # implemented by concrete Features


class SupportOnlyFeature(Feature):
    """ABC for a feature that only exists to support other features.

    It is not set in the feature dict for a repo.
    """
    val = None  # since we're not in the dict, store ourselves in our own class

    @classmethod
    def _pull_val(cls, _):
        return cls.val

    @classmethod
    def _store_val(cls, features, val):
        cls.val = val


# non-source files


class AllFileSizes(SupportOnlyFeature):
    """A map of {filepath: filesize} for all files in the repo."""
    @classmethod
    def _calculate(cls, user_repo, features):
        filesizes = {}

        for dirpath, dirnames, filenames in os.walk('.'):
            if '.git' in dirnames:
                dirnames.remove('.git')  # don't enter git db

            filepaths = (os.path.join(dirpath, fname) for fname in filenames)
            filepaths = (path for path in filepaths if
                         not os.path.islink(path))

            for f in filepaths:
                filesizes[f] = os.path.getsize(f)

        return filesizes


class NumAllFiles(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        return len(AllFileSizes._get_val(user_repo, features))


class AllFilesSize(Feature):
    """Total filesize of all files in the repo."""
    @classmethod
    def _calculate(cls, user_repo, features):
        return sum(size for size in
                   AllFileSizes._get_val(user_repo, features).itervalues())


class SrcFilesSize(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        return sum(size for fname, size in
                   AllFileSizes._get_val(user_repo, features).iteritems()
                   if fname.endswith('.py'))


class NumSrcFiles(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        return len(SourceFiles._get_val(user_repo, features))


class SrcFileRatio(Feature):
    """.py files / all files"""

    @classmethod
    def _calculate(cls, user_repo, features):
        return 100.0 * \
            NumSrcFiles._get_val(user_repo, features) / \
            NumAllFiles._get_val(user_repo, features)


class SrcFileVolRatio(Feature):
    """size of .py files / size of all files"""

    @classmethod
    def _calculate(cls, user_repo, features):
        return 100.0 * \
            SrcFilesSize._get_val(user_repo, features) / \
            AllFilesSize._get_val(user_repo, features)


class ReadmeSize(Feature):
    """Size of the readme, in bytes.

    0 can mean either non-existant or 0-length."""

    @classmethod
    def _calculate(cls, user_repo, features):
        matching = glob('*README*')

        if not matching:
            return 0

        #take the longest filename; hack to emulate GitHub's preference to
        #README.md over README
        readme_fn = max(matching, key=len)

        return utils.filesize_or_zero(readme_fn)


class SetupSize(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        return utils.filesize_or_zero('setup.py')


class LicenseSize(Feature):
    #TODO consider eg LICENSE.txt
    @classmethod
    def _calculate(cls, user_repo, features):
        return max(utils.filesize_or_zero(fn) for fn in ('LICENSE', 'COPYING'))


class TravisCfgSize(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        return utils.filesize_or_zero('.travis.yml')


class ContributingSize(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        return utils.filesize_or_zero('CONTRIBUTING')


# Code features


class SourceFiles(SupportOnlyFeature):
    """A list of all .py files in the repo."""
    @classmethod
    def _calculate(cls, user_repo, features):
        return [f for f in
                AllFileSizes._get_val(user_repo, features).iterkeys()
                if f.endswith('.py')]


class SourceContents(SupportOnlyFeature):
    """A dict {filename: contents} for all source files."""

    @classmethod
    def _calculate(cls, user_repo, features):
        contents = {}

        for py_file in SourceFiles._get_val(user_repo, features):
            try:
                with open(py_file, 'rb') as f:
                    contents[py_file] = f.read()
            except IOError:
                logging.exception("could not open %s/%s", user_repo, py_file)

        return contents


# too costly to calculate for large samples
#class Pep8Compliance(Feature):
#    """pep8 errors and warnings / ast nodes."""
#
#    @classmethod
#    def _calculate(cls, user_repo, features):
#        errors = utils.get_pep8_errors('.')
#
#        return 100.0 * errors / NumAstNodes._get_val(user_repo, features)


class Asts(SupportOnlyFeature):
    """A dict {filename: ast} for all .py files."""

    @classmethod
    def _calculate(cls, user_repo, features):
        asts = {}
        for src_fn, src in SourceContents._get_val(user_repo,
                                                   features).iteritems():
            try:
                ast = pyast.parse(src)
            except:
                #if their code does not compile, ignore it
                #TODO should probably be more strict against this,
                #could really throw off num_ast-relative features
                #maybe don't consider repos with non-compiling code?
                logging.exception("file %s/%s does not compile",
                                  user_repo, src_fn)
            else:
                #otherwise, include it
                asts[src_fn] = ast

        return asts


class AstNodeCounts(SupportOnlyFeature):
    """The counts of all ast nodes across all source."""

    @classmethod
    def _calculate(cls, user_repo, features):
        counter = Counter()

        for ast in Asts._get_val(user_repo, features).itervalues():
            counter.update(node.__class__.__name__
                           for node in pyast.walk(ast))

        return counter


class NumAstNodes(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        nodes = sum(count for count in
                    AstNodeCounts._get_val(user_repo, features).values())
        nodes += 1  # used as relative, don't want 0
        return nodes


class WithUsage(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        nodes = AstNodeCounts._get_val(user_repo, features).get('With', 0)
        return 100.0 * nodes / NumAstNodes._get_val(user_repo, features)


class CompUsage(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        nodes = AstNodeCounts._get_val(user_repo, features).get(
            'comprehension', 0)
        return 100.0 * nodes / NumAstNodes._get_val(user_repo, features)


class LambdaUsage(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        nodes = AstNodeCounts._get_val(user_repo, features).get('Lambda', 0)
        return 100.0 * nodes / NumAstNodes._get_val(user_repo, features)


class GlobalUsage(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        nodes = AstNodeCounts._get_val(user_repo, features).get('Global', 0)
        return 100.0 * nodes / NumAstNodes._get_val(user_repo, features)


class GenUsage(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        nodes = AstNodeCounts._get_val(user_repo, features).get(
            'GeneratorExp', 0)
        return 100.0 * nodes / NumAstNodes._get_val(user_repo, features)


class PrintUsage(Feature):
    @classmethod
    def _calculate(cls, user_repo, features):
        nodes = AstNodeCounts._get_val(user_repo, features).get('Print', 0)
        return 100.0 * nodes / NumAstNodes._get_val(user_repo, features)


class RelNumComments(Feature):
    """Number of comments / code volume."""

    @classmethod
    def _calculate(cls, user_repo, features):
        num = 0

        for src_fn, src in SourceContents._get_val(user_repo,
                                                   features).iteritems():
            strbuf = StringIO(src)
            try:
                toks = tokenize.generate_tokens(strbuf.readline)
                num += len([t for t in toks if t[0] == tokenize.COMMENT])
            except:
                #similar to does not compile error
                logging.exception("file %s/%s does not tokenize",
                                  user_repo, src_fn)

        #consider storing fractions and converting out later
        return 100.0 * num / NumAstNodes._get_val(user_repo, features)


class DocStringUsage(Feature):
    """Percent of definitions with docstrings."""

    @classmethod
    def _calculate(cls, user_repo, features):
        def_nodes = 1  # avoid division by zero
        doc_def_nodes = 0

        for root in Asts._get_val(user_repo, features).itervalues():
            for node in pyast.walk(root):
                if isinstance(node, (pyast.FunctionDef, pyast.ClassDef,
                                     pyast.Module)):

                    def_nodes += 1

                    docstring = pyast.get_docstring(node)

                    if docstring:
                        doc_def_nodes += 1

        return 100.0 * doc_def_nodes / def_nodes


class DocStringVolume(Feature):
    """Sum of docstring lengths / def nodes"""

    @classmethod
    def _calculate(cls, user_repo, features):
        def_nodes = 1  # avoid division by zero
        docstring_len = 0

        for root in Asts._get_val(user_repo, features).itervalues():
            for node in pyast.walk(root):
                if isinstance(node, (pyast.FunctionDef, pyast.ClassDef,
                                     pyast.Module)):

                    def_nodes += 1

                    docstring = pyast.get_docstring(node)

                    if docstring:
                        docstring_len += len(docstring)

        return 100.0 * docstring_len / def_nodes
