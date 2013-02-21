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


#These decorators register a feature as normal/support.
def feature(f):
    all_features[f.__name__] = f
    return f


def support_feature(f):
    _support_features[f.__name__] = f
    return f

# non-source files
#features take a Repo and return their value.
#they assume curdir is the repo's directory


@support_feature
def all_file_sizes(repo):
    """A map of {filepath: filesize} for all files in the repo."""
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


@support_feature
def src_file_sizes(repo):
    return {f: size for (f, size) in repo._calc('all_file_sizes').iteritems()
            if f.endswith('.py')}


@feature
def num_all_files(repo):
    return len(repo._calc('all_file_sizes'))


@feature
def size_all_files(repo):
    """Total filesize of all files in the repo."""
    return sum(size for size in repo._calc('all_file_sizes').itervalues())


@feature
def size_src_files(repo):
    return sum(size for fname, size in repo._calc('src_file_sizes').iteritems())


@feature
def num_src_files(repo):
    return len(repo._calc('src_file_sizes'))


@feature
def ratio_src_files(repo):
    """.py files / all files"""
    return 100.0 * repo._calc('num_src_files ') / repo._calc('num_all_files')


@feature
def ratio_vol_src_files(repo):
    """size of .py files / size of all files"""
    return 100.0 * repo._calc('size_src_files ') / repo._calc('size_all_files')


@feature
def readme_size(repo):
    """Size of the readme, in bytes.

    0 can mean either non-existant or 0-length."""

    matching = glob('*README*')

    if not matching:
        return 0

    #take the longest filename; hack to emulate GitHub's preference to
    #eg README.md over README
    readme_fn = max(matching, key=len)

    return utils.filesize_or_zero(readme_fn)


@feature
def setuppy_size(repo):
    return utils.filesize_or_zero('setup.py')


@feature
def license_size(repo):
    #TODO consider eg LICENSE.txt
    return max(utils.filesize_or_zero(fn) for fn in ('LICENSE', 'COPYING'))


@feature
def travis_cfg_size(repo):
    return utils.filesize_or_zero('.travis.yml')


@feature
def contributing_size(repo):
    return utils.filesize_or_zero('CONTRIBUTING')


# Code features

@support_feature
def source_contents(repo):
    """A dict {filename: contents} for all source files."""
    contents = {}

    for py_file, size in repo._calc('src_file_sizes').iteritems():
        try:
            with open(py_file, 'rb') as f:
                contents[py_file] = f.read()
        except IOError:
            logging.exception("could not open %s/%s", repo._calc('name'), py_file)

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


@support_feature
def asts(repo):
    """A dict {filename: ast} for all .py files."""
    asts = {}
    for src_fn, src in repo._calc('source_contents').iteritems():
        try:
            ast = pyast.parse(src)
        except:
            #if their code does not compile, ignore it
            #TODO should probably be more strict against this,
            #could really throw off num_ast-relative features
            #maybe don't consider repos with non-compiling code?
            logging.exception("file %s/%s does not compile",
                              repo.name, src_fn)
        else:
            #otherwise, include it
            asts[src_fn] = ast

    return asts


@support_feature
def ast_node_counts(repo):
    """A counter over ast node names for all source."""
    counter = Counter()

    for ast in repo._calc('asts').itervalues():
        counter.update(node.__class__.__name__
                       for node in pyast.walk(ast))

    return counter


@feature
def num_ast_nodes(repo):
    """Total number of ast nodes; a measure of code volume."""
    nodes = sum(count for count in repo._calc('ast_node_counts').values())
    nodes += 1  # used as relative, don't want 0
    return nodes


#These features refer to usage of certain language features.
@feature
def with_stmt_usage(repo):
    nodes = repo._calc('ast_node_counts').get('With', 0)
    return 100.0 * nodes / repo._calc('num_ast_nodes')


@feature
def compr_usage(repo):
    nodes = repo._calc('ast_node_counts').get('comprehension', 0)
    return 100.0 * nodes / repo._calc('num_ast_nodes')


@feature
def lambda_usage(repo):
    nodes = repo._calc('ast_node_counts').get('Lambda', 0)
    return 100.0 * nodes / repo._calc('num_ast_nodes')


@feature
def global_usage(repo):
    nodes = repo._calc('ast_node_counts').get('Global', 0)
    return 100.0 * nodes / repo._calc('num_ast_nodes')


@feature
def gen_exp_usage(repo):
    nodes = repo._calc('ast_node_counts').get('GeneratorExp', 0)
    return 100.0 * nodes / repo._calc('num_ast_nodes')


@feature
def print_usage(repo):
    nodes = repo._calc('ast_node_counts').get('Print', 0)
    return 100.0 * nodes / repo._calc('num_ast_nodes')


@feature
def comment_ratio(repo):
    """Number of comments / code volume."""
    num = 0

    for src_fn, src in repo._calc('source_contents').iteritems():
        strbuf = StringIO(src)
        try:
            toks = tokenize.generate_tokens(strbuf.readline)
            num += len([t for t in toks if t[0] == tokenize.COMMENT])
        except:
            #similar to does not compile error
            logging.exception("file %s/%s does not tokenize",
                              repo.name, src_fn)

    #consider storing fractions and converting out later
    return 100.0 * num / repo._calc('num_ast_nodes')


@feature
def docstring_ratio(repo):
    """Percent of function/class/module definitions with docstrings."""
    def_nodes = 1  # avoid division by zero
    doc_def_nodes = 0

    for root in repo._calc('asts').itervalues():
        for node in pyast.walk(root):
            if isinstance(node, (pyast.FunctionDef, pyast.ClassDef,
                                 pyast.Module)):

                def_nodes += 1

                docstring = pyast.get_docstring(node)

                if docstring:
                    doc_def_nodes += 1

    return 100.0 * doc_def_nodes / def_nodes


@feature
def docstring_avg_len(repo):
    def_nodes = 1  # avoid division by zero
    docstring_len = 0

    for root in repo._calc('asts').itervalues():
        for node in pyast.walk(root):
            if isinstance(node, (pyast.FunctionDef, pyast.ClassDef,
                                 pyast.Module)):

                def_nodes += 1

                docstring = pyast.get_docstring(node)

                if docstring:
                    docstring_len += len(docstring)

    return 100.0 * docstring_len / def_nodes
