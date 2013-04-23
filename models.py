"""This file contains the schema for the abandoned elaborated-repo sqlite
database. It is still used when choosing a new random sample."""

from base64 import b64encode, b64decode
import cPickle as pickle
import datetime
import json
import os

import msgpack
from peewee import SqliteDatabase, Model
from peewee import BooleanField, DateTimeField, IntegerField, TextField
from recordtype import recordtype

from config import config
from features import all_features as real_features
from features import _support_features
import utils

all_features = dict(real_features.items() + _support_features.items())

erepo_db = SqliteDatabase('erepo.db', threadlocals=True)
erepo_db.connect()


class _Serializable(object):
    """Mixin to support serialization of a custom class.

    By default, recordtype._asdict is used."""

    __slots__ = ()

    @classmethod
    def _pack(cls, obj):
        return obj._asdict()

    @classmethod
    def _unpack(cls, data):
        return cls(**data)


class _MsgpackMeta(type):
    """Set on a class to enable serialization with msgpack.

    _Serializable becomes a base, so classes can override _un/pack.

    Note that msgpack encodes Unicodes to utf8."""

    def __new__(cls, name, bases, dct):
        #Insert our methods.
        dct['load'] = cls.load
        dct['dump'] = cls.dump
        bases = bases + (_Serializable,)

        c = super(_MsgpackMeta, cls).__new__(cls, name, bases, dct)

        #Subclasses get registered so we know how to pack/unpack them.
        classes = getattr(_MsgpackMeta, '_reg_classes', set())
        classes.add(c)

        _MsgpackMeta._names = {"%s" % reg.__name__: reg for reg in classes}
        _MsgpackMeta._reg_classes = classes

        return c

    #load/dump are the user interface - they can handle all registered classes
    @classmethod
    def load(cls, filepath=None):
        """Load the contents of the given filepath.
        If None, assume '<current_snapshot>/repos.msgpack'"""

        if filepath is None:
            filepath = os.path.join(config['current_snapshot'], 'repos.msgpack')

        with open(filepath, 'rb') as f:
            records = msgpack.load(f, object_hook=cls._loader, use_list=False)

        return records

    @classmethod
    def dump(cls, records, filepath=None):
        """Dump in the same fashion as load."""

        if filepath is None:
            filepath = os.path.join(config['current_snapshot'], 'repos.msgpack')

        with utils.FaultTolerantFile(filepath) as f:
            msgpack.dump(records, f, default=cls._dumper)

    #behind the scenes, _loader and _dumper do the work
    @classmethod
    def _loader(cls, obj):
        reg_class = cls._names.get(obj.get('__cls__'))
        if reg_class:
            return reg_class._unpack(obj['data'])

        return obj

    @classmethod
    def _dumper(cls, obj):
        if obj.__class__ in cls._reg_classes:
            reg_cls = obj.__class__
            return {
                "__cls__": reg_cls.__name__,
                'data': reg_cls._pack(obj)
            }

        return obj


_YMD = recordtype(
    'YMD',
    'year month day'
)


class YMD(_YMD):
    """Same purpose as datetime.date, but small and serializable."""

    __metaclass__ = _MsgpackMeta

    @staticmethod
    def from_date(date):
        """Factory from datetime.date or datetime.datetime."""
        return YMD(date.year, date.month, date.day)


_Repo = recordtype('Repo', (
    ['name',  # in 'user/repo' format
     'stars',
     'fetch_ymd',  # YMD of data acquisition
     ] +

    # these are all GitHub apiv3 names:
    ['clone_url',
     'created_at',
     'description',
     'fork',
     'forks',
     'git_url',
     'has_downloads',
     'has_issues',
     'has_wiki',
     'homepage',
     'html_url',
     'id',
     'language',
     'master_branch',
     'open_issues',
     'private',
     'pushed_at',
     'size',
     'ssh_url',
     'svn_url',
     'updated_at',
     'url',
     ] +
    [(fname, None) for fname in all_features]
))


class Repo(_Repo):
    """A repo stores a snapshot of GitHub repo metadata retrieved from
    `http://developer.github.com/v3/repos/#get` on some date, and any features
    calculated on that repo's code/metadata.
    """
    __metaclass__ = _MsgpackMeta

    def __str__(self):
        return self.name.encode('utf-8')

    @classmethod
    def _pack(cls, obj):
        """Don't write out support features."""
        d = {k: v for (k, v) in obj._asdict().iteritems()
             if k not in _support_features}

        return d

    def _calc(self, feature_name, overwrite=False):
        """Perform one-time calculation of a feature."""
        #even though __getattribute__ is cleaner, sometimes you do want the value
        #without calculating (eg when writing out)

        if feature_name not in all_features:
            raise ValueError("%s is not a valid feature name" % feature_name)

        val = getattr(self, feature_name)

        if val is None or overwrite:
            val = all_features[feature_name](self)
            setattr(self, feature_name, val)

        return val

    def _clear_support_features(self):
        """Set all support features to None.
        This can be used during calculation to limit memory use."""
        for k in _support_features:
            setattr(self, k, None)

    @property
    def username(self):
        return self.name.split('/')[0]

    @property
    def reponame(self):
        return self.name.split('/')[1]

    @property
    def creation_date(self):
        return datetime.datetime(*[int(x) for x in self.created_at[:10].split('-')])

    def calculate_features(self, features=None, overwrite=False):
        """Change to this repo's directory in the current snapshot,
        then calculate the given features.
        If features is None, calculate all features."""

        if features is None:
            features = all_features.keys()

        code_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            config['current_snapshot'], 'code', self.username, self.reponame
        )

        with utils.cd(code_dir):
            for f in features:
                self._calc(f, overwrite)

    @classmethod
    def load_sample(cls, sample_path=None, separate=False):
        """Load only repos in the given sample.
        If sample_path is None, assume '<current_snapshot>/<current_sample>'.

        If separate is true, return a dict mapping class name to repo lists."""

        repos = cls.load()

        # memoize for probable write_update
        cls._last_loaded = repos

        if sample_path is None:
            sample_path = os.path.join(config['current_snapshot'], config['current_sample'])

        with open(sample_path, 'rb') as f:
            separated_names = json.load(f)
            assert isinstance(separated_names, dict), 'did you try to load an old flat sample?'

        if separate:
            return {clsname: [r for r in repos if r.name in set(names)]
                    for (clsname, names) in separated_names.items()}
        else:
            nameset = set()
            for names in separated_names.values():
                nameset.update(set(names))

            return [r for r in repos if r.name in nameset]

    @classmethod
    def write_update(cls, records, filepath=None):
        """Like dump, but updates repos with a duplicate name."""
        loaded = getattr(cls, '_last_loaded', None)
        if loaded is None:
            loaded = cls.load(filepath)

        cur_repos = {r.name: r for r in loaded}
        new_repos = {r.name: r for r in records}

        cur_repos.update(new_repos)

        cls.dump(cur_repos.values(), filepath)
        cls._last_loaded = None

    @staticmethod
    def from_erepo(erepo):
        grepo_to_erepo = {'name': '_user_repo',
                          'stars': 'watchers'}

        grepo_to_erepo.update({f: f for f in erepo.__dict__['_data']
                               if f not in ('_user_repo', '_stars', 'watchers', '_elaborated_at',
                                            '_elaborated', '_error', '_features',
                                            '_flagged')})

        kwargs = {g_f: getattr(erepo, e_f) for (g_f, e_f) in grepo_to_erepo.items()}
        kwargs['fetch_ymd'] = YMD.from_date(erepo._elaborated_at)

        return Repo(**kwargs)


class ERepoModel(Model):
    class Meta:
        database = erepo_db


class ERepo(ERepoModel):
    #start initially populated fields
    #_form to avoid name collisions with github fields
    _user_repo = TextField(primary_key=True)  # eg simon/awesome-repo.
    _stars = IntegerField(null=True)  # no longer used as of 2013/03, use watchers instead

    _elaborated = BooleanField(default=False)  # ie 'has been processed'
    _error = BooleanField(default=False)  # eg download failed

    #base64 encoded python pickle of a dict
    _features = TextField(default='KGRwMAou')  # default=b64(pickle.dumps({}))
    _flagged = BooleanField(default=False)  # ie 'need to process'
    #end init populated fields
    #all others are set at elaboration-time

    _elaborated_at = DateTimeField(null=True)
    #start github api3 names
    clone_url = TextField(null=True)
    created_at = DateTimeField(null=True)
    description = TextField(null=True)
    fork = BooleanField(null=True)
    forks = IntegerField(null=True)
    git_url = TextField(null=True)
    has_downloads = BooleanField(null=True)
    has_issues = BooleanField(null=True)
    has_wiki = BooleanField(null=True)
    homepage = TextField(null=True)
    html_url = TextField(null=True)
    id = IntegerField(null=True)
    language = TextField(null=True)
    master_branch = TextField(null=True)
    open_issues = IntegerField(null=True)
    private = BooleanField(null=True)
    pushed_at = DateTimeField(null=True)
    size = IntegerField(null=True)
    ssh_url = TextField(null=True)
    svn_url = TextField(null=True)
    updated_at = DateTimeField(null=True)
    url = TextField(null=True)

    watchers = IntegerField(null=True)  # added before 2013/03 elaboration

    #to handle feature encoding/decoding.
    def get_features(self):
        return pickle.loads(b64decode(self._features))

    def set_features(self, val):
        self._features = b64encode(pickle.dumps(val))

    def get_vis(self):
        """Return a pretty-printed string to visualize the data."""
        lines = []
        lines.append("%s" % self._user_repo)
        for k in sorted(self._data.iterkeys()):
            lines.append("  %s: %s" % (k, self._data[k]))

        res = '\n'.join(lines)
        return res.encode('utf-8')

    def __repr__(self):
        quoted = "'%s'" % self._user_repo
        return quoted.encode('utf-8')
