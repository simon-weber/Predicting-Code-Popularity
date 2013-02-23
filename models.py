"""This file contains the schema for the abandoned elaborated-repo sqlite
database. It is still used when choosing a new random sample."""

from base64 import b64encode, b64decode
import cPickle as pickle
import logging
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

    #TODO make these aware of a current snapshot/sample
    #load/dump are the user interface - they can handle all registered classes
    @classmethod
    def load(cls, filepath=None):
        """Load the contents of the given filepath.
        If None, assume '<current_snapshot>/repos.msgpack'"""

        if filepath is None:
            filepath = os.path.join(config['current_snapshot'], 'repos.msgpack')

        with open(filepath, 'rb') as f:
            records = msgpack.load(f, object_hook=cls._loader)

        return records

    @classmethod
    def dump(cls, records, filepath=None):
        """Dump in the same fashion as load."""

        if filepath is None:
            filepath = os.path.join(config['current_snapshot'], 'repos.msgpack')

        with utils.FaultTolerantFile(filepath, 'wb') as f:
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

    def _calc(self, feature):
        """Perform one-time calculation of a feature."""
        #even though __getattribute__ is cleaner, sometimes you do want the value
        #without calculating (eg when writing out)
        val = getattr(self, feature)

        if val is None:
            val = all_features[feature](self)
            setattr(self, feature, val)

        return val

    @property
    def username(self):
        self.name.split('/')[0]

    @property
    def reponame(self):
        self.name.split('/')[1]

    def calculate_features(self):
        """Change to this repo's directory, then calculate all its features."""
        #TODO make aware of current snapshot

        pass
        #old calculate:
        #def calculate(cls, user_repo, features):
        #    """Set our feature value on *features*."""
        #    #also factors out boilerplate from actual calculation

        #    logging.info('fcalc: %s(%s)', cls.__name__,  user_repo)

        #    try:
        #        repo_path = os.path.join(repo_dir, *user_repo.split('/'))

        #        if os.getcwd().endswith(repo_path):
        #            # we're already in the directory from another feature
        #            repo_path = '.'

        #        with utils.cd(repo_path):
        #            retval = cls._calculate(user_repo, features)
        #            cls._set_val(user_repo, features, retval)
        #            #logging.info('found: %s', retval)
        #    except:
        #        #logging.exception('exception during fcalc')
        #        raise

    @staticmethod
    def from_erepo(erepo):
        grepo_to_erepo = {'name': '_user_repo',
                          'stars': '_stars'}

        grepo_to_erepo.update({f: f for f in erepo.__dict__['_data']
                               if f not in ('_user_repo', '_stars', '_elaborated_at',
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
    _stars = IntegerField(index=True)

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
