"""This file contains the schema for the abandoned elaborated-repo sqlite
database. It is still used when choosing a new random sample."""

from base64 import b64encode, b64decode
import collections
import cPickle as pickle

import msgpack
from peewee import SqliteDatabase, Model
from peewee import BooleanField, DateTimeField, IntegerField, TextField
from recordtype import recordtype

from features import all_features

erepo_db = SqliteDatabase('erepo.db', threadlocals=True)
erepo_db.connect()


class _SerializableAsDictMeta(type):
    """Set as metaclass to use type()._asdict as serialization data.
    
    This keeps a record of classes for which it is the metaclass, and 
    exposes `load` and `dump` which can serialize all of them."""

    def __new__(cls, name, bases, dct):
        dct['load'] = cls.load
        dct['dump'] = cls.dump
        c = super(_SerializableAsDictMeta, cls).__new__(cls, name, bases, dct)

        #Subclasses get registered so we know how to pack/unpack them.
        classes = getattr(cls, '_reg_classes', set())
        classes.add(c)

        cls._names = {"%s" % reg.__name__: reg for reg in classes}
        cls._reg_classes = classes

        return c


    #TODO make these aware of a current snapshot/sample
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            records = msgpack.load(f, object_hook=cls._loader)

        return records

    @classmethod
    def dump(cls, records, filepath):
        with open(filepath, 'wb') as f:
            msgpack.dump(records, f, default=cls._dumper)

    @classmethod
    def _loader(cls, obj):
        reg_class = cls._names.get(obj.get('__cls__'))
        if reg_class:
            return reg_class(**obj['as_dict'])

        return obj

    @classmethod
    def _dumper(cls, obj):
        if obj.__class__ in cls._reg_classes:
            return {
                "__cls__": obj.__class__.__name__,
                'as_dict': obj._asdict()
            }

        return obj


_FRepo = recordtype(
    'FRepo',
    ['name'] + [fname for fname in all_features],
    default=None  # None signals an uncalculated feature
)


class FRepo(_FRepo):
    """A _F_eature repo stores calculated features for some repo."""
    __metaclass__ = _SerializableAsDictMeta


_YMD = recordtype(
    'YMD',
    'year month day'
)

class YMD(_YMD):
    """Same purpose as datetime.date, but small and serializable."""

    __metaclass__ = _SerializableAsDictMeta

    @staticmethod
    def from_date(date):
        """Factory from datetime.date or datetime.datetime."""
        return YMD(date.year, date.month, date.day)


_GRepo = recordtype('GRepo',
                    ['name',  # in 'user/repo' format
                     'stars',
                     'fetch_ymd',  # YMD of data acquisition
                     # these are all GitHub apiv3 names:
                     'clone_url',
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
                     ])


class GRepo(_GRepo):
    """A _G_itHub repo stores a snapshot of GitHub repo metadata retrieved from
    `http://developer.github.com/v3/repos/#get` on some date.
    
    Note that strings will be encoded as utf8 after a dump/load cycle."""

    __metaclass__ = _SerializableAsDictMeta

    def __str__(self):
        return self.name.encode('utf-8')

    @property
    def username(self):
        self.name.split('/')[0]

    @property
    def reponame(self):
        self.name.split('/')[1]

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

        return GRepo(**kwargs)


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
