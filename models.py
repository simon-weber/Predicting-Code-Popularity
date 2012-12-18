"""This file contains the schema for the abandoned elaborated-repo sqlite
database. It is still used when choosing a new random sample."""

from base64 import b64encode, b64decode
import cPickle as pickle

from peewee import SqliteDatabase, Model
from peewee import BooleanField, DateTimeField, IntegerField, TextField

erepo_db = SqliteDatabase('erepo.db', threadlocals=True)
erepo_db.connect()


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
