"""Various utilies used across the project."""

from contextlib import contextmanager
import logging
import os
import shutil
import tarfile
from tempfile import NamedTemporaryFile
import urllib  # I'm as surprised as you are, this was easiest

from config import config
#from sample import repos


class cd:
    def __init__(self, newPath):
        self.newPath = newPath

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def download(repo):
    """Return True if the repo is downloaded and ready for further processing."""

    logging.info("download: %s", repo)

    user, reponame = repo.name.split('/')

    result_path = os.path.join(config['current_snapshot'], 'code', user, reponame)
    if os.path.exists(result_path):
        # don't redownload
        logging.info("already downloaded")
        return True

    tarball_url = ("https://github.com/"
                   "{}/archive/{}.tar.gz").format(repo.name, repo.master_branch)

    tarball_fn = "{}_{}.tar.gz".format(user, reponame)
    inner_dir = "{}-{}".format(reponame, repo.master_branch)

    success = False
    attempts = 0

    while not success and attempts < 3:

        try:
            _, headers = urllib.urlretrieve(tarball_url, tarball_fn)

            logging.debug('extract')
            with tarfile.open(tarball_fn, 'r:gz') as tarball:
                tarball.extractall()

            logging.debug('rm tarball')
            os.remove(tarball_fn)

            logging.debug('add to tree')
            shutil.move(inner_dir, result_path)
        except:
            logging.exception('problem downloading')
            #try to clean up
            try:
                os.remove(tarball_fn)
                shutil.rmtree(inner_dir)
            except:
                logging.exception('problem cleaning up from failure')
        else:
            success = True
        finally:
            attempts += 1

    if not success:
        logging.error("could not download %s", repo)
    else:
        logging.info('downloaded')

    return success


def filesize_or_zero(filename):
    """If the file exists, return the size in bytes.
    Otherwise, return 0.
    """
    if os.path.exists(filename):  # race condition ignored
        return os.path.getsize(filename)
    else:
        return 0


#FaulTolerantFile recipe from:
#from: http://bit.ly/R2jZNp
if not hasattr(os, 'replace'):
    os.replace = os.rename


@contextmanager
def FaultTolerantFile(name):
    dirpath, filename = os.path.split(name)
    # use the same dir for os.rename() to work
    with NamedTemporaryFile(dir=dirpath, prefix=filename, suffix='.tmp') as f:
        yield f  # opened with mode w+b by default
        f.flush()   # libc -> OS
        os.fsync(f)  # OS -> disc (note: on OSX it is not enough)
        f.delete = False  # don't delete tmp file if `replace()` fails
        f.close()
        os.replace(f.name, name)


#def load_f_dicts(merge_new=True):
#    try:
#        with open(feature_pickle_name, 'rb') as f:
#            f_dicts = pickle.load(f)
#    except IOError:
#        logging.warning('unable to load f_dicts; using empty collection')
#        f_dicts = {user_repo: {} for user_repo in repos}
#
#    if merge_new:
#        for user_repo in repos:
#            if user_repo not in f_dicts:
#                f_dicts[user_repo] = {}
#
#    return f_dicts


#def persist_f_dicts(f_dicts):
#    with FaultTolerantFile(feature_pickle_name) as f:
#        pickle.dump(f_dicts, f, pickle.HIGHEST_PROTOCOL)


def create_bimap(els):
    """Return a tuple (forward, backwards) where each el in els is mapped as
        forward[el] = i
        backward[i] = el
        """
    forward = {el: i for
               el, i in zip(els, range(len(els)))}
    backward = dict(reversed(pair) for pair in forward.items())

    return (forward, backward)


#def get_pep8_errors(repo_path):
#    """Return the number of pep8 errors + warnings for some repo."""
#
#    style_checker = pep8.StyleGuide(quiet=True,
#                                    parse_argv=False,
#                                    config_file=False)
#
#    return style_checker.check_files([repo_path]).get_count()
