import datetime
import logging
import sys
import time

from github import Github

from models import Repo, Author

logging.basicConfig(filename='author_fetch.log', level=logging.DEBUG)

FILE = 'authors.msg'


class Elaborator(object):
    def __init__(self):
        self._authed_gh = Github('USER', 'PASS')
        self._noauth_gh = Github()
        self.gh = self._authed_gh

    def _wait_for_rate_limit(self):
        while self.gh.rate_limiting[0] == 0:
            logging.info('waiting on rate limit...')

            #try for the 60 unauthed requests, too
            if self.gh is self._noauth_gh:
                self.gh = self._authed_gh
            else:
                self.gh = self._noauth_gh

            self._gh_request(
                'GET',
                '/rate_limit'
            )
            time.sleep(60)

    def _gh_request(self, verb, url):
        #a hack to allow me to save requests
        req = getattr(self.gh, '_Github__requester')
        headers, data = req.requestAndCheck(verb, url, None, None)
        return data


def progress_bar(processed, total):
    pct_done = int(100.0 * processed / total)

    bar = '#' * (pct_done / 5)
    bar = bar.ljust(20)

    sys.stdout.write("\rfetching [{}] {}%".format(bar, pct_done))
    sys.stdout.flush()


def fetch():
    sys.stdout.write('loading')
    sys.stdout.flush()
    repos = Repo.load_sample()
    authors = {author.login: author for author in Author.load(FILE)}

    seen = 0
    total = len(repos)
    failures = []
    last_write = datetime.datetime.now()

    el = Elaborator()

    for repo in repos:
        seen += 1

        if repo.username in authors:
            logging.info("already fetched %s", repo.username)
            continue

        try:
            gh_data = el._gh_request(
                'GET',
                '/users/' + repo.username
            )
        except:
            #loop really needs to keep running
            logging.exception("problem! %s", repo)
            failures.append(repo)
            continue

        authors[repo.username] = Author(**{key: gh_data.get(key, None) for key in
                                           ['login',  # "octocat"
                                            'id',  # 1
                                            'avatar_url',  # "https://github.com/images/error/octocat_happy.gif"
                                            'gravatar_id',  # "somehexcode"
                                            'url',  # "https://api.github.com/users/octocat"
                                            'name',  # "monalisa octocat"
                                            'company',  # "GitHub"
                                            'blog',  # "https://github.com/blog"
                                            'location',  # "San Francisco"
                                            'email',  # "octocat@github.com"
                                            'hireable',  # false
                                            'bio',  # "There once was..."
                                            'public_repos',  # 2
                                            'public_gists',  # 1
                                            'followers',  # 20
                                            'following',  # 0
                                            'html_url',  # "https://github.com/octocat"
                                            'created_at',  # "2008-01-14T04:33:35Z"
                                            'type',  # "User"
                                            ]})

        logging.info("fetched %s", repo.username)

        progress_bar(seen, total)

        since_write = datetime.datetime.now() - last_write

        if since_write > datetime.timedelta(minutes=5):
            sys.stdout.write("\r(writing results)")
            sys.stdout.flush()
            Author.dump(authors.values(), FILE)

            last_write = datetime.datetime.now()

    print  # from progress bar line

    if failures:
        print "%s failures:" % len(failures)
        for f in failures:
            print "  %s" % f
        print

    print 'writing out...'
    Author.dump(authors.values(), FILE)


if __name__ == '__main__':
    fetch()
