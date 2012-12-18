"""The script used to gather repo metadata from the GitHub api."""

import datetime
import logging
import time

from github import Github, GithubException

from config import gh_user, gh_pass
from models import ERepo

logging.basicConfig(filename='erepo_work.log', level=logging.INFO)


class Elaborator(object):
    def __init__(self):
        self._authed_gh = Github(gh_user, gh_pass)
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

    def _next_erepo(self):
        try:
            erepo = ERepo.get(ERepo._elaborated == False)
        except ERepo.DoesNotExist:
            erepo = None

        return erepo

    def _gh_request(self, verb, url):
        #a hack to allow me to save requests
        req = getattr(self.gh, '_Github__requester')
        headers, data = req.requestAndCheck(verb, url, None, None)
        return data

    def run(self):
        erepo = self._next_erepo()

        while erepo:
            try:
                logging.info('elaborating:%s', erepo)

                self._wait_for_rate_limit()

                #I assume nobody puts a / in their name
                assert len(erepo._user_repo.split('/')) == 2
                user, repo_name = erepo._user_repo.split('/')

                gh_data = self._gh_request(
                    'GET',
                    '/repos/' + user + '/' + repo_name
                )

                #another hack: gh_data contains non-model fields,
                # but .save() only keeps model fields.
                for k, v in gh_data.items():
                    setattr(erepo, k, v)

                erepo._elaborated = True
                erepo._elaborated_at = datetime.datetime.now()

            except GithubException as gh_e:
                logging.warning("gherror(%s): %s", gh_e, erepo)
                logging.warning("delete: %s", erepo)
                erepo.delete_instance()
                erepo = None

            except:
                #hack hat trick!
                #but officer, this loop really needs to run at all costs
                logging.exception("%s", erepo)

                erepo._elaborated = False
                erepo._error = True

            finally:
                if erepo: erepo.save()  # may have deleted it
                erepo = self._next_erepo()
                time.sleep(3)  # out of courtesy


if __name__ == '__main__':
    e = Elaborator()
    e.run()
