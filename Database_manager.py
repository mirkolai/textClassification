from sklearn.externals import joblib
from Tweet import make_tweet
import os.path
import pymysql
import config as cfg

class Database_manager(object):

    db=None
    cur=None

    def __init__(self):
        """
         If you want to recover tweets from a mysql db set the config.py:
         for example  mysql = {
         'host': 'yourhost',
         'user': 'yourmysqluser',
         'passwd': 'yourpassword',
         'db': 'dbname'}
        """
        self.db = pymysql.connect(host=cfg.mysql['host'],
                 user=cfg.mysql['user'],
                 passwd=cfg.mysql['passwd'],
                 db=cfg.mysql['db'],
                 charset='utf8')
        self.cur = self.db.cursor()
        self.cur.execute('SET NAMES utf8mb4')
        self.cur.execute("SET CHARACTER SET utf8mb4")
        self.cur.execute("SET character_set_connection=utf8mb4")
        self.db.commit()


    def return_tweets(self):
        """Return an array containing tweets.
           Tweets are encoded as Tweet objects.
        """
        """
         You could recover tweets from db or csv file

        """
        tweets=[]
        self.cur.execute("SELECT `ID`, `Text`,`Label`  FROM `training` where target = 'Hillary Clinton'"
                         "UNION "
                         "SELECT `ID`, `Text`,`Label`  FROM `test` where target = 'Hillary Clinton'")

        for tweet in self.cur.fetchall():
                id=tweet[0]
                text=tweet[1]
                label=tweet[2]

                """
                Create a new istance of a Tweet object
                """
                this_tweet=make_tweet(id, text, label)

                tweets.append(this_tweet)


        return tweets

    def return_tweets_training(self):
        """Return an array containing tweets.
           Tweets are encoded as Tweet objects.
        """
        """
         You could recover tweets from db or csv file

        """
        tweets=[]
        self.cur.execute("SELECT `ID`, `Text`,`Label`  FROM `training` where target = 'Hillary Clinton'")

        for tweet in self.cur.fetchall():
                id=tweet[0]
                text=tweet[1]
                label=tweet[2]

                """
                Create a new istance of a Tweet object
                """
                this_tweet=make_tweet(id, text, label)

                tweets.append(this_tweet)


        return tweets


    def return_tweets_test(self):
        """Return an array containing tweets.
           Tweets are encoded as Tweet objects.
        """
        """
         You could recover tweets from db or csv file

        """
        tweets=[]
        self.cur.execute("SELECT `ID`, `Text`,`Label`  FROM `test` where target = 'Hillary Clinton'")

        for tweet in self.cur.fetchall():
                id=tweet[0]
                text=tweet[1]
                label=tweet[2]

                """
                Create a new istance of a Tweet object
                """
                this_tweet=make_tweet(id, text, label)

                tweets.append(this_tweet)


        return tweets




def make_database_manager():
    database_manager = Database_manager()

    return database_manager




