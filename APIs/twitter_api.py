import pandas as pd
import os
from personality_features import get_lang_based_scores, get_personality_scores
from tweepy import API, OAuthHandler, TweepError
from keys import twitter_access_token, twitter_access_token_secret, twitter_consumer_key, \
    twitter_consumer_secret
import warnings

warnings.filterwarnings("ignore")

auth = OAuthHandler(twitter_consumer_key, twitter_consumer_secret)
auth.set_access_token(twitter_access_token, twitter_access_token_secret)
api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

users = pd.read_csv('artists_twitter.csv', header=None)
users = users.drop(0, axis=1)
users = users.drop(0, axis=0)
users = users.rename(columns={1: 'artist_name', 2: 'twitter_name'})
users = users.reset_index(drop=True)

col_list = ['artist_name', 'followers_count', 'A', 'ANXIETY', 'AVOIDANCE', 'C', 'E', 'N', 'O']


def append_none_user(user):
    print('Skip: ', user['artist_name'])
    none_user = {'artist_name': 0, 'followers_count': 0, 'E': 0, 'AVOIDANCE': 0, 'C': 0, 'O': 0, 'N': 0, 'A': 0,
                 'ANXIETY': 0}
    none_user_df = pd.Series(none_user)
    none_user_df['artist_name'] = user['artist_name']
    return none_user_df


def write_artist2csv(df_pers):
    if not os.path.isfile('artist_personality.csv'):
        df_pers.to_csv('artist_personality.csv', header=True, index=False,
                       columns=col_list)
    else:
        df_pers.to_csv('artist_personality.csv', mode='a', header=False, index=False, columns=col_list)


# Get tweets from a particular user
def get_user_tweets(from_i, to_i, page_count, tweet_count, use_user_features):
    downloaded_tweets_count = 0

    for index in range(from_i, to_i):  # len(users)
        user = users.iloc[index]

        if user['twitter_name'] != 'none':
            try:
                print(index, "User: ", user['artist_name'])
                user_tweets = []
                for i in range(0, page_count):
                    statuses = api.user_timeline(screen_name=user['twitter_name'],
                                                 count=tweet_count, page=i, lang="en",
                                                 tweet_mode="extended")
                    for status in statuses:
                        status_dict = dict()
                        status_dict["user_id"] = status.user.id
                        status_dict["id"] = status.id
                        status_dict["text"] = status.full_text
                        status_dict["favorite_count"] = status.favorite_count
                        status_dict["retweet_count"] = status.retweet_count
                        status_dict["created_at"] = status.created_at
                        user_tweets.append(status_dict)

                if use_user_features:
                    # NOT USED
                    return False
                    '''
                    user_dict = dict()
                    user_info = []
                    user_dict['twitter_id'] = status.user.id
                    user_dict['name'] = status.user.name
                    user_dict['screen_name'] = status.user.screen_name
                    user_dict['followers_count'] = status.user.followers_count
                    user_dict['friends_count'] = status.user.friends_count
                    user_dict['listed_count'] = status.user.listed_count
                    user_dict['statuses_count'] = status.user.statuses_count
                    user_dict['favourites_count'] = status.user.favourites_count
                    user_dict['description'] = status.user.description
                    user_dict['created_at'] = status.user.created_at
                    user_dict['location'] = status.user.location
                    user_info.append(user_dict)
                    users_with_personality = get_personality_scores(pd.DataFrame(user_info), pd.DataFrame(user_tweets))
                    '''
                else:
                    downloaded_tweets_count += len(user_tweets)
                    print(downloaded_tweets_count)
                    if len(user_tweets) > 0:
                        users_with_personality = get_lang_based_scores(pd.DataFrame(user_tweets))
                        users_with_personality.insert(0, 'artist_name', user['artist_name'], True)
                        users_with_personality.insert(1, 'followers_count', status.user.followers_count, True)
                        users_with_personality.pop('user_id')
                        write_artist2csv(users_with_personality)
                    else:
                        write_artist2csv(pd.DataFrame(append_none_user(user)).transpose())

            except TweepError as err:
                print(err.args, print(err.args[0]))
                if err.args == ('Not authorized.',) or err.api_code == 34:
                    write_artist2csv(pd.DataFrame(append_none_user(user)).transpose())
                else:
                    print("Stopped! Error at index: ", index)
        else:
            write_artist2csv(pd.DataFrame(append_none_user(user)).transpose())


get_user_tweets(from_i=0, to_i=users.shape[0], page_count=10, tweet_count=20, use_user_features=False)
