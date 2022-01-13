import tweepy
import csv
import os
import pandas as pd
from utilities.keys import twitter_access, twitter_access_secret, twitter_api_key, twitter_api_key_secret

# Consumer keys and access tokens, used for OAuth
consumer_key = twitter_api_key
consumer_secret = twitter_api_key_secret
access_token = twitter_access
access_token_secret = twitter_access_secret

# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)


api = tweepy.API(auth)
all_users = pd.read_csv('artists.csv', sep='\n', header=None)
all_users.columns = ['artist_name']

screen_names = []

for i in range(1633, len(all_users)):
    user_name = all_users['artist_name'][i]
    user = api.search_users(user_name, 5)

    for u in range(len(user.ids())):
        if user[u].verified:
            print(i, user_name, user[u].screen_name)
            screen_names.append([user_name, user[u].screen_name])
            break
    else:
        print(i, user_name, 'none')
        screen_names.append([user_name, 'none'])


df = pd.DataFrame(screen_names)
df.columns = ['artist_name', 'artist_screen_name']

df.to_csv('artists_twitter.csv')

# df1 = pd.read_csv('artists_twitter.csv', header=None)
# df1.drop('Unnamed: 0', axis=1)