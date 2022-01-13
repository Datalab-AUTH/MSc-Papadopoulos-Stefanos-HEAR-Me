import pymongo
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def mongo_to_csv(user_type=None):
    print('Connect to MongoDB')
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["hearm"]

    col_reviewers = db["reviewers_aoty"]
    results_reviewers = col_reviewers.find({})
    data = pd.DataFrame(list(results_reviewers))  # Data reviewers

    # To include Raters
    if user_type == 'UsersAll':
        col_raters = db["raters_aoty"]
        results_raters = col_raters.find({})
        data_raters = pd.DataFrame(list(results_raters))
        data_raters.insert(3, 'review', ' ')
        data_raters.insert(4, 'review anger', 0)
        data_raters.insert(5, 'review joy', 0)
        data_raters.insert(6, 'review love', 0)
        data_raters.insert(7, 'review sadness', 0)
        data_raters.insert(8, 'review surprise', 0)
        data_raters.insert(9, 'review sentiment', 0)
        data = pd.concat([data, data_raters], sort='False', ignore_index=True)

    data = data.rename({'user name': 'user_name',
                        'Album Name': 'album_name',
                        'review anger': 'r_anger',
                        'review joy': 'r_joy',
                        'review love': 'r_love',
                        'review sadness': 'r_sadness',
                        'review surprise': 'r_surprise',
                        'review sentiment': 'r_sentiment',
                        'Total Critic Rating': 'critic_t_rating',
                        'Total Audience Rating': 'audience_t_rating',
                        'Total Number of User Ratings': 'num_user_ratings',
                        'Artist Name': 'artist_name'
                        }, axis=1)

    data = data.drop_duplicates(subset=['user_name', 'album_name'], keep='first')
    data = data[data.rating != 'NR']
    data['rating'] = pd.to_numeric(data['rating'])

    data = data[[
        'user_name', 'album_name', 'artist_name', 'rating', 'Genres',
        'r_anger', 'r_joy', 'r_love', 'r_sadness', 'r_surprise', 'r_sentiment',
        'critic_t_rating', 'audience_t_rating', 'num_user_ratings', 'V_view_count',
        'L_relaxed_mood', 'L_happy_mood', 'L_sad_mood', 'L_angry_mood',
        'L_anger', 'L_fear', 'L_joy', 'L_sadness', 'L_surprise', 'L_love', 'L_sentiment',
        'M_valence', 'M_arousal',
        'V_anger', 'V_fear', 'V_joy', 'V_sadness', 'V_surprise', 'V_love', 'V_sentiment'
    ]]

    data_artist = pd.read_csv('../artist_personality.csv')
    data = pd.merge(data, data_artist, on='artist_name', how='left')
    del data_artist
    data = data.fillna(0)

    data['Genres'] = data['Genres'].apply(' / '.join)
    data['critic_t_rating'] = pd.to_numeric(data['critic_t_rating'])
    data['audience_t_rating'] = pd.to_numeric(data['audience_t_rating'])
    data['num_user_ratings'] = data['num_user_ratings'].str.replace(",", "")
    data['num_user_ratings'] = pd.to_numeric(data['num_user_ratings'])
    data[['rating',
          'critic_t_rating',
          'audience_t_rating',
          'num_user_ratings',
          'V_view_count',
          'followers_count', 'A', 'ANXIETY', 'AVOIDANCE', 'C', 'E', 'N', 'O',
          'r_sentiment', 'L_sentiment', 'V_sentiment'
          ]] = MinMaxScaler(feature_range=(0, 1)).fit_transform(data[['rating',
                                                                      'critic_t_rating',
                                                                      'audience_t_rating',
                                                                      'num_user_ratings',
                                                                      'V_view_count',
                                                                      'followers_count',
                                                                      'A', 'ANXIETY', 'AVOIDANCE',
                                                                      'C', 'E', 'N', 'O',
                                                                      'r_sentiment', 'L_sentiment', 'V_sentiment'
                                                                      ]])
    data = data.round(3)
    data.to_csv(user_type + '.csv', sep=';')
