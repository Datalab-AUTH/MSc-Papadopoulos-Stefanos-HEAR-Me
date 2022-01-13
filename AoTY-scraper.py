from bs4 import BeautifulSoup
import requests
import time
import pickle
import azapi
from langdetect import detect
from APIs.spotify_api import get_track_valence_arousal
import spotipy
from utilities.keys import spotify_cid, spotify_secret, custom_api, google_api  # , youtube_key
from spotipy.oauth2 import SpotifyClientCredentials
from APIs.youtube_api import search_track
from googleapiclient.discovery import build
from Emotion.EmotionAnalysis import predict_emotion
from Mood.MoodDetection import predict_lyric_mood
from lyrics_extractor import SongLyrics
from utilities.keys import yt_keys
import pymongo

cid = spotify_cid
secret = spotify_secret
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

extract_lyrics = SongLyrics(custom_api, google_api)  # Google Custom Search Engine for Lyrics Scraping

# Scrapes an individual user review
def users_review(user):
    user_name = user.find(itemprop='name').contents[0]
    user_album_rating = user.find(itemprop='ratingValue').contents[0]

    user_review = user.find(class_='albumReviewText user')

    # If not : the whole text is not visible and the 'view more' page must be scrapped
    if user_review.find('a') is None:
        user_review_text = user_review.contents[0]
        if not user_review_text == []:
            try:
                if detect(user_review_text) == 'en':
                    user_emotion = predict_emotion([user_review_text], False)
                else:
                    return 'not english'
            except:
                return 'not english'
        else:
            return 'empty'
    else:
        full_review_link = 'https://www.albumoftheyear.org' + \
                           user.find(class_='albumReviewText user').find('a').get('href')
        user_full_review_link = requests.get(full_review_link)
        user_review_soup = BeautifulSoup(user_full_review_link.content, 'html.parser')
        user_review_text = user_review_soup.find(class_='userReviewText').contents[0]
        if not user_review_text == []:
            try:
                if detect(user_review_text) == 'en':
                    user_emotion = predict_emotion([user_review_text], False)
                else:
                    return 'not english'
            except:
                return 'not english'
        else:
            return 'empty'

    user_dict = {"user name": user_name,
                 "rating": user_album_rating,
                 "review": user_review_text,
                 "review anger": user_emotion['anger'],
                 "review joy": user_emotion['joy'],
                 "review love": user_emotion['love'],
                 "review sadness": user_emotion['sadness'],
                 "review surprise": user_emotion['surprise'],
                 "review sentiment": user_emotion['sentiment']
                 }
    return user_dict


# Scrapes information and user review/ratings of an album
def album_review(album):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["hearm"]

    col_album = db["albums_aoty"]
    col_reviewers = db["reviewers_aoty"]
    col_raters = db["raters_aoty"]

    album_res = album.find_all('a')
    album_date = album.find(class_='albumListDate').contents[0]

    link = 'https://www.albumoftheyear.org' + album_res[0].get('href')

    # ----------------------------- ALBUM PAGE -----------------------------

    album_page = requests.get(link)
    soup_album = BeautifulSoup(album_page.content, 'html.parser')
    album_results = soup_album.find(id='centerContent')

    artist_name = soup_album.find(class_='artist').find(itemprop='name').contents[0]
    album_title = soup_album.find(class_='albumTitle').find(itemprop='name').contents[0]
    album_details = album_results.find(class_='albumTopBox info')

    print(' ### Scraping Artist : ', artist_name, ' ### ')
    print(' ### Album Title : ', album_title, ' ### ')

    ad_items = album_details.find_all(itemprop="genre")
    album_genres = []
    for ad_item in ad_items:
        album_genres.append(ad_item.contents[0])

    album_track_list = album_results.find(class_='trackList').contents[0]
    album_track_list = [a.text.strip() for a in album_track_list.find_all('li')]  # Remove html tags

    try:
        total_critic_rating = soup_album.find(class_='albumCriticScore').find('a').contents[0]
    except:
        total_critic_rating = 0
    try:
        total_user_rating = soup_album.find(class_='albumUserScore').find('a').contents[0]
    except:
        total_user_rating = 0
    try:
        total_num_ratings = \
            soup_album.find(class_='albumUserScoreBox').find(class_='text numReviews').find('strong').contents[0]
    except:
        total_num_ratings = 0

    tf_vector = pickle.load(open('Mood\\final_tfidf2.sav', 'rb'))
    mood_model = pickle.load(open('Mood\\final_LR2.sav', 'rb'))

    API = azapi.AZlyrics(accuracy=0.5)
    API.artist = artist_name

    album_info = {
        'Album Name': album_title,
        'Artist Name': artist_name,
        'Genres': album_genres,
        'Track List': album_track_list,
        'Release Date': album_date,
        'Total Critic Rating': total_critic_rating,
        'Total Audience Rating': total_user_rating,
        'Total Number of User Ratings': total_num_ratings
    }
    album_features = {
        'L_relaxed_mood': 0, 'L_happy_mood': 0, 'L_sad_mood': 0, 'L_angry_mood': 0,
        'L_anger': 0, 'L_fear': 0, 'L_joy': 0, 'L_sadness': 0, 'L_surprise': 0, 'L_love': 0, 'L_sentiment': 0,
        'M_valence': 0, 'M_arousal': 0,
        'V_anger': 0, 'V_fear': 0, 'V_joy': 0, 'V_sadness': 0, 'V_surprise': 0, 'V_love': 0, 'V_sentiment': 0,
        'V_view_count': 0
    }

    # ----------------------------- Track - Level Analysis -----------------------------
    for track in album_track_list:
        print('          ', track + " !")
        API.title = track
        try:
            Lyrics = API.getLyrics(save=False)  # save=True, ext='txt', path='lyrics'
        except:
            Lyrics = 1

        if not Lyrics == 1:
            L_mood = predict_lyric_mood(Lyrics, mood_model, tf_vector)
            album_features[L_mood] = album_features.get(L_mood, 0) + 1

            L_emotions = predict_emotion([Lyrics], False)
            for Lem in L_emotions:
                album_features['L_' + Lem] = album_features['L_' + Lem] + L_emotions[Lem]

        # Alternative Source of Lyric Scraping. Custom Google Search Engine for Lyrics
        else:
            try:
                google_lyrics = extract_lyrics.get_lyrics(artist_name + ' ' + track)
                L_mood = predict_lyric_mood(google_lyrics['lyrics'], mood_model, tf_vector)
                album_features[L_mood] = album_features.get(L_mood, 0) + 1

                L_emotions = predict_emotion([google_lyrics['lyrics']], False)
                for Lem in L_emotions:
                    album_features['L_' + Lem] = album_features['L_' + Lem] + L_emotions[Lem]
            except:
                print('lyrics for : ', track, ' were not found!')
                pass

        M_valence, M_arousal = get_track_valence_arousal(sp, artist_name, track)
        album_features['M_valence'] = album_features.get('M_valence', 0) + M_valence
        album_features['M_arousal'] = album_features.get('M_arousal', 0) + M_arousal

        try:
            V_emotion, V_view_count = search_track(youtube_object, artist_name + ' ' + track)
            if V_emotion != 0:
                for em in V_emotion:
                    album_features['V_' + em] = album_features.get('V_' + em, 0) + V_emotion[em]
            album_features['V_view_count'] = album_features['V_view_count'] + int(V_view_count)
        except:
            print('          Youtube API Error!')
            print('          Break!')
            print('          Stopped on album : ', album_title, 'by artist : ', artist_name)
            return 0

        print('Before scraping the next song : Sleep for 5')
        time.sleep(5)

    if len(album_track_list) > 0:
        for i in album_features:
            if i != 'V_view_count':
                album_features[i] = float(album_features[i] / len(album_track_list))

    album_info.update(album_features)

    try:
        col_album.insert_one(album_info)
        print('          Inserted One : Album')
    except pymongo.errors.DuplicateKeyError:
        print(' --- Exception: Error at Album Insertion --- ')
    album_info.pop('_id')
    album_info.pop('Track List')

    print('Scraped Album : Sleep for 10')
    time.sleep(10)

    # ----------------------------- USER REVIEWS PAGE -----------------------------
    try:
        users_link = 'https://www.albumoftheyear.org' + \
                     soup_album.find(id='users').find(class_='viewAll').find('a').get('href')
        users_page = requests.get(users_link)
        users_soup = BeautifulSoup(users_page.content, 'html.parser')
        page_count_users = 1
        user_that_reviewed = []

        # Scrape User-Review pages
        while True:
            if page_count_users == 1:
                users_all = users_soup.find_all(class_='albumReviewRow')
                page_count_users += 1
                if not users_all == []:
                    for u in users_all:
                        try:
                            user_dict_reviews = users_review(u)
                            if not user_dict_reviews == 'not english' and not user_dict_reviews == 'empty':
                                user_that_reviewed.append(user_dict_reviews['user name'])
                                user_dict_reviews.update(album_info)
                                try:
                                    col_reviewers.insert_one(user_dict_reviews)
                                except pymongo.errors.DuplicateKeyError:
                                    pass
                            else:
                                # print('Empty or Non-English review. Was not inserted.')
                                pass
                        except:
                            # print('Error at Inserting Reviewer')
                            pass

            elif not users_soup.find(class_='pageSelect next') is None:
                print('Reviewers : Sleep for 5')
                time.sleep(5)
                if users_soup.find(class_='pageSelect next').contents[0] == 'Next':
                    page_count_users += 1
                    users_link = 'https://www.albumoftheyear.org' + \
                                 soup_album.find(id='users').find(class_='viewAll').find('a').get('href') + '?p=' + \
                                 str(page_count_users)
                    users_page = requests.get(users_link)
                    users_soup = BeautifulSoup(users_page.content, 'html.parser')
                    users_all = users_soup.find_all(class_='albumReviewRow')
                    if not users_all == []:
                        for u in users_all:
                            try:
                                user_dict_reviews = users_review(u)
                                if not user_dict_reviews == 'not english' and not user_dict_reviews == 'empty':
                                    user_that_reviewed.append(user_dict_reviews['user name'])
                                    user_dict_reviews.update(album_info)
                                    try:
                                        col_reviewers.insert_one(user_dict_reviews)

                                    except pymongo.errors.DuplicateKeyError:
                                        # print('Exception: Reviewer Insert')
                                        pass
                                else:
                                    # print('Empty or Non English review. Was not inserted.')
                                    pass
                            except:
                                # print('Error at Inserting Reviewer')
                                pass
            else:
                break
    except:
        print('          Error in USER - REVIEW page! Skip this section!')

    # ----------------------------- USER RATING PAGE -----------------------------

    def raters(users_rt, already_reviewed, album_data, page_count):
        if not users_rt == []:
            for user in users_rt:
                user_name = user.find(class_='userName').find('a').contents[0]
                user_rating = user.find(class_='rating').contents[0]
                if user_name not in already_reviewed:
                    user_dict_ratings = {"user name": user_name, "rating": user_rating}
                    user_dict_ratings.update(album_data)
                    try:
                        col_raters.insert_one(user_dict_ratings)
                    except pymongo.errors.DuplicateKeyError:
                        pass
                        # print('Exception: Rater Insert')
                else:
                    # print(user_name, ' is already in reviewers')
                    pass

    # print('Scraped Reviewers' Page : Sleep for 10')
    time.sleep(10)

    user_rating_page_count = 1
    try:
        while True:
            if user_rating_page_count == 1:
                try:
                    users_ratings_link = 'https://www.albumoftheyear.org' + \
                                         soup_album.find(id='users').find(class_='viewAll').find('a').get('href') \
                                         + '?p=' + str(user_rating_page_count) + '&type=ratings'
                except:
                    link = link[:-4]
                    users_ratings_link = link + '/user-reviews/' + '?type=ratings'

                users_rating_page = requests.get(users_ratings_link)
                users_rating_soup = BeautifulSoup(users_rating_page.content, 'html.parser')
                users_all_ratings = users_rating_soup.find_all(class_='userRatingBlock')
                user_rating_page_count += 1
                try:
                    raters(users_all_ratings, user_that_reviewed, album_info, user_rating_page_count)
                except:
                    # print('Could Not Insert : Rater')
                    pass

            elif not users_rating_soup.find(class_='pageSelect next') is None:
                time.sleep(5)
                # print('Raters: Sleep for 5')
                if users_rating_soup.find(class_='pageSelect next').contents[0] == 'Next':
                    user_rating_page_count += 1
                    users_ratings_link = 'https://www.albumoftheyear.org' + \
                                         soup_album.find(id='users').find(class_='viewAll').find('a').get('href') \
                                         + '?p=' + str(user_rating_page_count) + '&type=ratings'

                    users_rating_page = requests.get(users_ratings_link)
                    users_rating_soup = BeautifulSoup(users_rating_page.content, 'html.parser')
                    users_all_ratings = users_rating_soup.find_all(class_='userRatingBlock')
                    try:
                        raters(users_all_ratings, user_that_reviewed, album_info, user_rating_page_count)
                    except:
                        # print('Could Not Insert : Rater')
                        pass
            else:
                break
    except:
        print('          Error at Scraping USER RATINGS page! Skip this section')
        pass
    return 1
    # ----------------------------- END -----------------------------


# ----------------------------- MAIN -----------------------------

# Year 2020 : Pages 1 to 28 are done!
# Year 2019 : Pages 1 to 31 are done!
# Year 2018 : Pages 1 to 34 are done!
year = 2017
page_count_albums = 1
URL = 'https://www.albumoftheyear.org/ratings/6-highest-rated/' + str(year) + '/' + str(page_count_albums)
page = requests.get(URL)
print(page.status_code)
soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id='centerContent')

key_value = 0  # 0 - 8

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
youtube_object = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=yt_keys[key_value])

while True:
    if page_count_albums == 1:
        albums = results.find_all(class_='albumListRow')
        for index, Album in enumerate(albums):

            if 0 <= index <= 24:
                print('Album Index:', index, 'with Youtube Key: ', key_value)
                a = album_review(Album)
                if a == 0:
                    print('--- Error at Page: ' + str(page_count_albums) + '! \n at Index: ' + str(index) + '!')
                    print('--- Last used key:', key_value, ' ---')
                    key_value += 1
                    print('--- Use next key: ', key_value, ' ---')

                    print('--- Re-scrape the album where the error occurred ---')
                    youtube_object = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                                           developerKey=yt_keys[key_value])
                    a = album_review(Album)
                    if a == 0:
                        print("------ Last Page: ", page_count_albums, ' ------')
                        print("------ Last Item: ", index, ' ------')
                        break

                print('New Album : Sleep for 5')
                time.sleep(5)

            if index == 24:
                page_count_albums += 1

    if results.find(class_='pageSelect next'):
        print('New Album Page : Sleep for 10')
        time.sleep(10)
        print('Scraping Album-Page: ', page_count_albums)
        URL = 'https://www.albumoftheyear.org/ratings/6-highest-rated/' + str(year) + '/' + str(page_count_albums)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.find(id='centerContent')
        albums = results.find_all(class_='albumListRow')
        for index, Album in enumerate(albums):
            if 0 <= index <= 24:
                print('Album Index:', index, 'with Youtube Key: ', key_value, ' ---')
                a = album_review(Album)
                if a == 0:
                    print('--- Error at Page: ' + str(page_count_albums) + '! \n at Index: ' + str(index) + '!')
                    print('--- Last used key:', key_value, ' ---')
                    key_value += 1
                    print('--- Use next key: ', key_value, ' ---')

                    print('--- Re-scrape the album where the error occurred ---')
                    youtube_object = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                                           developerKey=yt_keys[key_value])
                    a = album_review(Album)
                    if a == 0:
                        print("------ Last Page: ", page_count_albums, ' ------')
                        print("------ Last Item: ", index, ' ------')
                        break

                print('New Album : Sleep for 5')
                time.sleep(5)

            if index == 24:
                page_count_albums += 1
    else:
        break

    if key_value > len(yt_keys):
        print("------ End of keys! ------")
        print("------ Last Page: ", page_count_albums, ' ------')
        break
