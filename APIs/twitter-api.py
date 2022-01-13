import datetime
import time
import re
import langdetect
import pymongo
import tweepy
from langdetect import detect

from tweepy import API, OAuthHandler
from utilities.keys import twitter_access_token,twitter_access_token_secret,twitter_consumer_key,twitter_consumer_secret
from nltk import TweetTokenizer, WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords

auth = OAuthHandler(twitter_consumer_key, twitter_consumer_secret)
auth.set_access_token(twitter_access_token, twitter_access_token_secret)
api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# self.connection = MongoHandler()

# Preprocess tweet text
def preprocess_tweet(self, tweet):
    tweet_dict = dict()
    tweet_dict["_id"] = tweet["id"]
    created_at = time.strftime('%Y-%m-%d', time.strptime(tweet["created_at"], '%a %b %d %H:%M:%S +0000 %Y'))
    tweet_dict["created_at"] = created_at
    tweet_dict["text"] = preprocess_text(tweet["full_text"])
    tweet_dict["hashtags"] = [hashtag["text"] for hashtag in tweet["entities"]["hashtags"]]
    tweet_dict["mentions"] = [hashtag["name"] for hashtag in tweet["entities"]["user_mentions"]]
    tweet_dict["hashtags"] = [hashtag["text"] for hashtag in tweet["entities"]["hashtags"]]
    tweet_dict["urls"] = [hashtag["url"] for hashtag in tweet["entities"]["urls"]]
    tweet_dict["user_id"] = tweet["user"]["id"]
    tweet_dict["user_name"] = tweet["user"]["name"]
    tweet_dict["user_screen_name"] = tweet["user"]["screen_name"]
    tweet_dict["user_location"] = tweet["user"]["location"]
    tweet_dict["user_followers"] = tweet["user"]["followers_count"]
    tweet_dict["user_friends"] = tweet["user"]["friends_count"]
    tweet_dict["user_listed"] = tweet["user"]["listed_count"]
    tweet_dict["user_favourites"] = tweet["user"]["favourites_count"]
    ts = time.strftime('%Y-%m', time.strptime(tweet["user"]["created_at"], '%a %b %d %H:%M:%S +0000 %Y'))
    date_time_obj = datetime.datetime.strptime(ts, '%Y-%m')
    end_date = datetime.datetime.now()
    num_months = (end_date.year - date_time_obj.year) * 12 + (end_date.month - date_time_obj.month)
    tweet_dict["user_months"] = num_months
    tweet_dict["user_statuses"] = tweet["user"]["statuses_count"]
    tweet_dict["user_verified"] = int(tweet["user"]["verified"])
    tweet_dict["retweets"] = tweet["retweet_count"]
    tweet_dict["favorites"] = tweet["favorite_count"]
    tweet_dict["is_quoted"] = tweet["is_quote_status"]
    self.connection.store_to_collection(tweet_dict, "twitter_new")
    return tweet_dict

# Get tweets from a particular user
def get_user_tweets(self):
    re_list = []
    users = []
    count_users = 0
    for user in users:
        try:
            print("User: ", user)
            user_tweets = []
            count_tweets = 0
            for i in range(1, 20):
                statuses = self.api.user_timeline(screen_name=user,
                                                  count=50, page=i, lang="en",
                                                  tweet_mode="extended")
                for status in statuses:
                    if detect(status.full_text) == 'en' and len(status.full_text.split()) >= 5:
                        # and not status.full_text.startswith("RT @"):
                        status_dict = dict()
                        status_dict["_id"] = status.id
                        status_dict["user_name"] = status.author.screen_name
                        status_dict["location"] = status.author.location
                        status_dict["description"] = preprocess_text(status.author.description)
                        status_dict[
                            'date'] = f"{status.created_at.year}-{status.created_at.month}-{status.created_at.day}"
                        clean_text = preprocess_text(re.sub(r'^RT\s@\w+:', r'', status.full_text))
                        status_dict["text"] = clean_text

                        # status_dict["sentiment"] = round(sentiment_analyzer_scores(status.full_text)['compound'], 3)

                        # anger, anticipation, disgust, fear, joy, _negative, _positive, sadness, surprise, trust = get_emotions(
                        #     clean_text)
                        # status_dict["anger"] = anger
                        # status_dict["anticipation"] = anticipation
                        # status_dict["disgust"] = disgust
                        # status_dict["fear"] = fear
                        # status_dict["joy"] = joy
                        # status_dict["sadness"] = sadness
                        # status_dict["surprise"] = surprise
                        # status_dict["trust"] = trust
                        #
                        # subj = TextBlob(''.join(status.full_text)).sentiment
                        # status_dict["subjectivity"] = round(subj[1], 3)

                        # status_dict["label"] = 0 # non - denier
                        # status_dict["label"] = 1 # denier
                        user_tweets.append(status_dict)
                # re_list.append(statuses)

            for status_dict in user_tweets:
                try:
                    self.connection.store_to_collection(status_dict,
                                                        "twitter_profiles_1K")  # new_twitter_profiles for training data
                    count_tweets += 1
                except pymongo.errors.DuplicateKeyError:
                    # print(status_dict.id)
                    print("exception")
                    continue
            print("Found ", count_tweets, " relevant tweets by the user: ", user)
            count_users += 1
            if (count_users % 20) == 0:
                print("test sleep!")
                time.sleep(300)
                print("test sleep ended!!!")
            if count_users > 1001:
                print("break!")
                break
        except tweepy.error.TweepError:
            print("Locked profile!")
            continue
        except langdetect.lang_detect_exception.LangDetectException:
            continue

    return re_list

# Preprocesses text from tweeter
def preprocess_text(tweet_text):
    tweet_tokenizer = TweetTokenizer()

    punctuations = '''.~-@#$%^&*'":;`|/<>()'''
    en_stopwords = [
        word for word in set(stopwords.words('english'))
        if word not in [
            "not", "but", "neither", "nor", "either", "or", "rather", "never", "none", "nobody", "nowhere", "hardly"
        ]
    ]
    tokens = [token.lower().lstrip("@").lstrip("#") for token in tweet_tokenizer.tokenize(tweet_text)]
    contractions = get_contractions()
    tokens_no_contra = [contractions[token].split() if token in contractions else [token] for token in tokens]
    flat_list = [item for sublist in tokens_no_contra for item in sublist]
    tokens_semi_final = [token for token in flat_list if token not in punctuations and token not in en_stopwords]
    final_t = [token.replace("'s", "") for token in tokens_semi_final if
               not re.match('((www\.[^\s]+)|(https?://[^\s]+))', token)]

    text = []
    wnl = WordNetLemmatizer()
    tagged = pos_tag(final_t)
    for word, tag_prior in tagged:
        tag = nltk_tag_to_wordnet_tag(tag_prior)
        word = "not" if word == "n't" else word
        if tag:
            text.append(wnl.lemmatize(word.lower(), tag))
        else:
            text.append(wnl.lemmatize(word.lower()))

    return text

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Get hashtags from tweets
def get_hashtags(text):
    tweet_tokenizer = TweetTokenizer()
    return [token for token in tweet_tokenizer.tokenize(text) if re.match("^#(\w+)", token)]

# Get mentions from tweets
def get_mentions(text):
    tweet_tokenizer = TweetTokenizer()
    return [token for token in tweet_tokenizer.tokenize(text) if
            re.match("^@(?!.*\.\.)(?!.*\.$)[^\W][\w.]{0,29}$", token)]

def get_contractions():
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "I would",
        "i'd've": "I would have",
        "i'll": "I will",
        "i'll've": "I will have",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }
    return contractions

