import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from itertools import islice
from sklearn.naive_bayes import MultinomialNB
from Mood.MoodDetection import nrc_vad
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import model_selection, metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.linear_model import SGDClassifier
from skopt import BayesSearchCV
import pandas as pd
import azapi
import pickle

def lyric_collection():
    data = pd.read_csv("data/ml_balanced.csv")

    API = azapi.AZlyrics(accuracy=0.5)
    lyric_df = pd.DataFrame(columns=['Artist', 'Track', 'Lyrics', 'Valence', 'Arousal', 'Dominance'])

    for index, row in islice(data.iterrows(), 0, None):  # 1001
        print(row['Artist'], row['Title'], row['Mood'], index)

        API.artist = row['Artist']
        API.title = row['Title']

        lyrics = API.getLyrics(save=False, sleep=5)

        if not lyrics == 0:
            L_valence, L_arousal, L_dominance = nrc_vad(lyrics)  # lyric's Mood from NRC VAD
            if L_valence > 0 or L_arousal > 0 or L_dominance > 0:
                mood_dict = {'Artist': row['Artist'], 'Track': row['Title'], 'Lyrics': lyrics,
                        'Valence': L_valence, 'Arousal': L_arousal, 'Dominance': L_dominance, 'Target': row['Mood']}
                lyric_df = lyric_df.append(mood_dict, ignore_index=True)

    lyric_df.to_csv('lyric_dataset.csv', encoding='utf-8', index=False)


def prepare_lyric_dataset():
    pd.options.mode.chained_assignment = None
    data = pd.read_csv("lyric_dataset1.csv")
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    clean_text_list = []

    for index, row in islice(data.iterrows(), 0, None):
        words = word_tokenize(row['Lyrics'])
        words = [word.lower() for word in words]
        words = [w for w in words if not w in stop_words]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        stemmed = [porter.stem(word) for word in stripped]
        clean_text_list.append(stemmed)

        valence, arousal, dominance = nrc_vad(' '.join(stemmed))
        print(valence, arousal, dominance)
        data['Valence'][index] = valence
        data['Arousal'][index] = arousal
        data['Dominance'][index] = dominance

    data.insert(3, "Clean Text", clean_text_list, True)

    # Normalize Mood values to {0,1} range
    minMaxScaler = preprocessing.MinMaxScaler(copy="True", feature_range=(0, 1))

    Valence = pd.DataFrame.from_dict(data['Valence'])
    minMaxScaler.fit(Valence)
    Valence = minMaxScaler.transform(Valence)
    Valence = pd.DataFrame.from_dict(Valence)

    Arousal = pd.DataFrame.from_dict(data['Arousal'])
    minMaxScaler.fit(Arousal)
    Arousal = minMaxScaler.transform(Arousal)
    Arousal = pd.DataFrame.from_dict(Arousal)

    Dominance = pd.DataFrame.from_dict(data['Dominance'])
    minMaxScaler.fit(Dominance)
    Dominance = minMaxScaler.transform(Dominance)
    Dominance = pd.DataFrame.from_dict(Dominance)

    data['Valence'] = Valence
    data['Arousal'] = Arousal
    data['Dominance'] = Dominance

    data.to_csv('lyrics_clean5.csv', encoding='utf-8', index=False)

def k_fold_evaluation(x, y, model):
    accuracy_model = []
    precision_model = []
    recall_model = []
    f1_model = []
    kf = model_selection.StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        accuracy_model.append(metrics.accuracy_score(y_test, y_predict, normalize=True) * 100)
        precision_model.append(metrics.precision_score(y_test, y_predict, average="macro") * 100)
        recall_model.append(metrics.recall_score(y_test, y_predict, average="macro") * 100)
        f1_model.append(metrics.f1_score(y_test, y_predict, average="macro") * 100)

    print("K Fold Accuracy scores:", accuracy_model)
    print("Mean KFold Accuracy: ", round(np.mean(accuracy_model), 4))
    print("K Fold Precision scores:", precision_model)
    print("Mean K Fold Precision scores:", round(np.mean(precision_model), 4))
    print("K Fold Recall scores:", recall_model)
    print("Mean K Fold Recall scores:", round(np.mean(recall_model), 4))
    print("K Fold F1 scores:", f1_model)
    print("Mean KFold F1: ", round(np.mean(f1_model), 4))

def model_train():
    data = pd.read_csv('lyrics_clean5.csv')
    X = data.drop(columns=['Artist', 'Track', 'Lyrics', 'Target'])

    my_tags = ['relaxed', 'happy', 'sad', 'angry']
    data[['Target']] = data[['Target']].replace(['relaxed'], 0)
    data[['Target']] = data[['Target']].replace(['happy'], 1)
    data[['Target']] = data[['Target']].replace(['sad'], 2)
    data[['Target']] = data[['Target']].replace(['angry'], 3)
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    X_mood_train = pd.DataFrame(
        {'Valence': X_train['Valence'], 'Arousal': X_train['Arousal'], 'Dominance': X_train['Dominance']})
    X_mood_test = pd.DataFrame(
        {'Valence': X_test['Valence'], 'Arousal': X_test['Arousal'], 'Dominance': X_test['Dominance']})

    tf_vector = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), norm='l2')

    # Fit & Transform of train into a Tf-Idf vector
    tf_vector.fit(X_train['Clean Text'])
    Tfidf_vector = tf_vector.transform(X_train['Clean Text'])
    X_TF_train = pd.DataFrame(Tfidf_vector.todense())
    X_TF_mood_train = pd.concat([X_TF_train, X_mood_train], axis=1)

    # Transform Only of Test Data with the Tf-Idf vector
    X_TF_test = tf_vector.transform(X_test['Clean Text'])
    X_TF_test = pd.DataFrame(X_TF_test.todense())
    X_TF_mood_test = pd.concat([X_TF_test, X_mood_test], axis=1)

    # Oversampling with SMOTE
    print('Performing oversampling')
    print(sorted(Counter(y_train).items()))
    sm = SMOTE(random_state=2)  # sampling_strategy=1
    # X_TF_train_Res, y_train_Res = sm.fit_sample(X_TF_train, y_train)
    # X_mood_train_Res, y_train_Res = sm.fit_sample(X_mood_train, y_train)
    X_TF_mood_train_Res, y_train_Res = sm.fit_sample(X_TF_mood_train, y_train)
    print(sorted(Counter(y_train_Res).items()))

    # Grid and Bayes Search
    # model_optimal(X_mood_train_Res, y_train_Res,X_TF_mood_test,y_test)

    # Model Training
    # model = MultinomialNB(alpha=1)
    model = LogisticRegression(solver='lbfgs', C=100, multi_class='auto', random_state=42)
    # model = SGDClassifier(alpha=0.001,loss='hinge',penalty='l2',shuffle=True)
    # k_fold_evaluation(X_TF_mood_train_Res,y_train_Res,model)

    model.fit(X_TF_mood_train_Res, y_train_Res)
    y_predict = model.predict(X_TF_mood_test)

    print("Accuracy Score: %.8f" % accuracy_score(y_test, y_predict))
    print("Precision: %.8f" % metrics.precision_score(y_test, y_predict, average="macro"))
    print("Recall: %.8f" % metrics.recall_score(y_test, y_predict, average="macro"))
    print("F1: %.8f" % metrics.f1_score(y_test, y_predict, average="macro"))
    print(classification_report(y_test, y_predict, target_names=my_tags))

def model_optimal(x_train, y_train, x_test, y_test):
    opt = BayesSearchCV(
        # MultinomialNB(),
        # {'alpha': [0, 0.2, 0.4, 0.6, 0.8, 1]},
        # SGDClassifier(loss='hinge',penalty='l2',shuffle=True),
        # {'alpha': [0.0001, 0.001, 0.01, 0.1] },
        LogisticRegression(multi_class='auto'),
        {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        cv=3)

    opt.fit(x_train, y_train)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(x_test, y_test))
    print("best params: %s" % str(opt.best_params_))
    print(opt.best_estimator_)

    # best SGD params: OrderedDict([('alpha', 0.001)])
    # best LR params: OrderedDict([('C', 100)])
    # Best MNB params OrderedDict([('alpha', 1.0])

def finalise_model():
    data = pd.read_csv('lyrics_clean5.csv')
    data[['Target']] = data[['Target']].replace(['relaxed'], 0)
    data[['Target']] = data[['Target']].replace(['happy'], 1)
    data[['Target']] = data[['Target']].replace(['sad'], 2)
    data[['Target']] = data[['Target']].replace(['angry'], 3)
    y = data['Target']

    tf_vector = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), norm='l2')
    print('Performing oversampling')
    print("Before OverSampling, counts of label '3': {}".format(sum(y == 3)))
    print("Before OverSampling, counts of label '2': {}".format(sum(y == 2)))
    print("Before OverSampling, counts of label '1': {}".format(sum(y == 1)))
    print("Before OverSampling, counts of label '0': {}".format(sum(y == 0)))

    X_mood = pd.DataFrame({'Valence': data['Valence'], 'Arousal': data['Arousal'], 'Dominance': data['Dominance']})

    tf_vector.fit(data['Clean Text'])
    filename1 = 'final_tfidf2.sav'
    pickle.dump(tf_vector, open(filename1, 'wb'))

    tfidf_vector = tf_vector.transform(data['Clean Text'])
    X_TF = pd.DataFrame(tfidf_vector.todense())
    X_TF_mood = pd.concat([X_TF, X_mood], axis=1)

    sm = SMOTE(random_state=2)
    X_TF_mood_Res, y_Res = sm.fit_sample(X_TF_mood, y)
    print(sorted(Counter(y_Res).items()))

    model = LogisticRegression(solver='lbfgs', C=100, multi_class='auto', random_state=42)
    model.fit(X_TF_mood_Res, y_Res)

    filename2 = 'final_LR2.sav'
    pickle.dump(model, open(filename2, 'wb'))