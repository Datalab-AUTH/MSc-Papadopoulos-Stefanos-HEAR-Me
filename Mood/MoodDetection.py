import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk.stem.porter import PorterStemmer
import string

def preprocess_lyric_tf(input_text, tf_vector):

    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    words = [word.lower() for word in input_text.split()]
    words = [w for w in words if not w in stop_words]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    stemmed = [porter.stem(word) for word in stripped]

    valence, arousal, dominance = nrc_vad(' '.join(stemmed))
    mood_df = pd.DataFrame({"valence": [valence], "arousal": [arousal], "dominance": [dominance]})

    tf_text = tf_vector.transform([' '.join(stemmed)])
    tf_text = pd.DataFrame(tf_text.todense())

    tf_mood = pd.concat([tf_text, mood_df], axis=1)
    return tf_mood

def predict_lyric_mood(lyric, model, tf):
    tf_mood = preprocess_lyric_tf(lyric, tf)
    y_predict = model.predict(tf_mood)
    # my_tags = ['relaxed', 'happy', 'sad', 'angry']
    if y_predict[0] == 0:
        return 'L_relaxed_mood'
    elif y_predict[0] == 1:
        return 'L_happy_mood'
    elif y_predict[0] == 2:
        return 'L_sad_mood'
    else:
        return 'L_angry_mood'

def nrc_vad(input_text):
    if input_text is None:
        return 0, 0, 0

    filepath = "mood\\NRC-VAD-Lexicon.txt"
    lexicon_df = pd.read_csv(filepath, sep='\t')
    words_mood = lexicon_df.iloc[:, 0].tolist()

    arousal = valence = dominance = word_count = 0

    try:
        for word in input_text.split():
            if word in words_mood:
                word_count += 1
                mood = lexicon_df.loc[lexicon_df['Word'] == word]
                valence += mood.iat[0,1]  # Mood['Valence'].values[0] or Mood.iloc[0]['Valence']
                arousal += mood.iat[0,2]
                dominance += mood.iat[0,3]

        if word_count > 0:
            num_of_words = len(input_text)
            valence = valence / word_count
            arousal = arousal / word_count
            dominance = dominance / word_count
        else:
            arousal = valence = dominance = 0
    except:
        return 0,0,0
    return round(valence,4), round(arousal,4), round(dominance,4)


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