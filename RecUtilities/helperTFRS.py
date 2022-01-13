import pandas as pd
import numpy as np
import random
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_recommenders as tfrs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from RecSystem.RecUtilities.FetchData import fetch_df


def data_to_tf(data_source, research_stage):
    if research_stage == 'tuning':
        data, _, train_set, val_set, cold_items, unique_cold_items, popularity_dict \
            = fetch_df(data_source=data_source, encode=False, k_bin=0, tuning=True)
    else:
        data, _, train_set, val_set, cold_items, unique_cold_items, popularity_dict \
            = fetch_df(data_source=data_source, encode=False, k_bin=0, tuning=False)

    def pd_to_tf(input_data):
        tf_data = tf.data.Dataset.from_tensor_slices((dict(input_data)))

        # ------------------------------- Prepare Interaction Data ------------------------------- #
        interactions_tf = tf_data.map(lambda x: {
            "user_name": x["user_name"],
            "album_name": x["album_name"],
            "rating": x["rating"],
            "Genres": x["Genres"],
            "r_anger": x["r_anger"],
            "r_joy": x["r_joy"],
            "r_love": x["r_love"],
            "r_sadness": x["r_sadness"],
            "r_surprise": x["r_surprise"],
            "r_sentiment": x["r_sentiment"],
            "L_relaxed_mood": x['L_relaxed_mood'],
            "L_happy_mood": x['L_happy_mood'],
            "L_sad_mood": x['L_sad_mood'],
            "L_angry_mood": x['L_angry_mood'],
            "L_anger": x["L_anger"],
            "L_fear": x["L_fear"],
            "L_joy": x["L_joy"],
            "L_sadness": x["L_sadness"],
            "L_surprise": x["L_surprise"],
            "L_love": x["L_love"],
            "L_sentiment": x["L_sentiment"],
            "M_valence": x["M_valence"],
            "M_arousal": x["M_arousal"],
            "V_anger": x["V_anger"],
            "V_fear": x["V_fear"],
            "V_joy": x["V_joy"],
            "V_sadness": x["V_sadness"],
            "V_surprise": x["V_surprise"],
            "V_love": x["V_love"],
            "V_sentiment": x["V_sentiment"],
            "V_view_count": x["V_view_count"],
            "critic_t_rating": x["critic_t_rating"],
            "audience_t_rating": x["audience_t_rating"],
            "num_user_ratings": x["num_user_ratings"],
            "followers_count": x["followers_count"],
            "ANXIETY": x["ANXIETY"],
            "AVOIDANCE": x["AVOIDANCE"],
            "O": x["O"],
            "C": x["C"],
            "E": x["E"],
            "A": x["A"],
            "N": x["N"]
        })

        # for x in interactions_tf.take(1).as_numpy_iterator():
        #     pprint.pprint(x)

        # ------------------------------- Prepare Album Data ------------------------------- #
        album_data = input_data[['album_name', 'Genres',
                                 'L_relaxed_mood', 'L_happy_mood', 'L_sad_mood', 'L_angry_mood',
                                 'L_anger', 'L_fear', 'L_joy', 'L_sadness', 'L_surprise', 'L_love', 'L_sentiment',
                                 'M_valence', 'M_arousal',
                                 'V_anger', 'V_fear', 'V_joy', 'V_sadness', 'V_surprise', 'V_love', 'V_sentiment',
                                 'critic_t_rating', 'audience_t_rating', 'num_user_ratings', 'V_view_count',
                                 "followers_count",
                                 "ANXIETY", "AVOIDANCE", "O", "C", "E", "A", "N"
                                 ]].drop_duplicates(subset=['album_name']).reset_index(drop=True)

        tf_album_data = tf.data.Dataset.from_tensor_slices((dict(album_data)))

        albums_tf = tf_album_data.map(lambda x: {
            "album_name": x["album_name"],
            "Genres": x["Genres"],
            'L_relaxed_mood': x["L_relaxed_mood"],
            'L_happy_mood': x["L_happy_mood"],
            'L_sad_mood': x["L_sad_mood"],
            'L_angry_mood': x["L_angry_mood"],
            "L_anger": x["L_anger"],
            "L_fear": x["L_fear"],
            "L_joy": x["L_joy"],
            "L_sadness": x["L_sadness"],
            "L_surprise": x["L_surprise"],
            "L_love": x["L_love"],
            "L_sentiment": x["L_sentiment"],
            "M_valence": x["M_valence"],
            "M_arousal": x["M_arousal"],
            "V_anger": x["V_anger"],
            "V_fear": x["V_fear"],
            "V_joy": x["V_joy"],
            "V_sadness": x["V_sadness"],
            "V_surprise": x["V_surprise"],
            "V_love": x["V_love"],
            "V_sentiment": x["V_sentiment"],
            "V_view_count": x["V_view_count"],
            "critic_t_rating": x["critic_t_rating"],
            "audience_t_rating": x["audience_t_rating"],
            "num_user_ratings": x["num_user_ratings"],
            "followers_count": x["followers_count"],
            "ANXIETY": x["ANXIETY"],
            "AVOIDANCE": x["AVOIDANCE"],
            "O": x["O"],
            "C": x["C"],
            "E": x["E"],
            "A": x["A"],
            "N": x["N"]
        })
        # for x in albums.take(5).as_numpy_iterator():
        #     pprint.pprint(x)
        return interactions_tf, albums_tf

    train_interaction, _ = pd_to_tf(train_set)
    val_interaction, _ = pd_to_tf(val_set)

    interactions, albums = pd_to_tf(data)
    cold_item_interactions, cold_item_albums = pd_to_tf(cold_items)

    return (interactions, albums, train_interaction, _, val_interaction,
            popularity_dict, cold_item_interactions, cold_item_albums, unique_cold_items)  # val_interaction

# ------------------------------- Model Evaluation ------------------------------- #
def plot_history(hist, choice):
    if hist:
        if choice == 'topk' or choice == 'both':
            plt.plot([element * 100 for element in hist.history['factorized_top_k/top_100_categorical_accuracy']],
                     label='Training top 100')
            plt.plot([element * 100 for element in hist.history['val_factorized_top_k/top_100_categorical_accuracy']],
                     label='Validation top 100')
            plt.plot([element * 100 for element in hist.history['factorized_top_k/top_10_categorical_accuracy']],
                     label='Training top 10')
            plt.plot([element * 100 for element in hist.history['val_factorized_top_k/top_10_categorical_accuracy']],
                     label='Validation top 10')

            plt.legend(loc='best')
            plt.grid(True)
            plt.title('Training History')
            plt.ylabel('Factorized Top-K')
            plt.xlabel('Epochs')
            plt.show()

        if choice == 'rmse' or choice == 'both':
            plt.plot(hist.history['root_mean_squared_error'], label='Training RMSE')
            plt.plot(hist.history['val_root_mean_squared_error'], label='Validation RMSE')
            plt.legend(loc='best')
            plt.grid(True)
            plt.title('Training History : RMSE')
            plt.ylabel('RMSE')
            plt.xlabel('Epochs')
            plt.show()


def eval_diversity(model, shuffled_interactions, albums, popularity_dict, n_users, cold_list):
    # ----------------------- Perform Recommendations -----------------------
    nn = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
    albums_identifiers = albums.map(lambda x: x["album_name"])
    nn.index(candidates=albums.batch(1024).map(model.candidate_model), identifiers=albums_identifiers)
    # for row in shuffled_interactions.batch(1).take(2):
    #     print(f"Best recommendations: {nn(row)[1].numpy()[:, :3].tolist()}")

    topk = 10
    covered_items = []
    total_rec_list = []
    total_rec_list_100 = []
    for row in shuffled_interactions.batch(1).take(1200):
        rec_list = nn(row)[1].numpy()[:, :topk].flatten()
        rec_list_100 = nn(row)[1].numpy()[:, :100].flatten()
        total_rec_list_100.append(rec_list_100)

        total_rec_list.append(rec_list.tolist())
        for item in rec_list:
            covered_items.append(item)

    # ----------------------- Calculate Cold Item Probability -----------------------

    cold_check_counter = 0
    for rec100 in total_rec_list_100:
        rl = np.char.decode(rec100.astype(np.bytes_), 'UTF-8')
        cold_check_counter += len(set(cold_list).intersection(rl))
    cold_prob = cold_check_counter / 1200

    # ----------------------- Calculate Coverage Rate -----------------------
    set_list = set(covered_items)
    unique_list = (list(set_list))
    unique_album_titles = np.unique(np.concatenate(list(albums.batch(1000).map(lambda x: x["album_name"]))))
    coverage = len(unique_list) / len(unique_album_titles) * 100
    # print("Item Coverage at top-" + str(topk) + " (based on 1000 random samples): " + "{:.2f}".format(coverage) + " %")

    # ----------------------- Calculate Personalisation Score -----------------------
    total_rec_list_vect = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(total_rec_list)
    similarity = cosine_similarity(X=total_rec_list_vect, dense_output=False)
    upper_right = np.triu_indices(similarity.shape[0], k=1)
    average_similarity = np.mean(similarity[upper_right])
    personalization = round((1 - average_similarity), 2) * 100
    # print("Personalization Score (average dissimilarity at top-" + str(topk) + " recommendations): ", personalization,
    #       "%")

    # ----------------------- Calculate Novelty Score -----------------------
    mean_self_info = []
    k = 0
    for sublist in total_rec_list:
        self_information = 0
        k += 1
        for i in sublist:
            i = i.decode("utf-8")
            self_information += np.sum(-np.log2(popularity_dict[i] / n_users))
        mean_self_info.append(self_information / topk)
    avg_novelty = round((sum(mean_self_info) / k), 2)
    # print("Mean Novelty score at top-" + str(topk) + " recommendations: ", avg_novelty)
    return coverage, personalization, avg_novelty, cold_prob


def eval_accuracy(model, cached_test):
    test_accuracy = model.evaluate(cached_test, return_dict=True, verbose=0)
    test_list = [test_accuracy['factorized_top_k/top_1_categorical_accuracy'],
                 test_accuracy['factorized_top_k/top_5_categorical_accuracy'],
                 test_accuracy['factorized_top_k/top_10_categorical_accuracy'],
                 test_accuracy['factorized_top_k/top_50_categorical_accuracy'],
                 test_accuracy['factorized_top_k/top_100_categorical_accuracy']]
    # print('Mean Factorized Top-K Testing accuracy', test_accuracy["factorized_top_k"] * 100,
    #       round(test_accuracy["factorized_top_k"].mean() * 100, 3))
    # print('Root Mean Squared Error: ', round(test_accuracy['root_mean_squared_error'], 4))

    from statistics import mean
    return [el * 100 for el in test_list], mean(test_list) * 100, test_accuracy['root_mean_squared_error']


def data_permutation(model, test):
    cached_test = test.batch(4096).cache()
    test_accuracy = model.evaluate(cached_test, return_dict=True, verbose=0)["factorized_top_k"]
    actual_score = test_accuracy.mean()

    for feature_batch in test.take(1):
        columns = list(feature_batch.keys())
    columns.pop(0)  # pop user_name
    columns.pop(0)  # pop album_name
    columns.pop(0)  # pop rating
    columns.pop(0)  # pop Genres

    data_perm = list(test.as_numpy_iterator())

    np.random.seed(42)
    random.seed(42)
    prep_score = []

    for _, col in enumerate(columns):
        # shuffle column
        shuff_test = data_perm.copy()
        shuff_test_df = pd.DataFrame(shuff_test)

        shuff_test_df[col].min()
        shuff_test_df[col].max()

        shuff_test_df[col] = np.random.randint(0, 1000, shuff_test_df.shape[0]) / 1000

        perm_test = tf.data.Dataset.from_tensor_slices(dict(shuff_test_df)).batch(4096).cache()

        # compute score
        test_accuracy = model.evaluate(perm_test, return_dict=True, verbose=0)["factorized_top_k"]
        score = test_accuracy.mean()
        prep_score.append(score)

    feature_imp = pd.DataFrame([(actual_score - x) for x in prep_score],
                               index=columns)
    feature_imp.columns = ['Feature Importance']
    return feature_imp
