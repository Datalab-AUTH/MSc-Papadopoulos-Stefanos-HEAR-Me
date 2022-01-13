import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp


def mean_factorized_score(actual_test, predicted_recs):

    # Calculate Mean Accuracy At K
    def calc_maak(actual, predicted, k):
        num_hits = 0.0

        for index, item in enumerate(actual):
            if item in predicted[index][:k]:
                num_hits += 1.0
        return num_hits / len(predicted)

    fs1 = calc_maak(actual_test, predicted_recs, 1)
    fs5 = calc_maak(actual_test, predicted_recs, 5)
    fs10 = calc_maak(actual_test, predicted_recs, 10)
    fs50 = calc_maak(actual_test, predicted_recs, 50)
    fs100 = calc_maak(actual_test, predicted_recs, 100)
    cf_mak = (fs1 + fs5 + fs10 + fs50 + fs100) / 5
    return fs1 * 100, fs5 * 100, fs10 * 100, fs50 * 100, fs100 * 100, cf_mak * 100


def calc_diversity(recs, data, unique_cold_items, popularity_dict, top_items=1200):
    sample_1K = random.sample(recs, top_items)

    cold_check_counter = 0
    for sample in sample_1K:
        nk = set(unique_cold_items).intersection(sample)
        cold_check_counter += len(nk)

    cold_prob = cold_check_counter / 1200

    sample_recs_10 = [l[:10] for l in sample_1K]
    flat_list = [item for sublist in sample_recs_10 for item in sublist]
    set_list = set(flat_list)
    coverage = len(set_list) / len(data['album_name'].unique()) * 100

    # ----------------------- Calculate Personalization Score -----------------------
    df = pd.DataFrame(data=sample_recs_10).reset_index().melt(id_vars='index', value_name='item')
    df = df[['index', 'item']].pivot(index='index', columns='item', values='item')
    df = pd.notna(df) * 1
    rec_matrix = sp.csr_matrix(df.values)
    similarity = cosine_similarity(X=rec_matrix, dense_output=False)
    upper_right = np.triu_indices(similarity.shape[0], k=1)
    average_similarity = np.mean(similarity[upper_right])
    personalization = (1 - average_similarity) * 100

    # ----------------------- Calculate Novelty Score -----------------------
    u = data['user_name'].unique().shape[0]
    mean_self_info = []
    k = 0
    for sublist in sample_recs_10:
        self_information = 0
        k += 1
        for i in sublist:
            self_information += np.sum(-np.log2(popularity_dict[i] / u))
        mean_self_info.append(self_information / 10)
    avg_novelty = round((sum(mean_self_info) / k), 2)
    return coverage, personalization, avg_novelty, cold_prob
