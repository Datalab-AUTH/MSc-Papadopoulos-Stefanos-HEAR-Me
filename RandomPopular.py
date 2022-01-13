from RecSystem.RecUtilities.Results2Csv import results_to_csv
from RecSystem.RecUtilities.FetchData import fetch_df
from RecSystem.RecUtilities.BaselineEvaluation import mean_factorized_score, calc_diversity


def run_rand_rec(data_sources):

    for data_source in data_sources:
        data, _, _, test_set, cold_items, unique_cold_items, popularity_dict \
            = fetch_df(data_source=data_source, encode=True, k_bin=0, tuning=True)

        # ----------------------- Popularity Recommendations -----------------------
        def popular_recs():
            popularity_recs = data['album_name'].value_counts().head(100).index.tolist()

            pop_recs = []
            for _ in range(len(test_set)):
                pop_recs.append(popularity_recs)

            p_acc1, p_acc5, p_acc10, p_acc50, p_acc100, p_macc = mean_factorized_score(test_set['album_name'], pop_recs)
            pop_coverage_score, pop_personalization_score, pop_novelty_score, pop_cold_prob = \
                calc_diversity(pop_recs,
                               data,
                               unique_cold_items,
                               popularity_dict)

            cold_i_pop_recs = []
            for _ in range(len(cold_items)):
                cold_i_pop_recs.append(popularity_recs)

            cold1, cold5, cold10, cold50, cold100, _ = mean_factorized_score(cold_items['album_name'], cold_i_pop_recs)

            results_to_csv({'data_comp': data_source,
                            'model': 'pop_rand',
                            'current': 'popularity',
                            'Acc@1': round(p_acc1, 4),
                            'Acc@5': round(p_acc5, 4),
                            'Acc@10': round(p_acc10, 4),
                            'Acc@50': round(p_acc50, 4),
                            'Acc@100': round(p_acc100, 4),
                            'MeanAcc': round(p_macc, 4),
                            'Coverage': round(pop_coverage_score, 2),
                            'Personalization': round(pop_personalization_score, 2),
                            'Novelty': round(pop_novelty_score, 2),
                            'ColdProb': round(pop_cold_prob, 2),
                            'Cold@10': round(cold10, 2),
                            'Cold@50': round(cold50, 2),
                            'Cold@100': round(cold100, 2)
                            })
        # ----------------------- Random Recommendations -----------------------
        def random_recs():
            album_data = data.drop_duplicates(subset=['album_name']).reset_index(drop=True)

            rand_recs = []
            for _ in range(len(test_set)):
                random_predictions = album_data['album_name'].sample(100).values.tolist()
                rand_recs.append(random_predictions)

            r_acc1, r_acc5, r_acc10, r_acc50, r_acc100, r_macc = mean_factorized_score(test_set['album_name'], rand_recs)
            rand_coverage_score, rand_personalization_score, rand_novelty_score, rand_cold_prob = \
                calc_diversity(rand_recs,
                               data,
                               unique_cold_items,
                               popularity_dict,
                               )
            print(rand_coverage_score, rand_personalization_score, rand_novelty_score)

            cold_i_rand_recs = []
            for _ in range(len(cold_items)):
                random_predictions = album_data['album_name'].sample(100).values.tolist()
                cold_i_rand_recs.append(random_predictions)

            cold1, cold5, cold10, cold50, cold100, _ = mean_factorized_score(cold_items['album_name'], cold_i_rand_recs)

            results_to_csv({'data_comp': data_source,
                            'model': 'pop_rand',
                            'current': 'random',
                            'Acc@1': round(r_acc1, 4),
                            'Acc@5': round(r_acc5, 4),
                            'Acc@10': round(r_acc10, 4),
                            'Acc@50': round(r_acc50, 4),
                            'Acc@100': round(r_acc100, 4),
                            'MeanAcc': round(r_macc, 4),
                            'Coverage': round(rand_coverage_score, 2),
                            'Personalization': round(rand_personalization_score, 2),
                            'Novelty': round(rand_novelty_score, 2),
                            'ColdProb': round(rand_cold_prob, 2),
                            'Cold@10': round(cold10, 2),
                            'Cold@50': round(cold50, 2),
                            'Cold@100': round(cold100, 2)
                            })

        popular_recs()
        random_recs()