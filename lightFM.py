import time
from lightfm import LightFM
from sklearn.model_selection import ParameterGrid
from lightfm.data import Dataset
from RecSystem.RecUtilities.Results2Csv import results_to_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from RecSystem.RecUtilities.FetchData import fetch_df
from RecSystem.Recommenders.reco_utils.recommender.lightfm.lightfm_utils import track_model_metrics


def train_lightFM(stage, data_sources):
    if not (stage == 'tuning' or stage == 'final_testing'):
        return 'choose --- tuning or --- final_testing'

    if stage == 'tuning':
        tuning = True  # Return Validation Set
    else:
        tuning = False  # Return Testing Set

    for data_source in data_sources:

        data, filtered_data, train_set, test_set, cold_items, unique_cold_items, popularity_dict \
            = fetch_df(data_source=data_source, encode=False, k_bin=4, tuning=tuning)

        data_albums = data.drop_duplicates(subset=['album_name']).reset_index(drop=True)

        item_features_pd = data_albums[[
            'album_name',
            'Genres',
            'L_relaxed_mood',
            'L_happy_mood',
            'L_sad_mood',
            'L_angry_mood',
            'L_anger',
            'L_fear',
            'L_joy',
            'L_sadness',
            'L_surprise',
            'L_love',
            'L_sentiment',
            'M_valence',
            'M_arousal',
            'V_anger',
            'V_fear',
            'V_joy',
            'V_sadness',
            'V_surprise',
            'V_love',
            'V_sentiment',
            'followers_count',
            'A',
            'ANXIETY',
            'AVOIDANCE',
            'C',
            'E',
            'N',
            'O',
            'critic_t_rating',
            'audience_t_rating',
            'num_user_ratings',
            'V_view_count'
        ]]

        user_features_pd = data[[
            'album_name',
            'user_name',
            'r_anger',
            'r_joy',
            'r_love',
            'r_sadness',
            'r_surprise',
            'r_sentiment']]

        dataset = Dataset()
        dataset.fit(users=(data['user_name']),
                    items=(data_albums['album_name']),
                    user_features=user_features_pd.values.flatten(),
                    item_features=item_features_pd.values.flatten())

        n_users, n_items = dataset.interactions_shape()
        print('Num users: {}, num_items {}.'.format(n_users, n_items))

        u_f = dataset.build_user_features((x['user_name'], [
            x['album_name'],
            x['r_anger'],
            x['r_joy'],
            x['r_love'],
            x['r_sadness'],
            x['r_surprise'],
            x['r_sentiment']]) for i, x in data.iterrows())  # > Data.iterrows() / mean_user_feat.iterrows()

        i_f = dataset.build_item_features((x['album_name'], {
            x['Genres'],
            x['L_relaxed_mood'],
            x['L_happy_mood'],
            x['L_sad_mood'],
            x['L_angry_mood'],
            x['L_anger'],
            x['L_fear'],
            x['L_joy'],
            x['L_sadness'],
            x['L_surprise'],
            x['L_love'],
            x['L_sentiment'],
            x['M_valence'],
            x['M_arousal'],
            x['V_anger'],
            x['V_fear'],
            x['V_joy'],
            x['V_sadness'],
            x['V_surprise'],
            x['V_love'],
            x['V_sentiment'],
            x['followers_count'],
            x['A'],
            x['ANXIETY'],
            x['AVOIDANCE'],
            x['C'],
            x['E'],
            x['N'],
            x['O'],
            x['critic_t_rating'],
            x['audience_t_rating'],
            x['num_user_ratings'],
            x['V_view_count']
        }) for _, x in data_albums.iterrows())

        (train_interactions, _) = dataset.build_interactions(train_set[['user_name', 'album_name']].values)
        (test_interactions, _) = dataset.build_interactions(test_set[['user_name', 'album_name']].values)

        def plot_output(output_df):
            train_out = output_df[output_df['stage'] == 'train']
            train_recall = train_out[train_out['metric'] == 'Recall']

            test_out = output_df[output_df['stage'] == 'test']
            test_recall = test_out[test_out['metric'] == 'Recall']

            epoch_count = range(0, len(test_recall))
            plt.plot(epoch_count, train_recall['value'].values, 'r--', label='Train Recall')  #
            plt.plot(epoch_count, test_recall['value'].values, 'b-', label='Test Recall')  #
            plt.legend(['Test Recall'])
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout(False)
            plt.show()

        def train_lfm(viz, loss, components, num_epochs, lr, item_feat, user_feat, verbose, write):
            start_train_time = time.time()
            model = LightFM(no_components=components,
                            loss=loss,
                            learning_rate=lr)
            if not viz:
                model.fit(train_interactions,
                          epochs=num_epochs,
                          item_features=item_feat,
                          user_features=user_feat,
                          verbose=verbose)
                training_time = time.time() - start_train_time

                evaluation_start_time = time.time()
                evaluation_results = get_recs(model=model,
                                              eval_data=test_set,
                                              user_feat=user_feat,
                                              item_feat=item_feat)

                if user_feat is not None:
                    data_uf = 'user'
                else:
                    data_uf = 'none'

                if item_feat is not None:
                    data_if = 'item'
                else:
                    data_if = 'none'

                exp_results = {'data_comp': data_source+loss,
                               'model': 'lm_results',
                               'item_feat': data_if,
                               'user_feat': data_uf,
                               'components': components,
                               'num_epochs': num_epochs,
                               'learning_rate': lr
                               }
                exp_results.update(evaluation_results)

                cold_eval = get_recs(model=model,
                                     eval_data=cold_items,
                                     user_feat=user_feat,
                                     item_feat=item_feat)

                exp_results.update({'ColdAcc@10': cold_eval['Acc@10'],
                                    'ColdAcc@50': cold_eval['Acc@50'],
                                    'ColdAcc@100': cold_eval['Acc@100']
                                    })

                evaluation_time = time.time() - evaluation_start_time
                exp_results.update({'Training Time': round(training_time, 4),
                                    'Evaluation Time': round(evaluation_time, 4)
                                    })
                if write:
                    results_to_csv(exp_results)
                else:
                    print(exp_results)
            else:
                output, _ = track_model_metrics(model=model,
                                                train_interactions=train_interactions,
                                                test_interactions=test_interactions, k=100,
                                                user_features=user_feat,
                                                item_features=item_feat,
                                                no_epochs=num_epochs)
                plot_output(output)

        def get_recs(model, eval_data, user_feat, item_feat):
            k_items = 10  # Value set for Coverage/Personalization/Novelty
            total_rec_list = []
            covered_items = []
            count_interactions = 0
            cold_check_counter = 0

            num_hits = {'1': 0, '5': 0, '10': 0, '50': 0, '100': 0}

            for index, interaction in eval_data.iterrows():
                scores = pd.Series(model.predict(dataset._user_id_mapping[interaction['user_name']],
                                                 np.arange(n_items),
                                                 user_features=user_feat,
                                                 item_features=item_feat))
                scores.index = list(dataset._item_id_mapping.keys())
                scores = list(pd.Series(scores.sort_values(ascending=False).index))

                ground_truth = interaction['album_name']
                for tpk in num_hits.keys():
                    predicted_list = scores[0:int(tpk)]
                    if ground_truth in predicted_list:
                        num_hits[tpk] += 1
                    if tpk == str(k_items) and count_interactions < 1200:
                        total_rec_list.append(predicted_list)
                        for item in predicted_list:
                            covered_items.append(item)
                        count_interactions += 1
                    if tpk == str(100) and count_interactions < 1200:
                        nk = set(unique_cold_items).intersection(predicted_list)
                        cold_check_counter += len(nk)

            cold_prob = cold_check_counter / 1200

            for key in num_hits.keys():
                num_hits[key] = num_hits[key] / eval_data.shape[0]

            for key in num_hits.keys():
                num_hits[key] = round(num_hits[key] * 100, 4)

            # ----------------------- Calculate Coverage Rate -----------------------
            set_list = set(covered_items)
            unique_list = (list(set_list))
            item_coverage = round(len(unique_list) / n_items * 100, 2)

            # ----------------------- Calculate Personalisation Score -----------------------
            total_rec_list_vect = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(
                total_rec_list)
            similarity = cosine_similarity(X=total_rec_list_vect, dense_output=False)
            upper_right = np.triu_indices(similarity.shape[0], k=1)
            average_similarity = np.mean(similarity[upper_right])
            personalization = round((1 - average_similarity) * 100, 2)

            # ----------------------- Calculate Novelty Score -----------------------
            mean_self_info = []
            k = 0
            for sublist in total_rec_list:
                self_information = 0
                k += 1
                for i in sublist:
                    self_information += np.sum(-np.log2(popularity_dict[i] / n_users))
                mean_self_info.append(self_information / k_items)
            avg_novelty = round((sum(mean_self_info) / k), 2)

            MeanAcc = 0
            for key in num_hits.keys():
                MeanAcc += num_hits[key]
            MeanAcc = round(MeanAcc / len(num_hits), 4)

            eval_results = {'Acc@1': num_hits['1'],
                            'Acc@5': num_hits['5'],
                            'Acc@10': num_hits['10'],
                            'Acc@50': num_hits['50'],
                            'Acc@100': num_hits['100'],
                            'MeanAcc': MeanAcc,
                            'Coverage': item_coverage,
                            'Personalization': personalization,
                            'Novelty': avg_novelty,
                            'ColdProb': round(cold_prob, 4)
                            }

            return eval_results

        if stage == 'tuning':
            # Tuning of hyper-parameters phase
            params = {'feature_combination': [[None, None], [None, u_f], [i_f, None], [i_f, u_f]],
                      'lr': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                      'loss': ['warp', 'bpr'],
                      'num_epochs': [5, 10, 20],  # 5, 10,
                      'components': [10, 30, 50]
                      }

            parameter_grid = ParameterGrid(params)
            for parameters in parameter_grid:
                train_lfm(viz=False,
                          loss=parameters['loss'],
                          components=parameters['components'],
                          num_epochs=parameters['num_epochs'],
                          lr=parameters['lr'],
                          item_feat=parameters['feature_combination'][0],
                          user_feat=parameters['feature_combination'][1],
                          verbose=False,
                          write=True,  # Write to CSV
                          )
        elif stage == 'final_testing':
            print('Final Testing LightFM')
            # The best parameter combinations found in the tuning stage

            if data_source == 'UsersEmotions':
                # Item Features User Features, Components, Epochs, Learning Rate
                best_parameters = [
                    [[None, None], 30, 10, 0.05],
                    [[i_f, None], 50, 20, 0.15],
                    [[None, u_f], 10, 20, 0.15],
                    [[i_f, u_f], 10, 20, 0.3],
                ]
            else:
                best_parameters = [
                    [[None, None], 10, 20, 0.05],
                    [[i_f, None], 10, 20, 0.3],
                    [[None, u_f], 50, 20, 0.3],
                    [[i_f, u_f], 50, 20, 0.3],
                ]

            for parameters in best_parameters:
                train_lfm(viz=False,
                          loss='warp',
                          components=parameters[1],
                          num_epochs=parameters[2],
                          lr=parameters[3],
                          item_feat=parameters[0][0],
                          user_feat=parameters[0][1],
                          verbose=False,
                          write=True)