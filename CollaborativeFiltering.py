import time
import pandas as pd
# import recmetrics
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.knns import KNNBasic
from sklearn.model_selection import ParameterGrid
from surprise import Reader, Dataset
from surprise.dataset import DatasetAutoFolds
from RecSystem.RecUtilities.Results2Csv import results_to_csv
from RecSystem.RecUtilities.FetchData import fetch_df
from surprise.accuracy import rmse
from RecSystem.RecUtilities.BaselineEvaluation import mean_factorized_score, calc_diversity


def trainCF(stage, data_sources, rating_columns):
    if not (stage == 'tuning' or stage == 'final_testing'):
        return 'choose --- tuning or --- final_testing'

    if stage == 'tuning':
        tuning = True  # Return Validation Set
    else:
        tuning = False  # Return Testing Set

    for data_source in data_sources:
        data, _, train_data, test_data, cold_items, unique_cold_items, popularity_dict \
            = fetch_df(data_source=data_source, encode=True, k_bin=0, tuning=tuning)

        for rating_column in rating_columns:
            if rating_column == 'Ratings':
                rating_type = 'rating'
            elif rating_column == 'Ratings+Sentiment':

                def conditional_combine(df):
                    import statistics
                    if data_source == 'UsersAll':
                        df['rating_comb'] = df[['rating', 'r_sentiment']].mean(axis=1).where(
                            df['r_sentiment'] != statistics.median(data.r_sentiment),
                            df['rating'])
                    else:
                        df['rating_comb'] = df[['rating', 'r_sentiment']].mean(axis=1)

                    df['rating_comb'] = df['rating_comb'].round(3)
                    return df

                data = conditional_combine(data)
                train_data = conditional_combine(train_data)
                test_data = conditional_combine(test_data)
                cold_items = conditional_combine(cold_items)
                rating_type = 'rating_comb'
            else:
                return 'Choose between Ratings or Ratings+Sentiment as the rating column'

            scale = (data[rating_type].min(), data[rating_type].max())
            reader = Reader(rating_scale=scale)

            train_set = Dataset.load_from_df(train_data[['user_name', 'album_name', rating_type]], reader=reader)
            train_set = DatasetAutoFolds.build_full_trainset(train_set)

            test_set = list(test_data[['user_name', 'album_name', rating_type]].to_records(index=False))

            # ----------------------- Model CF -----------------------
            def make_predictions(user_id, k, model):
                recommended_items = pd.DataFrame(model.loc[user_id])
                recommended_items.columns = ["predicted_rating"]
                recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)
                recommended_items = recommended_items.head(k)
                return recommended_items.index.tolist()

            def train_baseline(model, input_test_set, exp_results_dict):
                start_time = time.time()
                model.fit(train_set)
                training_time = round(time.time() - start_time, 4)
                start_time = time.time()

                test_cf = model.test(input_test_set)
                rmse_score = rmse(test_cf)

                full_test_pivot = data.pivot_table(index='user_name', columns='album_name', values=rating_type).fillna(
                    0)
                test_stack_df = full_test_pivot.stack().reset_index()
                test_stack_df = test_stack_df.rename(columns={0: 'actual'})
                test_cf = model.test(list(test_stack_df.to_records(index=False)))
                del test_stack_df
                test_cf = pd.DataFrame(test_cf)
                test_cf.drop("details", inplace=True, axis=1)
                test_cf.columns = ['user_name', 'album_name', 'actual', 'cf_predictions']
                cf_model = test_cf.pivot_table(index='user_name', columns='album_name', values='cf_predictions').fillna(
                    0)
                del test_cf

                test_set_cf = pd.DataFrame.from_records(input_test_set,
                                                        columns=['user_name', 'album_name', rating_type])
                cf_recs = []
                for user in test_set_cf['user_name']:
                    cf_predictions = make_predictions(user, 100, cf_model)
                    cf_recs.append(cf_predictions)

                acc1, acc5, acc10, acc50, acc100, macc = mean_factorized_score(test_set_cf['album_name'], cf_recs)
                coverage_score, personalization_score, novelty_score, cold_prob = \
                    calc_diversity(cf_recs,
                                   data,
                                   unique_cold_items,
                                   popularity_dict)

                cold_cf_recs = []
                for user in cold_items['user_name']:
                    try:
                        cf_predictions = make_predictions(user, 100, cf_model)
                        cold_cf_recs.append(cf_predictions)
                    except:
                        cold_cf_recs.append([])

                _, _, cold_acc10, cold_acc50, cold_acc100, _ = mean_factorized_score(cold_items['album_name'],
                                                                                     cold_cf_recs)

                evaluation_time = round(time.time() - start_time, 4)
                exp_results_dict.update({
                    'Acc@1': round(acc1, 4),
                    'Acc@5': round(acc5, 4),
                    'Acc@10': round(acc10, 4),
                    'Acc@50': round(acc50, 4),
                    'Acc@100': round(acc100, 4),
                    'MeanAcc': round(macc, 4),
                    'RMSE': round(rmse_score, 4),
                    'Coverage': round(coverage_score, 2),
                    'Personalization': round(personalization_score, 2),
                    'Novelty': round(novelty_score, 2),
                    'ColdProb': round(cold_prob, 4),
                    'ColdAcc@10': round(cold_acc10, 4),
                    'ColdAcc@50': round(cold_acc50, 4),
                    'ColdAcc@100': round(cold_acc100, 4),
                    'TrainingTime': training_time,
                    'EvaluationTime': evaluation_time
                })

                return exp_results_dict

            if stage == 'tuning':
                params = {'lr_all': [0.0005, 0.001, 0.01, 0.1],
                          'num_epochs': [10, 20],
                          'factors': [10, 20]}

                parameter_grid = ParameterGrid(params)
                for parameters in parameter_grid:
                    exp_results = {'data_comp': data_source + rating_column,
                                   'model': 'svd',
                                   'num_factors': parameters['factors'],
                                   'num_epochs': parameters['num_epochs'],
                                   'learning_rate': parameters['lr_all']}
                    svd = SVD(n_factors=parameters['factors'], n_epochs=parameters['num_epochs'],
                              lr_all=parameters['lr_all'])
                    exp_results = train_baseline(model=svd, input_test_set=test_set, exp_results_dict=exp_results)
                    results_to_csv(exp_results)
            else:
                # Factors, Epochs, Learning Rate
                best_parameters = [10, 10,
                                   0.001]  # for Both datasets this was the best performing parameter combination
                exp_results = {'data_comp': data_source + rating_column,
                               'model': 'svd',
                               'num_factors': best_parameters[0],
                               'num_epochs': best_parameters[1],
                               'learning_rate': best_parameters[2]}
                svd = SVD(n_factors=best_parameters[0], n_epochs=best_parameters[1], lr_all=best_parameters[2])
                exp_results = train_baseline(model=svd, input_test_set=test_set, exp_results_dict=exp_results)
                results_to_csv(exp_results)

            if stage == 'tuning':
                # ----- Hyper-parameter tuning for kNN -----
                params = {'k': [10, 50, 100],
                          'metric': ['MSD', 'cosine'],
                          'user_based': [True, False]}
                parameter_grid = ParameterGrid(params)
                for parameters in parameter_grid:
                    exp_results = {'data_comp': data_source + rating_column,
                                   'model': 'kNN + ' + parameters['metric'],
                                   'k': parameters['k'],
                                   'dist_metric': parameters['metric'],
                                   'user_based': parameters['user_based']}
                    knn = KNNBasic(k=parameters['k'], sim_options={'name': parameters['metric'],
                                                                   'user_based': parameters['user_based']},
                                   verbose=False)
                    exp_results = train_baseline(model=knn, input_test_set=test_set, exp_results_dict=exp_results)
                    results_to_csv(exp_results)
            else:
                # K, Distance Metric, User-Based
                best_parameters = [10, 'MSD', True]
                exp_results = {'data_comp': data_source + rating_column,
                               'model': 'kNN + ' + best_parameters[1],
                               'k': best_parameters[0],
                               'dist_metric': best_parameters[1],
                               'user_based': best_parameters[2]}
                knn = KNNBasic(k=best_parameters[0], sim_options={'name': best_parameters[1],
                                                                  'user_based': best_parameters[2]}, verbose=False)
                exp_results = train_baseline(model=knn, input_test_set=test_set, exp_results_dict=exp_results)
                results_to_csv(exp_results)
