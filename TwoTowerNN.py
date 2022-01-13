import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.model_selection import ParameterGrid
from RecSystem.RecUtilities.Results2Csv import results_to_csv
from RecSystem.RecUtilities.helperTFRS import data_to_tf, eval_accuracy, data_permutation, eval_diversity, plot_history
from typing import Dict, Text
import time

tf.autograph.set_verbosity(10)
tf.autograph.experimental.do_not_convert()

# ------------------------------- Data Preparation ------------------------------- #
# Define the threshold for what is considers a Cold Item and a Cold User. Default is 1
# user_type : use 'IncludeAll' for using Raters as well as Reviewers


def train_2tnn(stage, data_sources):
    if not (stage == 'tuning' or stage == 'final_testing'):
        return 'choose --- tuning or --- final_testing'

    for data_source in data_sources:

        interactions, albums, train_interactions, _, val_interactions, \
        popularity_dict, cold_item_interactions, cold_item_albums, unique_cold_items \
            = data_to_tf(data_source=data_source, research_stage=stage)

        user_id_lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
        user_id_lookup.adapt(interactions.map(lambda x: x["user_name"]))
        unique_user_ids = np.unique(np.concatenate(list(interactions.batch(1000).map(lambda x: x["user_name"]))))

        album_title_lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
        album_title_lookup.adapt(albums.map(lambda x: x["album_name"]))
        unique_album_titles = np.unique(np.concatenate(list(albums.batch(1000).map(lambda x: x["album_name"]))))

        cached_train = train_interactions.batch(2048)
        cached_val = val_interactions.batch(4096).cache()
        cached_cold_items = cold_item_interactions.batch(4096).cache()

        def run_tfrs(layer_size_list, embedding_size, dropout_rate, lr, user_em, item_feat, epochs, plot, data_perm,
                     write):
            start_time = time.time()

            # ------------------------------- Model Definition ------------------------------- #
            class UserModel(tf.keras.Model):
                def __init__(self, use_user_emotions):
                    super().__init__()
                    self._use_user_emotions = use_user_emotions
                    self.user_embedding = tf.keras.Sequential([
                        tf.keras.layers.experimental.preprocessing.StringLookup(
                            vocabulary=unique_user_ids, mask_token=None),
                        tf.keras.layers.Embedding(len(unique_user_ids) + 2, embedding_size)])
                    if use_user_emotions:
                        self.normalized_user_emotions = tf.keras.layers.experimental.preprocessing.Normalization()

                def call(self, inputs):
                    if not self._use_user_emotions:
                        return self.user_embedding(inputs["user_name"])
                    else:
                        return tf.concat([self.user_embedding(inputs["user_name"]),
                                          self.normalized_user_emotions(inputs['r_anger']),
                                          self.normalized_user_emotions(inputs['r_joy']),
                                          self.normalized_user_emotions(inputs['r_love']),
                                          self.normalized_user_emotions(inputs['r_sadness']),
                                          self.normalized_user_emotions(inputs['r_surprise']),
                                          self.normalized_user_emotions(inputs["r_sentiment"])], axis=1)

            class RankingModel(tf.keras.Model):

                def __init__(self, layer_sizes):
                    super().__init__()

                    # Compute embeddings for users.
                    self.user_embeddings = tf.keras.Sequential([
                        tf.keras.layers.experimental.preprocessing.StringLookup(
                            vocabulary=unique_user_ids, mask_token=None),
                        tf.keras.layers.Embedding(len(unique_user_ids) + 2, embedding_size)
                    ])

                    # Compute embeddings for albums.
                    self.album_embedding = tf.keras.Sequential([
                        tf.keras.layers.experimental.preprocessing.StringLookup(
                            vocabulary=unique_album_titles, mask_token=None),
                        tf.keras.layers.Embedding(len(unique_album_titles) + 2, embedding_size)
                    ])

                    self.ratings = tf.keras.Sequential()

                    # Use the ReLU activation for all but the last layer
                    for layer_size in layer_sizes:
                        self.ratings.add(tf.keras.layers.Dense(layer_size, activation="relu"))

                    # No activation for the last layer.
                    self.ratings.add(tf.keras.layers.Dense(1))

                def call(self, inputs):
                    user_embedding = self.user_embeddings(inputs["user_name"])
                    album_embedding = self.album_embedding(inputs["album_name"])

                    return self.ratings(tf.concat([user_embedding, album_embedding], axis=1))

            class QueryModel(tf.keras.Model):
                def __init__(self, use_user_emotions, layer_sizes):
                    super().__init__()
                    self.use_user_emotions = use_user_emotions
                    self.embedding_model = UserModel(use_user_emotions)
                    self.dense_layers = tf.keras.Sequential()

                    # Use the ReLU activation for all but the last layer.
                    if len(layer_sizes) > 1:
                        for layer_size in layer_sizes[:-1]:
                            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))
                            # self.dense_layers.add(tf.keras.layers.Dropout(dropout_rate))

                    # No activation for the last layer.
                    for layer_size in layer_sizes[-1:]:
                        self.dense_layers.add(tf.keras.layers.Dense(layer_size))

                def call(self, inputs):
                    feature_embedding = self.embedding_model(inputs)
                    return self.dense_layers(feature_embedding)

            class AlbumModel(tf.keras.Model):

                def __init__(self, use_item_features):
                    super().__init__()

                    max_tokens = 10000
                    self._use_item_features = use_item_features
                    self.title_embedding = tf.keras.Sequential([
                        tf.keras.layers.experimental.preprocessing.StringLookup(
                            vocabulary=unique_album_titles, mask_token=None),
                        tf.keras.layers.Embedding(len(unique_album_titles) + 2, embedding_size)])

                    if use_item_features:
                        self.normalized_features = tf.keras.layers.experimental.preprocessing.Normalization()

                        def split_comma(input_data):
                            return tf.strings.split(input_data, sep=" / ")

                        self.genres_text_embedding = tf.keras.Sequential([
                            tf.keras.layers.experimental.preprocessing.TextVectorization(
                                max_tokens=max_tokens,
                                standardize=None,
                                split=split_comma),
                            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
                            # We average the embedding of individual words to get one embedding vector per genre.
                            tf.keras.layers.GlobalAveragePooling1D(),
                        ])

                def call(self, inputs):
                    if not self._use_item_features:
                        return self.title_embedding(inputs["album_name"])
                    elif self._use_item_features:
                        return tf.concat([
                            self.title_embedding(inputs['album_name']),
                            self.genres_text_embedding(inputs["Genres"]),
                            self.normalized_features(inputs['L_relaxed_mood']),
                            self.normalized_features(inputs['L_happy_mood']),
                            self.normalized_features(inputs['L_sad_mood']),
                            self.normalized_features(inputs['L_angry_mood']),
                            self.normalized_features(inputs['L_anger']),
                            self.normalized_features(inputs['L_fear']),
                            self.normalized_features(inputs['L_joy']),
                            self.normalized_features(inputs['L_sadness']),
                            self.normalized_features(inputs['L_surprise']),
                            self.normalized_features(inputs['L_love']),
                            self.normalized_features(inputs['L_sentiment']),
                            self.normalized_features(inputs['M_valence']),
                            self.normalized_features(inputs['M_arousal']),
                            self.normalized_features(inputs['V_anger']),
                            self.normalized_features(inputs['V_fear']),
                            self.normalized_features(inputs['V_joy']),
                            self.normalized_features(inputs['V_sadness']),
                            self.normalized_features(inputs['V_surprise']),
                            self.normalized_features(inputs['V_love']),
                            self.normalized_features(inputs['V_sentiment']),
                            self.normalized_features(inputs['V_view_count']),
                            self.normalized_features(inputs['critic_t_rating']),
                            self.normalized_features(inputs['audience_t_rating']),
                            self.normalized_features(inputs['num_user_ratings']),
                            self.normalized_features(inputs['followers_count']),
                            self.normalized_features(inputs['ANXIETY']),
                            self.normalized_features(inputs['AVOIDANCE']),
                            self.normalized_features(inputs['O']),
                            self.normalized_features(inputs['C']),
                            self.normalized_features(inputs['E']),
                            self.normalized_features(inputs['A']),
                            self.normalized_features(inputs['N'])], axis=1)

            class CandidateModel(tf.keras.Model):
                def __init__(self, use_item_features, layer_sizes):
                    super().__init__()
                    self.use_item_features = use_item_features
                    self.embedding_model = AlbumModel(use_item_features)
                    self.dense_layers = tf.keras.Sequential()

                    # Use the ReLU activation for all but the last layer.
                    if len(layer_sizes) > 1:
                        for layer_size in layer_sizes[:-1]:
                            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))
                            # self.dense_layers.add(tf.keras.layers.Dropout(dropout_rate))

                    # No activation for the last layer.
                    for layer_size in layer_sizes[-1:]:
                        self.dense_layers.add(tf.keras.layers.Dense(layer_size))

                def call(self, inputs):
                    feature_embedding = self.embedding_model(inputs)
                    return self.dense_layers(feature_embedding)

            class AoTY_Model(tfrs.models.Model):

                def __init__(self, use_user_emotions, use_item_features, layer_sizes, rating_weight: float,
                             retrieval_weight: float) -> None:
                    super().__init__()
                    self.query_model = QueryModel(use_user_emotions, layer_sizes)
                    self.candidate_model = CandidateModel(use_item_features, layer_sizes)
                    self.rating_model = RankingModel(layer_sizes)

                    # The tasks
                    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.RootMeanSquaredError()])
                    self.retrieval_task = tfrs.tasks.Retrieval(
                        metrics=tfrs.metrics.FactorizedTopK(candidates=albums.batch(128).map(self.candidate_model),
                                                            k=100))
                    # The loss weights.
                    self.rating_weight = rating_weight
                    self.retrieval_weight = retrieval_weight

                def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
                    query_embeddings = self.query_model({
                        "user_name": features["user_name"],
                        "r_anger": features["r_anger"],
                        "r_joy": features["r_joy"],
                        "r_love": features["r_love"],
                        "r_sadness": features["r_sadness"],
                        "r_surprise": features["r_surprise"],
                        "r_sentiment": features["r_sentiment"]
                    })
                    album_embeddings = self.candidate_model({
                        "album_name": features["album_name"],
                        "Genres": features["Genres"],
                        "L_relaxed_mood": features["L_relaxed_mood"],
                        "L_happy_mood": features["L_happy_mood"],
                        "L_sad_mood": features["L_sad_mood"],
                        "L_angry_mood": features["L_angry_mood"],
                        "L_anger": features["L_anger"],
                        "L_fear": features["L_fear"],
                        "L_joy": features["L_joy"],
                        "L_sadness": features["L_sadness"],
                        "L_surprise": features["L_surprise"],
                        "L_love": features["L_love"],
                        "L_sentiment": features["L_sentiment"],
                        "M_valence": features["M_valence"],
                        "M_arousal": features["M_arousal"],
                        "V_anger": features["V_anger"],
                        "V_fear": features["V_fear"],
                        "V_joy": features["V_joy"],
                        "V_sadness": features["V_sadness"],
                        "V_surprise": features["V_surprise"],
                        "V_love": features["V_love"],
                        "V_sentiment": features["V_sentiment"],
                        "V_view_count": features["V_view_count"],
                        "critic_t_rating": features["critic_t_rating"],
                        "audience_t_rating": features["audience_t_rating"],
                        "num_user_ratings": features["num_user_ratings"],
                        "followers_count": features["followers_count"],
                        "ANXIETY": features["ANXIETY"],
                        "AVOIDANCE": features["AVOIDANCE"],
                        "O": features["O"],
                        "C": features["C"],
                        "E": features["E"],
                        "A": features["A"],
                        "N": features["N"]
                    })

                    rating_predictions = self.rating_model({
                        "user_name": features["user_name"],
                        "album_name": features["album_name"],
                    })

                    rating_loss = self.rating_task(
                        labels=features["rating"],
                        predictions=rating_predictions)
                    retrieval_loss = self.retrieval_task(query_embeddings, album_embeddings)

                    # And combine them using the loss weights.
                    return self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss

            # ------------------------------- Model Training ------------------------------- #
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_factorized_top_k/top_100_categorical_accuracy',
                # monitor='val_loss',
                patience=2,
                mode='auto',
                restore_best_weights=True)

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_factorized_top_k/top_100_categorical_accuracy',
                factor=0.5,
                patience=3,
                verbose=0,
                mode='auto',
                min_delta=0.0001,
                cooldown=0,
                min_lr=0)

            model = AoTY_Model(use_user_emotions=user_em,
                               use_item_features=item_feat,
                               layer_sizes=layer_size_list,
                               rating_weight=1.0,
                               retrieval_weight=1.0)

            model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=lr))
            # model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr))
            # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

            history = model.fit(cached_train,
                                epochs=epochs,
                                verbose=0,
                                validation_data=cached_val,
                                callbacks=[early_stop, reduce_lr],
                                shuffle=True
                                )
            training_time = round(time.time() - start_time, 4)

            # ------------------------------- Evaluation ------------------------------- #
            start_time = time.time()
            # Evaluation based on Mean Top-K accuracy and RMSE score
            if plot:
                plot_history(history, choice='topk')  # choice can be : 'topk' / 'rmse' / 'both'

            # Calculate Accuracy and RMSE
            accuracy_list, mean_accuracy, rmse_score = eval_accuracy(model, cached_val)

            # Evaluation based on Item Coverage, Personalization and Novelty
            coverage_score, personalization_score, novelty_score, cold_prob = eval_diversity(model,
                                                                                             val_interactions,
                                                                                             albums,
                                                                                             popularity_dict=popularity_dict,
                                                                                             n_users=len(
                                                                                                 unique_user_ids),
                                                                                             cold_list=unique_cold_items)
            evaluation_time = round(time.time() - start_time, 4)

            # Evaluation of Cold Users and Items
            cold_accuracy_list, cold_mean_accuracy, cold_rmse = eval_accuracy(model, cached_cold_items)

            if user_em:
                user_features = 'Yes'
            else:
                user_features = 'No'
            if item_feat:
                item_features = 'Yes'
            else:
                item_features = 'No'

            exp_results = {'data_comp': data_source,
                           'model': '2tnn_results',
                           'item_feat': item_features,
                           'user_feat': user_features,
                           'layer_size_list': layer_size_list,
                           'embedding_size': embedding_size,
                           'learning_rate': lr,
                           'epochs': len(history.history['loss']),
                           'Acc@1': round(accuracy_list[0], 4),
                           'Acc@5': round(accuracy_list[1], 4),
                           'Acc@10': round(accuracy_list[2], 4),
                           'Acc@50': round(accuracy_list[3], 4),
                           'Acc@100': round(accuracy_list[4], 4),
                           'MeanAcc': round(mean_accuracy, 4),
                           'RMSE': round(rmse_score, 4),
                           'Coverage': round(coverage_score, 2),
                           'Personalisation': personalization_score,
                           'Novelty': novelty_score,
                           'ColdProb': round(cold_prob, 4),
                           'ColdAcc@10': round(cold_accuracy_list[2], 4),
                           'ColdAcc@50': round(cold_accuracy_list[3], 4),
                           'ColdAcc@100': round(cold_accuracy_list[4], 4),
                           'Training Time': training_time,
                           'Evaluation Time': evaluation_time
                           }
            if write:
                results_to_csv(exp_results)
            else:
                print(exp_results)

            # Feature Importance based on Data permutation
            if data_perm:
                data_permutation(model, val_interactions)

            return history.history

        if stage == 'tuning':
            params = {'feature_combination': [[False, False], [False, True], [True, False], [True, True]],
                      # [USER FEATURES , ITEM FEATURES]
                      'lr': [0.01, 0.05, 0.1, 0.15, 0.2],
                      'num_epochs': [20],  # with Early Stopping
                      'embeddings_size': [16, 32, 64],
                      'layer_size_list': [[16], [32], [32, 64], [32, 64, 128]],
                      }

            parameter_grid = ParameterGrid(params)

            # Hyper Parameter tuning phase
            for parameters in parameter_grid:
                run_tfrs(layer_size_list=parameters['layer_size_list'],
                         embedding_size=parameters['embeddings_size'],
                         dropout_rate=0,
                         lr=parameters['lr'],
                         user_em=parameters['feature_combination'][0],
                         item_feat=parameters['feature_combination'][1],
                         epochs=parameters['num_epochs'],
                         plot=False,
                         data_perm=False,
                         write=True)
        else:
            if data_source == 'UsersEmotions':
                # User Features, Item Features, Layer Size, Embedding Size, Learning Rate
                best_parameters = [
                    [[False, False], [32], 64, 0.01],
                    [[True, False], [16], 32, 0.1],
                    [[False, True], [32], 32, 0.1],
                    [[True, True], [16], 32, 0.1],
                ]
            else:
                best_parameters = [
                    [[False, False], [32], 32, 0.1],
                    [[True, False], [16], 32, 0.1],
                    [[False, True], [16], 16, 0.1],
                    [[True, True], [32], 16, 0.1],
                ]

            for parameters in best_parameters:
                run_tfrs(layer_size_list=parameters[1],
                         embedding_size=parameters[2],
                         dropout_rate=0,
                         lr=parameters[3],
                         user_em=parameters[0][0],
                         item_feat=parameters[0][1],
                         epochs=20,
                         plot=True,
                         data_perm=False,
                         write=True)