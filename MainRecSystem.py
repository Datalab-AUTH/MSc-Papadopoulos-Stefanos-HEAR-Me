from RecSystem import TwoTowerNN, lightFM, CollaborativeFiltering, RandomPopular


def tune_rec_algorithms():
    # Get Random and 'Most Popular' Recommendations directly on the Test Set
    RandomPopular.run_rand_rec(data_sources=['UsersEmotions', 'UsersAll'])

    # Exhaustive parameter GridSearch for all three algorithms, on all possible feature combinations, for both datasets
    CollaborativeFiltering.trainCF(stage='tuning',
                                   data_sources=['UsersEmotions', 'UsersAll'],
                                   rating_columns=['Ratings', 'Ratings+Sentiment'])

    lightFM.train_lightFM(stage='tuning', data_sources=['UsersEmotions', 'UsersAll'])

    TwoTowerNN.train_2tnn(stage='tuning', data_sources=['UsersEmotions', 'UsersAll'])


def final_testing():
    # After the Fine Tuning phase, the best possible parameter combination for each dataset and feature combination
    # are set for the final testing on the Test Set.
    CollaborativeFiltering.trainCF(stage='final_testing',
                                   data_sources=['UsersEmotions', 'UsersAll'],
                                   rating_columns=['Ratings', 'Ratings+Sentiment'])
    lightFM.train_lightFM(stage='final_testing', data_sources=['UsersEmotions', 'UsersAll'])
    TwoTowerNN.train_2tnn(stage='final_testing', data_sources=['UsersEmotions', 'UsersAll'])

