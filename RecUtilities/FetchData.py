import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from RecSystem.Recommenders.reco_utils.dataset.python_splitters import python_stratified_split

def fetch_df(data_source, encode, k_bin, tuning):
    if data_source not in ['UsersEmotions', 'UsersAll']:
        raise Exception("Data source can be either : 'UsersEmotions' or 'UsersAll' ")

    print('Loading the Dataset...')
    data = pd.read_csv('../Data/'+data_source + '.csv', index_col=0, sep=';')
    data.Genres = data.Genres.fillna('-')

    print(f'Total count of Interactions : {data.shape[0]}')
    print(f'Total count of users : {data.user_name.unique().shape[0]}')
    print(f'Total count of albums : {data.album_name.unique().shape[0]}')

    if data_source == 'UsersEmotions':
        cold_threshold = 2
    else:
        cold_threshold = 20

    if encode:
        data['user_name'] = LabelEncoder().fit_transform(data['user_name'])
        data['album_name'] = LabelEncoder().fit_transform(data['album_name'])

    popularity_dict = dict(data.album_name.value_counts())

    if k_bin > 0:
        print('Transform into KBins')
        est = KBinsDiscretizer(n_bins=k_bin, encode='ordinal', strategy='uniform')
        bin_col = data.columns.difference(['user_name', 'album_name', 'artist_name', 'Genres', 'rating'])
        data[bin_col] = pd.DataFrame(est.fit_transform(data[bin_col]))

    print(f'Filter and Collect Cold Items below {cold_threshold} interactions')
    filtered_cold_items = data.groupby('album_name').filter(lambda x: len(x) <= cold_threshold)
    unique_cold_items = filtered_cold_items.album_name.unique()
    print(f"Cold Albums : {len(unique_cold_items)}")

    filtered_data = data.groupby('album_name').filter(lambda x: len(x) > cold_threshold)

    print('Stratified Split of the dataset into Train / Test')
    train_set, val_set = python_stratified_split(
        filtered_data, filter_by='user', min_rating=1, ratio=0.85,
        col_user='user_name', col_item='album_name', seed=42)

    train_set, final_test_set = python_stratified_split(
        train_set, filter_by='user', min_rating=1, ratio=0.85,
        col_user='user_name', col_item='album_name', seed=42)

    '''
    Split 70 / 15 / 15 for both sets
    
    UsersEmotions
        train / val / test
        46542, 8884, 7302 
    
    UsersAll
        train  / val / test
        265592 / 54291 / 45689
        
    # To ensure that there are no cold users in the test set are also on the train set! 
    trs = list(train_set.user_name.unique())
    vs =  list(val_set.user_name.unique())
    tst = list(final_test_set.user_name.unique())
    
    inter_vs = len(set(trs).intersection(vs)) 
    inter_tst = len(set(trs).intersection(tst))      
     
    if inter_vs == len(vs) and inter_tst == len(tst):
        print('No cold users are present') 
    else:
        print('Some cold users were found')
    '''

    print('Randomly Shuffle all Data')
    data = data.sample(frac=1, random_state=42)
    filtered_data = filtered_data.sample(frac=1, random_state=42)
    train_set = train_set.sample(frac=1, random_state=42)
    val_set = val_set.sample(frac=1, random_state=42)
    final_test_set = final_test_set.sample(frac=1, random_state=42)

    print('Round Data')
    data = data.round(3)
    filtered_data = filtered_data.round(3)
    train_set = train_set.round(3)
    val_set = val_set.round(3)
    final_test_set = final_test_set.round(3)

    if tuning:
        return data, filtered_data, train_set, val_set, filtered_cold_items, unique_cold_items, popularity_dict
    else:
        return data, filtered_data, train_set, final_test_set, filtered_cold_items, unique_cold_items, popularity_dict
