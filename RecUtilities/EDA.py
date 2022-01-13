import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import recmetrics
from collections import Counter
import numpy as np
import statistics

data = pd.read_csv('../../Data/UsersAll.csv', index_col=0, sep=';')
data['Genres'] = data['Genres'].str.split(" / ")

new_col_name = {'rating': 'Ratings',
                'r_anger': 'Review Anger',
                'r_joy': 'Review Joy',
                'r_love': 'Review Love',
                'r_sadness': 'Review Sadness',
                'r_surprise': 'Review Surprise',
                'r_sentiment': 'Review Sentiment',
                'critic_t_rating': 'Average Critic Rating',
                'audience_t_rating': 'Average User Rating',
                'num_user_ratings': 'Count of User Ratings',
                'V_view_count': 'Album Views Counts',
                'L_relaxed_mood': 'Lyric Relaxed Mood',
                'L_happy_mood': 'Lyrics Happy Mood',
                'L_sad_mood': 'Lyrics Sad Mood',
                'L_angry_mood': 'Lyrics Angry Mood',
                'L_anger': 'Lyrics Anger',
                'L_fear': 'Lyrics Fear',
                'L_joy': 'Lyrics Joy',
                'L_sadness': 'Lyrics Sadness',
                'L_surprise': 'Lyrics Surprise',
                'L_love': 'Lyrics Love',
                'L_sentiment': 'Lyrics Sentiment',
                'M_valence': 'Music Valence',
                'M_arousal': 'Music Arousal',
                'V_anger': 'Comments Anger',
                'V_fear': 'Comments Fear',
                'V_joy': 'Comments Joy',
                'V_sadness': 'Comments Sadness',
                'V_surprise': 'Comments Surprise',
                'V_love': 'Comments Love',
                'V_sentiment': 'Comments Sentiment',
                'followers_count': 'Twitter Followers'
                }

data = data.rename(columns=new_col_name)
data_albums = data.drop_duplicates(subset=['album_name']).reset_index(drop=True)
data_em = pd.read_csv('../../Data/UsersEmotions.csv', index_col=0, sep=';')
data_em = data_em.rename(columns=new_col_name)
mean_user_em = data_em.groupby("user_name").mean()


def statistical():
    album_stat = data.album_name.value_counts()
    print(album_stat.describe())
    print(statistics.median(album_stat))

    '''
    count      1997
    mean        185.162744
    std         359.730166
    min           1
    25 %         30
    50 %         69
    75 %        128
    max        3698
    median       69
    '''
    user_stat = data.user_name.value_counts()
    print(user_stat.describe())
    print(statistics.median(user_stat))

    '''
    count     12981
    mean         28.485479
    std          57.913615
    min           1
    25 %          3
    50 %          9
    75 %         29
    max        1423
    median        9
    '''


def data_sparsity():
    matrix = data.pivot_table(index='user_name', columns='album_name', values='Ratings').fillna(0).to_numpy()
    sparsity = 1.0 - (np.count_nonzero(matrix) / float(matrix.size))

    values = [int(matrix.size), np.count_nonzero(matrix)]
    labels = ['Possible Interactions', "Known Interactions"]

    plt.figure(figsize=(5, 5))
    plt.bar(labels, values, width=0.3, color='cornflowerblue', align='center')
    plt.title('Data Sparsity: ' + str(round(sparsity * 100, 2)) + '%')
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()

    em_values = [data.shape[0] - data_em.shape[0], data_em.shape[0]]
    em_labels = ["Missing Emotional Values", "Emotional Interactions"]
    em_sparsity = 1.0 - (data_em.shape[0] / (data.shape[0] - data_em.shape[0]))

    plt.figure(figsize=(5, 5))
    plt.bar(em_labels, em_values, width=0.3, color='coral', align='center')
    plt.title('Emotional Sparsity: ' + str(round(em_sparsity * 100, 2)) + '%')
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()


def tail_exploration():
    df = data.rename(columns={'album_name': 'Albums'})
    fig = plt.figure()
    recmetrics.long_tail_plot(df=df,
                              item_id_column="Albums",
                              interaction_type="Interactions",
                              percentage=0.5,
                              x_labels=False)
    fig.show()


def corr_data():
    def viz_cor(input_data, title, corr_method):
        corr = input_data.corr(corr_method)
        corr_matrix = np.tril(corr)
        plt.figure(figsize=(10, 6))
        chart = sn.heatmap(corr,
                           annot=True,
                           mask=corr_matrix,
                           fmt='.1g',
                           vmin=-1,
                           vmax=1,
                           center=0,
                           cmap='coolwarm',
                           linewidths=1,
                           linecolor='whitesmoke'
                           )
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title(title)
        plt.show()

    # ---------- Emotion / Rating ----------
    viz_cor(input_data=data_em[['Ratings', 'Review Anger', 'Review Joy', 'Review Love', 'Review Sadness',
                                'Review Surprise', 'Review Sentiment']],
            title='Emotions and Ratings',
            corr_method='spearman')

    # ---------- Popularity Metrics and Ratings ----------
    viz_cor(input_data=data[['Ratings', 'Average Critic Rating',
                             'Average User Rating', 'Count of User Ratings', 'Album Views Counts',
                             'Twitter Followers']],
            title='Popularity and User Ratings',
            corr_method='spearman')

    # ---------- Reviews vs YT Comments ----------
    viz_cor(input_data=data_em[['Ratings', 'Review Anger', 'Review Joy', 'Review Love', 'Review Sadness',
                                'Review Surprise', 'Review Sentiment',
                                'Comments Anger', 'Comments Fear',
                                'Comments Joy', 'Comments Sadness', 'Comments Surprise',
                                'Comments Love', 'Comments Sentiment']],
            title='Review Emotions and Comments Emotions',
            corr_method='spearman')

    viz_cor(input_data=data_albums[['O', 'C', 'E', 'A', 'N', 'ANXIETY', 'AVOIDANCE']],
            title='Ratings and Artist Personality', corr_method='spearman')

    viz_cor(input_data=data_albums[['Music Valence',
                                    'Music Arousal',
                                    'Lyric Relaxed Mood',
                                    'Lyrics Happy Mood',
                                    'Lyrics Sad Mood',
                                    'Lyrics Angry Mood'
                                    ]],
            title='Music Mood and Lyrics Emotions',
            corr_method='pearson')

def reviews_vs_comments():
    # --------------- Review Emotions vs Youtube Emotions ---------------

    (data_em * 100).hist(column=['Review Anger', 'Review Joy', 'Review Love', 'Review Sadness',
                                 'Review Surprise', 'Review Sentiment'],
                         sharex=True,
                         sharey=True,
                         bins=6,
                         rwidth=0.95,
                         color='coral')
    plt.show()

    (data_em * 100).hist(column=['Comments Anger',
                                 'Comments Joy', 'Comments Sadness', 'Comments Surprise',
                                 'Comments Love', 'Comments Sentiment'],
                         sharex=True,
                         sharey=True,
                         bins=6,
                         rwidth=0.95,
                         color='#86bf91')
    plt.show()

    (mean_user_em * 100).hist(column=['Review Anger', 'Review Joy', 'Review Love', 'Review Sadness',
                                      'Review Surprise', 'Review Sentiment'],
                              sharex=True,
                              sharey=True,
                              bins=3,
                              rwidth=0.95,
                              color='cornflowerblue')
    plt.show()

    (data_albums * 100).hist(column=['Comments Anger',
                                     'Comments Joy', 'Comments Sadness', 'Comments Surprise',
                                     'Comments Love', 'Comments Sentiment'],
                             sharex=True,
                             sharey=True,
                             rwidth=0.95,
                             bins=3,
                             color='#86bf91')
    plt.show()


def viz_ratings_sentiment():
    (data.Ratings * 100).plot.hist(bins=10, rwidth=0.95, color='cornflowerblue')
    plt.title('Ratings Distribution')
    plt.show()

    (data_em['Review Sentiment'] * 100).plot.hist(bins=10, rwidth=0.95, color='#86bf91')
    plt.title('Sentiment Distribution')
    plt.show()

    (data_em * 100).hist(column=['Ratings', 'Review Sentiment'],
                         sharex=True,
                         sharey=True,
                         bins=6,
                         rwidth=0.95,
                         color='cornflowerblue')
    plt.show()

    (data_em[['Ratings', 'Review Sentiment']] * 100).plot.hist(bins=10, rwidth=0.95, color=['cornflowerblue', 'coral'],
                                                               alpha=0.4)
    plt.title('Ratings vs Sentiment')
    plt.show()


def viz_mev():
    cols = ['Review Sentiment', 'Review Joy', 'Review Love', 'Review Anger', 'Review Sadness', 'Review Surprise']
    col_counter = 0
    x = 2
    y = 3
    fig, axs = plt.subplots(x, y, figsize=(12, 10))
    # fig.suptitle('Missing Emotional Values', fontsize=10)

    for i in range(x):
        for j in range(y):
            axs[i][j].hist(data[cols[col_counter]], bins=4, color='cornflowerblue', label='Full Data', alpha=0.6,
                           rwidth=0.95)
            axs[i][j].hist(data_em[cols[col_counter]], bins=4, color='coral', label='Reviews', alpha=0.6, rwidth=0.95)
            axs[i][j].set_title(cols[col_counter])
            if i == 1:
                axs[i][j].set_xlabel('Value')
            if j == 0:
                axs[i][j].set_ylabel('Frequency')
            col_counter += 1
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    fig.show()


def viz_genres(num=20):
    def viz_df(df, plot_title, color):
        flat_df = [item for sublist in df for item in sublist]
        pop_list = Counter(flat_df).most_common(num)
        pop_df = pd.DataFrame(pop_list)
        pop_df.columns = ['Name', 'Genre']
        pop_df = pop_df.set_index('Name')
        pop_df.plot.barh(color=color)
        plt.title(plot_title)
        plt.xlabel('Count')
        plt.ylabel('Genre')
        plt.legend().set_visible(False)
        plt.show()

    interactions_genre = data.Genres.dropna()
    viz_df(interactions_genre, 'Most Interacted Genres', color='cornflowerblue')

    data_albums = data.drop_duplicates(subset=['album_name']).reset_index(drop=True)
    data_genres = data_albums.Genres.dropna()
    viz_df(data_genres, 'Most Frequent Album Genres', color='#86bf91')