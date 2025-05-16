import pandas as pd
import numpy as np

def prepare_features(df, dislike_file):
    df = pd.get_dummies(df, columns=['sentiment'], prefix='comment_class')
    sentiment_sums = df.groupby('videoId')[[
        'comment_class_negative', 'comment_class_neutral', 'comment_class_positive'
    ]].sum()
    ratios = df[['videoId', 'norm_ratio']].groupby('videoId').max()
    train_df = pd.merge(sentiment_sums, ratios, on='videoId')

    dislikes = pd.read_csv(dislike_file)
    dislikes['like-view'] = dislikes['likes'] / dislikes['view_count']
    like_view = dislikes[['video_id', 'like-view']]
    train_df = train_df.merge(like_view, left_index=True, right_on='video_id', how='left')

    train_df['total_comments'] = (
        train_df['comment_class_negative'] +
        train_df['comment_class_neutral'] +
        train_df['comment_class_positive']
    )

    for col in ['comment_class_negative', 'comment_class_neutral', 'comment_class_positive']:
        train_df[col] = train_df[col] / train_df['total_comments']

    train_df['norm_ratio'] = pd.to_numeric(train_df['norm_ratio'], errors='coerce')
    train_df = train_df.dropna()
    train_df['video_type'] = pd.cut(train_df['norm_ratio'], bins=[0, 2.5, 4, 8],
                                     labels=['Negative', 'Neutral', 'Positive'], include_lowest=True)
    train_df['video_type_clean'] = train_df['video_type'].map({'Negative': -1, 'Neutral': 0, 'Positive': 1})
    return train_df
