import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger("yt_pipeline")

def prepare_features(df, dislike_file, classification_type):
    try:
        df = pd.get_dummies(df, columns=['sentiment'], prefix='comment_class')
        logger.info("One-hot encoding completed for 'sentiment'.")
    except Exception as e:
        logger.error("Error during one-hot encoding", exc_info=True)
        raise

    try:
        sentiment_sums = df.groupby('videoId')[[
            'comment_class_negative',
            'comment_class_neutral',
            'comment_class_positive'
        ]].sum()
        logger.info("Aggregated sentiment counts by videoId.")
    except KeyError as e:
        logger.error("Missing expected sentiment columns", exc_info=True)
        raise

    try:
        ratios = df[['videoId', 'norm_ratio']].groupby('videoId').max()
        logger.info("Computed max norm_ratio per videoId.")
    except Exception as e:
        logger.error("Error computing norm_ratio groupby", exc_info=True)
        raise

    try:
        train_df = pd.merge(sentiment_sums, ratios, on='videoId')
        logger.info("Merged sentiment sums with norm_ratio.")
    except Exception as e:
        logger.error("Error merging sentiment sums and ratios", exc_info=True)
        raise

    if not os.path.exists(dislike_file):
        logger.error(f"Dislike file not found: {dislike_file}")
        raise FileNotFoundError(f"Dislike file not found: {dislike_file}")

    try:
        dislikes = pd.read_csv(dislike_file)
        dislikes['like-view'] = dislikes['likes'] / dislikes['view_count']
        like_view = dislikes[['video_id', 'like-view']]
        train_df = train_df.merge(like_view, left_index=True, right_on='video_id', how='left')
        logger.info("Merged like-view ratio into training data.")
    except Exception as e:
        logger.error("Error loading or merging dislike data", exc_info=True)
        raise

    try:
        train_df['total_comments'] = (
            train_df['comment_class_negative'] +
            train_df['comment_class_neutral'] +
            train_df['comment_class_positive']
        )
        for col in ['comment_class_negative', 'comment_class_neutral', 'comment_class_positive']:
            train_df[col] = train_df[col] / train_df['total_comments']
        logger.info("Normalized sentiment proportions.")
    except Exception as e:
        logger.error("Error normalizing sentiment proportions", exc_info=True)
        raise

    try:
        train_df['norm_ratio'] = pd.to_numeric(train_df['norm_ratio'], errors='coerce')
        train_df = train_df.dropna()

        if classification_type == "binary":
            train_df['video_type'] = pd.cut(
                train_df['norm_ratio'],
                bins=[0, 3, np.inf],
                labels=['Low', 'High'],
                include_lowest=True
            )
            train_df['video_type_clean'] = train_df['video_type'].map({'Low': 0, 'High': 1})
            logger.info("Binned norm_ratio into 2 binary classes.")
        else:
            train_df['video_type'] = pd.cut(
                train_df['norm_ratio'],
                bins=[0, 2.5, 4, 8],
                labels=['Negative', 'Neutral', 'Positive'],
                include_lowest=True
            )
            train_df['video_type_clean'] = train_df['video_type'].map({
                'Negative': -1,
                'Neutral': 0,
                'Positive': 1
            })
            logger.info("Binned norm_ratio into 3 multiclass labels.")

    except Exception as e:
        logger.error("Error binning norm_ratio or mapping class labels", exc_info=True)
        raise

    return train_df
