import os
import pandas as pd
import numpy as np
import logging

# Shared logger
logger = logging.getLogger("yt_pipeline")

def load_all_comments(data_dir):
    try:
        df_chris = pd.read_csv(os.path.join(data_dir, "chris_comments.csv"))
        df_charlie = pd.read_csv(os.path.join(data_dir, "Charlie_merged_comments.csv"))
        df_cici = pd.read_csv(os.path.join(data_dir, "merged_comments_cici_actual.csv"))
        df_jz = pd.read_csv(os.path.join(data_dir, "merged_data_jz.csv"))
        logger.info("Successfully loaded all comment CSV files.")
    except Exception as e:
        logger.error("[load_all_comments] Error loading one of the CSVs", exc_info=True)
        raise

    try:
        df_charlie = df_charlie[df_charlie['parentId'].isnull()]
        df_cici = df_cici[[
            'channelId', 'videoId', 'textDisplay', 'textOriginal', 'parentId',
            'authorDisplayName', 'authorProfileImageUrl', 'authorChannelUrl',
            'authorChannelId', 'canRate', 'viewerRating', 'likeCount',
            'publishedAt', 'updatedAt', 'commentId'
        ]]
        logger.info("Filtered Charlie/Cici datasets successfully.")
    except Exception as e:
        logger.error("[load_all_comments] Error selecting or filtering columns", exc_info=True)
        raise

    try:
        df = pd.concat([df_charlie, df_cici, df_jz, df_chris], ignore_index=True)
        df.drop(columns=[
            'parentId', 'authorProfileImageUrl', 'authorChannelUrl',
            'authorChannelId', 'canRate', 'viewerRating'
        ], inplace=True, errors='ignore')
        df = df[df['textDisplay'].notnull()]
        logger.info(f"Successfully merged all datasets. Final comment count: {len(df)}")
    except Exception as e:
        logger.error("[load_all_comments] Error during concatenation or cleaning", exc_info=True)
        raise

    return df

def merge_dislikes(df, dislike_path):
    if not os.path.exists(dislike_path):
        logger.error(f"[merge_dislikes] Dislike file not found: {dislike_path}")
        raise FileNotFoundError(f"Dislike file not found: {dislike_path}")

    try:
        dislikes = pd.read_csv(dislike_path)
        dislikes['likes'] = dislikes['likes'].replace(0, pd.NA)
        dislikes['dislikes'] = dislikes['dislikes'].replace(0, pd.NA)
        dislikes['ratio'] = dislikes['likes'] / dislikes['dislikes']
        dislikes['ratio'] = pd.to_numeric(dislikes['ratio'], errors='coerce')
        dislikes['norm_ratio'] = np.log(dislikes['ratio'])
        dislikes = dislikes.replace([np.inf, -np.inf], pd.NA).dropna(subset=['norm_ratio'])
        logger.info(f"Processed dislikes data with {len(dislikes)} usable rows.")
    except Exception as e:
        logger.error("[merge_dislikes] Error computing ratio/log columns", exc_info=True)
        raise

    try:
        dislikes.columns = [
            'videoId', 'title', 'channel_id', 'channel_title', 'published_at',
            'view_count', 'likes', 'dislikes', 'comment_count', 'tags',
            'description', 'comments', 'ratio', 'norm_ratio'
        ]
    except Exception as e:
        logger.error("[merge_dislikes] Error renaming columns — verify input format", exc_info=True)
        raise

    try:
        merged = pd.merge(df, dislikes, on='videoId', how='inner')
        logger.info(f"Merged comments with dislikes. Final merged rows: {len(merged)}")
    except Exception as e:
        logger.error("[merge_dislikes] Error merging with main DataFrame", exc_info=True)
        raise

    return merged
