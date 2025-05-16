import os
import pandas as pd
import numpy as np

def load_all_comments(data_dir):
    try:
        df_chris = pd.read_csv(os.path.join(data_dir, "chris_comments.csv"))
        df_charlie = pd.read_csv(os.path.join(data_dir, "Charlie_merged_comments.csv"))
        df_cici = pd.read_csv(os.path.join(data_dir, "merged_comments_cici_actual.csv"))
        df_jz = pd.read_csv(os.path.join(data_dir, "merged_data_jz.csv"))
    except Exception as e:
        print(f"[load_all_comments] Error loading one of the CSVs: {e}")
        raise

    try:
        df_charlie = df_charlie[df_charlie['parentId'].isnull()]
        df_cici = df_cici[['channelId', 'videoId', 'textDisplay', 'textOriginal', 'parentId',
                           'authorDisplayName', 'authorProfileImageUrl', 'authorChannelUrl',
                           'authorChannelId', 'canRate', 'viewerRating', 'likeCount',
                           'publishedAt', 'updatedAt', 'commentId']]
    except Exception as e:
        print(f"[load_all_comments] Error selecting or filtering columns: {e}")
        raise

    try:
        df = pd.concat([df_charlie, df_cici, df_jz, df_chris], ignore_index=True)
        df.drop(columns=[
            'parentId', 'authorProfileImageUrl', 'authorChannelUrl',
            'authorChannelId', 'canRate', 'viewerRating'
        ], inplace=True, errors='ignore')
        df = df[df['textDisplay'].notnull()]
    except Exception as e:
        print(f"[load_all_comments] Error during concatenation or cleaning: {e}")
        raise

    return df

def merge_dislikes(df, dislike_path):
    if not os.path.exists(dislike_path):
        raise FileNotFoundError(f"Dislike file not found: {dislike_path}")

    try:
        dislikes = pd.read_csv(dislike_path)
        dislikes['likes'] = dislikes['likes'].replace(0, pd.NA)
        dislikes['dislikes'] = dislikes['dislikes'].replace(0, pd.NA)
        dislikes['ratio'] = dislikes['likes'] / dislikes['dislikes']
        dislikes['ratio'] = pd.to_numeric(dislikes['ratio'], errors='coerce')
        dislikes['norm_ratio'] = np.log(dislikes['ratio'])
        dislikes = dislikes.replace([np.inf, -np.inf], pd.NA).dropna(subset=['norm_ratio'])
    except Exception as e:
        print(f"[merge_dislikes] Error computing ratio/log columns: {e}")
        raise

    try:
        dislikes.columns = ['videoId', 'title', 'channel_id', 'channel_title', 'published_at',
                            'view_count', 'likes', 'dislikes', 'comment_count', 'tags',
                            'description', 'comments', 'ratio', 'norm_ratio']
    except Exception as e:
        print(f"[merge_dislikes] Error renaming columns — verify input format: {e}")
        raise

    try:
        merged = pd.merge(df, dislikes, on='videoId', how='inner')
    except Exception as e:
        print(f"[merge_dislikes] Error merging with main DataFrame: {e}")
        raise

    return merged
