import os
import pandas as pd
import numpy as np

def load_all_comments(data_dir):
    df_chris = pd.read_csv(os.path.join(data_dir, "chris_comments.csv"))
    df_charlie = pd.read_csv(os.path.join(data_dir, "Charlie_merged_comments.csv"))
    df_cici = pd.read_csv(os.path.join(data_dir, "merged_comments_cici_actual.csv"))
    df_jz = pd.read_csv(os.path.join(data_dir, "merged_data_jz.csv"))

    df_charlie = df_charlie[df_charlie['parentId'].isnull()]
    df_cici = df_cici[['channelId', 'videoId', 'textDisplay', 'textOriginal', 'parentId',
                       'authorDisplayName', 'authorProfileImageUrl', 'authorChannelUrl',
                       'authorChannelId', 'canRate', 'viewerRating', 'likeCount',
                       'publishedAt', 'updatedAt', 'commentId']]
    
    df = pd.concat([df_charlie, df_cici, df_jz, df_chris], ignore_index=True)
    df.drop(columns=['parentId', 'authorProfileImageUrl', 'authorChannelUrl',
                     'authorChannelId', 'canRate', 'viewerRating'], inplace=True, errors='ignore')
    df = df[df['textDisplay'].notnull()]
    return df

def merge_dislikes(df, dislike_path):
    dislikes = pd.read_csv(dislike_path)

    dislikes['likes'] = dislikes['likes'].replace(0, pd.NA)
    dislikes['dislikes'] = dislikes['dislikes'].replace(0, pd.NA)
    dislikes['ratio'] = dislikes['likes'] / dislikes['dislikes']
    dislikes['ratio'] = pd.to_numeric(dislikes['ratio'], errors='coerce')
    dislikes['norm_ratio'] = np.log(dislikes['ratio'])
    dislikes = dislikes.replace([np.inf, -np.inf], pd.NA).dropna(subset=['norm_ratio'])
    dislikes.columns = ['videoId', 'title', 'channel_id', 'channel_title', 'published_at',
       'view_count', 'likes', 'dislikes', 'comment_count', 'tags',
       'description', 'comments', 'ratio', 'norm_ratio']

    return pd.merge(df, dislikes, on='videoId', how='inner')
