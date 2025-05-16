import numpy as np
import gensim.downloader as gdownload

def embed_comments(df):
    glove = gdownload.load('glove-twitter-50')
    
    def sentence_embed(sentence):
        if not sentence:
            return np.zeros(50)
        return sum((glove[word] if word in glove else np.zeros(50) for word in sentence), np.zeros(50)) / len(sentence)

    df = df[df['likeCount'] != 0].copy()
    df['word_vector'] = df['wordtoken'].apply(sentence_embed)
    df['tokenlength'] = df['wordtoken'].apply(len)
    return df[df['tokenlength'] > 0]
