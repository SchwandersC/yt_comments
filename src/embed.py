import numpy as np
import gensim.downloader as gdownload

def embed_comments(df):
    try:
        glove = gdownload.load('glove-twitter-50')
    except Exception as e:
        print(f"[embed_comments] Failed to load GloVe embeddings: {e}")
        raise

    def sentence_embed(sentence):
        try:
            if not isinstance(sentence, list) or not sentence:
                return np.zeros(50)
            return sum((glove[word] if word in glove else np.zeros(50) for word in sentence), np.zeros(50)) / len(sentence)
        except Exception as e:
            print(f"[sentence_embed] Failed to embed sentence: {sentence} | Error: {e}")
            return np.zeros(50)

    try:
        df = df[df['likeCount'] != 0].copy()
        df['word_vector'] = df['wordtoken'].apply(sentence_embed)
        df['tokenlength'] = df['wordtoken'].apply(len)

        empty_rows = df[df['tokenlength'] == 0]
        if not empty_rows.empty:
            print(f"[embed_comments] {len(empty_rows)} rows with empty tokenlength. Sample:")
            print(empty_rows[['textDisplay']].head(3))
    except Exception as e:
        print(f"[embed_comments] Error during embedding: {e}")
        raise

    return df[df['tokenlength'] > 0]
