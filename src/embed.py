import numpy as np
import gensim.downloader as gdownload
import logging

logger = logging.getLogger("yt_pipeline")

def embed_comments(df):
    try:
        glove = gdownload.load('glove-twitter-50')
        logger.info("Successfully loaded GloVe embeddings.")
    except Exception as e:
        logger.error("[embed_comments] Failed to load GloVe embeddings", exc_info=True)
        raise

    def sentence_embed(sentence):
        try:
            if not isinstance(sentence, list) or not sentence:
                return np.zeros(50)
            return sum(
                (glove[word] if word in glove else np.zeros(50)) for word in sentence
            ) / len(sentence)
        except Exception as e:
            logger.warning(f"[sentence_embed] Failed to embed sentence: {sentence[:10]} | Error: {e}")
            return np.zeros(50)

    try:
        df = df[df['likeCount'] != 0].copy()
        df['word_vector'] = df['wordtoken'].apply(sentence_embed)
        df['tokenlength'] = df['wordtoken'].apply(len)

        empty_rows = df[df['tokenlength'] == 0]
        if not empty_rows.empty:
            logger.warning(f"[embed_comments] {len(empty_rows)} rows with empty tokenlength. Showing sample:")
            logger.debug(empty_rows[['textDisplay']].head(3).to_string(index=False))
    except Exception as e:
        logger.error("[embed_comments] Error during embedding", exc_info=True)
        raise

    df_final = df[df['tokenlength'] > 0]
    logger.info(f"[embed_comments] Embedded {len(df_final)} comments with non-empty tokens.")
    return df_final
