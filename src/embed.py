import numpy as np
import gensim.downloader as gdownload
import logging

logger = logging.getLogger("yt_pipeline")
from gensim.models import KeyedVectors
glove = KeyedVectors.load_word2vec_format(
    './gensim-data/glove-twitter-50/glove-twitter-50.gz'
)



def embed_comments(df):
    """
    Generates sentence embeddings for each comment in the DataFrame using GloVe word vectors.

    This function:
    - Loads GloVe embeddings (`glove-twitter-50`)
    - Computes an average embedding vector for each tokenized comment
    - Filters out rows with zero tokens or likeCount == 0

    Args:
        df (pd.DataFrame): A DataFrame containing at least 'wordtoken' and 'likeCount' columns.

    Returns:
        pd.DataFrame: Filtered DataFrame with additional 'word_vector' and 'tokenlength' columns.
    """
    try:
        logger.info("Successfully loaded GloVe embeddings.")
    except Exception as e:
        logger.error("[embed_comments] Failed to load GloVe embeddings", exc_info=True)
        raise

    def sentence_embed(sentence):
        """
        Computes the average GloVe embedding for a list of tokens.

        Args:
            sentence (list): A list of word tokens.

        Returns:
            np.ndarray: A 50-dimensional embedding vector.
        """
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
