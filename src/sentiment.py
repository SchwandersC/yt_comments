from nltk.sentiment import SentimentIntensityAnalyzer
import logging

logger = logging.getLogger("yt_pipeline")
def score_sentiment(df):
    """
    Assigns sentiment labels ('positive', 'negative', 'neutral') to comments using VADER.

    This function uses NLTK's SentimentIntensityAnalyzer to evaluate the compound score
    of each text entry in the 'textDisplay' column. Labels are assigned based on:
    - score > 0.2 → 'positive'
    - score < -0.2 → 'negative'
    - otherwise   → 'neutral'

    Failed evaluations are logged and default to 'neutral'.

    Args:
        df (pd.DataFrame): DataFrame containing a 'textDisplay' column.

    Returns:
        pd.DataFrame: The same DataFrame with a new 'sentiment' column.
    """
    sia = SentimentIntensityAnalyzer()
    failed_rows = []

    def classify_sentiment(text):
        try:
            score = sia.polarity_scores(text)['compound']
            if score > 0.2:
                return 'positive'
            elif score < -0.2:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            failed_rows.append((text[:60], str(e)))
            return 'neutral'  # fallback label

    df['sentiment'] = df['textDisplay'].apply(classify_sentiment)

    if failed_rows:
        logger.warning(f"[score_sentiment] {len(failed_rows)} rows failed sentiment scoring.")
        for text, error in failed_rows[:3]:
            logger.debug(f"Failed row snippet: '{text}' | Error: {error}")

    logger.info(f"[score_sentiment] Sentiment assigned to {len(df)} rows.")
    return df
