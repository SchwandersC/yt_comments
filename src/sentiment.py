from nltk.sentiment import SentimentIntensityAnalyzer

def score_sentiment(df):
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
        print(f"[score_sentiment] {len(failed_rows)} rows failed sentiment scoring. Sample:")
        for text, error in failed_rows[:3]:
            print(f"  {text} | Error: {error}")

    print(f"[score_sentiment] Sentiment assigned to {len(df)} rows.")
    return df
