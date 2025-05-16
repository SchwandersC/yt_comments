from nltk.sentiment import SentimentIntensityAnalyzer

def score_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['textDisplay'].apply(
        lambda text: 'positive' if sia.polarity_scores(text)['compound'] > 0.2 else
                     'negative' if sia.polarity_scores(text)['compound'] < -0.2 else
                     'neutral'
    )
    print(df.shape)
    return df
