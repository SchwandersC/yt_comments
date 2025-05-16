from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import logging

# Use default logger if none provided
logger = logging.getLogger("yt_pipeline")

def plot_wordcloud(df, sentiment_label, output_dir="data/wordclouds", extra_stopwords=None):
    try:
        comments = df['textDisplay'][df['sentiment'] == sentiment_label]
        text = " ".join(comments)

        base_stopwords = set([
            "a", "the", "and", "is", "with", "this", "video", "people", "s", "one",
            "will", "know", "think", "make", "u", "time"
        ])
        if extra_stopwords:
            base_stopwords.update(extra_stopwords)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=STOPWORDS.union(base_stopwords)
        ).generate(text)

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{sentiment_label}_wordcloud.png")

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{sentiment_label.capitalize()} Word Cloud')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()

        logger.info(f"Saved word cloud to {filepath}")

    except Exception as e:
        logger.error(f"Failed to generate word cloud for '{sentiment_label}': {e}")
