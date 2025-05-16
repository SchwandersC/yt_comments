import os
import pickle
import pandas as pd
from src.load import load_all_comments, merge_dislikes
from src.clean import clean_comments, tokenize_comments
from src.embed import embed_comments
from src.sentiment import score_sentiment
from src.visualize import plot_wordcloud
from src.feature_engineering import prepare_features
from src.train import train_and_evaluate_all_models

# === CONFIG ===
DATA_DIR = "data"
RECOMPUTE = {
    "load_clean_tokenize": False,   # Set to True to force reload and re-clean raw comments
    "embed": False,                 # Set to True to re-run embedding
    "merge_dislikes": False,
    "sentiment": False,
    "features": False,
    "train": True                   # Always true if you want to rerun training
}

# === CACHING FILE PATHS ===
checkpoint_paths = {
    "cleaned": os.path.join(DATA_DIR, "checkpoint_cleaned.pkl"),
    "embedded": os.path.join(DATA_DIR, "checkpoint_embedded.pkl"),
    "merged": os.path.join(DATA_DIR, "checkpoint_merged.pkl"),
    "scored": os.path.join(DATA_DIR, "checkpoint_scored.pkl"),
    "features": os.path.join(DATA_DIR, "sentiment_data.pkl")
}

# === STEP 1: Load / Clean / Tokenize ===
if not RECOMPUTE["load_clean_tokenize"] and os.path.exists(checkpoint_paths["cleaned"]):
    print("Loading cleaned + tokenized data...")
    with open(checkpoint_paths["cleaned"], "rb") as f:
        df = pickle.load(f)
else:
    print("Loading and src raw comment data...")
    df = load_all_comments(DATA_DIR)
    df = clean_comments(df)
    df = tokenize_comments(df)
    with open(checkpoint_paths["cleaned"], "wb") as f:
        pickle.dump(df, f)

# === STEP 2: Embed Comments ===
if not RECOMPUTE["embed"] and os.path.exists(checkpoint_paths["embedded"]):
    print("Loading embedded data...")
    with open(checkpoint_paths["embedded"], "rb") as f:
        df = pickle.load(f)
else:
    print("Embedding comments...")
    df = embed_comments(df)
    with open(checkpoint_paths["embedded"], "wb") as f:
        pickle.dump(df, f)

# === STEP 3: Merge Dislikes ===
if not RECOMPUTE["merge_dislikes"] and os.path.exists(checkpoint_paths["merged"]):
    print("Loading merged data with dislikes...")
    with open(checkpoint_paths["merged"], "rb") as f:
        df = pickle.load(f)
else:
    print("Merging with dislike dataset...")
    df = merge_dislikes(df, os.path.join(DATA_DIR, "youtube_dislike_dataset.csv"))
    with open(checkpoint_paths["merged"], "wb") as f:
        pickle.dump(df, f)

# === STEP 4: Sentiment Scoring ===
if not RECOMPUTE["sentiment"] and os.path.exists(checkpoint_paths["scored"]):
    print("Loading sentiment-scored data...")
    with open(checkpoint_paths["scored"], "rb") as f:
        df = pickle.load(f)
else:
    print("Scoring sentiment...")
    df = score_sentiment(df)
    with open(checkpoint_paths["scored"], "wb") as f:
        pickle.dump(df, f)

# === STEP 5: Visualize Word Clouds ===
print("Plotting word clouds...")
plot_wordcloud(df, "positive")
plot_wordcloud(df, "negative")

# === STEP 6: Feature Engineering ===
if not RECOMPUTE["features"] and os.path.exists(checkpoint_paths["features"]):
    print("Loading precomputed features...")
    train_df = pd.read_pickle(checkpoint_paths["features"])
else:
    print("Preparing features...")
    train_df = prepare_features(df, os.path.join(DATA_DIR, "youtube_dislike_dataset.csv"))
    train_df.to_pickle(checkpoint_paths["features"])

# === STEP 7: Train & Evaluate Models ===
if RECOMPUTE["train"]:
    print("Training and evaluating models...")
    train_and_evaluate_all_models(train_df)
else:
    print("Skipping training step.")
