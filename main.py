import os
import json
import pickle
import argparse
import pandas as pd

from src.load import load_all_comments, merge_dislikes
from src.clean import clean_comments, tokenize_comments
from src.embed import embed_comments
from src.sentiment import score_sentiment
from src.visualize import plot_wordcloud
from src.feature_engineering import prepare_features
from src.train import train_and_evaluate_all_models

def load_config(config_path):
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config format. Use .json or .yaml")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML or JSON config file")
    parser.add_argument("--retrain-only", action="store_true", help="Only retrain models using existing feature data")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    DATA_DIR = config["data_dir"]
    RECOMPUTE = config["recompute"]
    classification_type = config.get("classification_type", "multiclass")  # new
    use_smote = config.get("use_smote", True)
    selected_models = config.get("models", ["random_forest", "decision_tree", "gradient_boosting"])


    checkpoint_paths = {
        "cleaned": os.path.join(DATA_DIR, "checkpoint_cleaned.pkl"),
        "embedded": os.path.join(DATA_DIR, "checkpoint_embedded.pkl"),
        "merged": os.path.join(DATA_DIR, "checkpoint_merged.pkl"),
        "scored": os.path.join(DATA_DIR, "checkpoint_scored.pkl"),
        "features": os.path.join(DATA_DIR, "sentiment_data.pkl")
    }

    if args.retrain_only:
        print("Retrain-only mode: loading precomputed features...")
        train_df = pd.read_pickle(checkpoint_paths["features"])
        train_and_evaluate_all_models(train_df)
        return

    # Step 1: Load / Clean / Tokenize
    if not RECOMPUTE["load_clean_tokenize"] and os.path.exists(checkpoint_paths["cleaned"]):
        print("Loading cleaned + tokenized data...")
        with open(checkpoint_paths["cleaned"], "rb") as f:
            df = pickle.load(f)
    else:
        print("Processing raw comment data...")
        df = load_all_comments(DATA_DIR)
        df = clean_comments(df)
        df = tokenize_comments(df)
        with open(checkpoint_paths["cleaned"], "wb") as f:
            pickle.dump(df, f)

    # Step 2: Embed
    if not RECOMPUTE["embed"] and os.path.exists(checkpoint_paths["embedded"]):
        print("Loading embedded data...")
        with open(checkpoint_paths["embedded"], "rb") as f:
            df = pickle.load(f)
    else:
        print("Embedding comments...")
        df = embed_comments(df)
        with open(checkpoint_paths["embedded"], "wb") as f:
            pickle.dump(df, f)

    # Step 3: Merge Dislikes
    if not RECOMPUTE["merge_dislikes"] and os.path.exists(checkpoint_paths["merged"]):
        print("Loading merged data with dislikes...")
        with open(checkpoint_paths["merged"], "rb") as f:
            df = pickle.load(f)
    else:
        print("Merging with dislike dataset...")
        df = merge_dislikes(df, os.path.join(DATA_DIR, "youtube_dislike_dataset.csv"))
        with open(checkpoint_paths["merged"], "wb") as f:
            pickle.dump(df, f)

    # Step 4: Sentiment Scoring
    if not RECOMPUTE["sentiment"] and os.path.exists(checkpoint_paths["scored"]):
        print("Loading sentiment-scored data...")
        with open(checkpoint_paths["scored"], "rb") as f:
            df = pickle.load(f)
    else:
        print("Scoring sentiment...")
        df = score_sentiment(df)
        with open(checkpoint_paths["scored"], "wb") as f:
            pickle.dump(df, f)

    # Step 5: Visualization
    print("Plotting word clouds...")
    plot_wordcloud(df, "positive")
    plot_wordcloud(df, "negative")

    # Step 6: Feature Engineering
    if not RECOMPUTE["features"] and os.path.exists(checkpoint_paths["features"]):
        print("Loading precomputed features...")
        train_df = pd.read_pickle(checkpoint_paths["features"])
    else:
        print("Preparing features...")
        train_df = prepare_features(
            df,
            os.path.join(DATA_DIR, "youtube_dislike_dataset.csv"),
            classification_type=classification_type
        )
        train_df.to_pickle(checkpoint_paths["features"])

    # Step 7: Training
    if RECOMPUTE["train"]:
        print("Training and evaluating models...")
        train_and_evaluate_all_models(train_df, use_smote=use_smote, selected_models=selected_models)
    else:
        print("Skipping training step.")

if __name__ == "__main__":
    main()
