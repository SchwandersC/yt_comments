import argparse
import pandas as pd
import numpy as np
import logging
import joblib
import os
from googleapiclient.discovery import build
from src.clean import clean_comments, tokenize_comments
from src.embed import embed_comments
from src.sentiment import score_sentiment
from src.feature_engineering import prepare_features
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("yt_pipeline")
logging.basicConfig(level=logging.INFO)

# Replace with your API key or load from env
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise EnvironmentError("YOUTUBE_API_KEY environment variable not set")


def get_video_metadata(video_id):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.videos().list(part="statistics", id=video_id)
    response = request.execute()
    stats = response["items"][0]["statistics"]
    return {
        "video_id": video_id,
        "view_count": int(stats.get("viewCount", 1)),
        "like_count": int(stats.get("likeCount", 1)),
    }


def get_video_comments(video_id):
    """
    Retrieves all top-level comments from a YouTube video using the YouTube Data API.

    Args:
        video_id (str): The ID of the YouTube video.
        api_key (str): Your YouTube Data API key.

    Returns:
        pd.DataFrame: A DataFrame containing all top-level comments with like counts.
    """
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=1000,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "videoId": video_id,
                "textDisplay": comment.get("textDisplay", ""),
                "likeCount": comment.get("likeCount", 0)
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return pd.DataFrame(comments)


def load_final_model(path="data/models/final_model.pkl"):
    return joblib.load(path)


def main(video_id, model_path, classification_type):
    logger.info(f"Fetching data for video: {video_id}")
    meta = get_video_metadata(video_id)
    df = get_video_comments(video_id)
    print(df.shape)
    if df.empty:
        logger.error("No comments found for this video.")
        return

    logger.info("Running preprocessing...")
    df = clean_comments(df)
    df = tokenize_comments(df)
    df = embed_comments(df)
    df = score_sentiment(df)

    # Fake dislike file using metadata for prepare_features
    fake_dislike = pd.DataFrame.from_records([
        {
            "video_id": meta["video_id"],
            "likes": meta["like_count"],
            "view_count": meta["view_count"]
        }
    ])
    fake_path = "temp_dislike.csv"
    fake_dislike.to_csv(fake_path, index=False)
    print(df.columns)
    features_df = prepare_features(df, fake_path, classification_type, inference_mode=True)
    X = features_df[[
        'comment_class_negative', 'comment_class_neutral',
        'comment_class_positive', 'like-view']]

    model = load_final_model(model_path)
    prediction = model.predict(X)
    majority = pd.Series(prediction).mode()[0]
    logger.info(f"Predicted video class: {majority}")

def run_video_prediction(video_id, model_path="data/models/final_model.pkl", classification_type="binary"):
    """
    Runs end-to-end prediction for a given video ID and returns the majority sentiment class.

    Args:
        video_id (str): YouTube video ID.
        model_path (str): Path to saved model.
        classification_type (str): "binary" or "multiclass".

    Returns:
        int or str: Predicted class label (e.g., 0, 1, or -1).
    """
    meta = get_video_metadata(video_id)
    df = get_video_comments(video_id)

    if df.empty:
        raise ValueError("No comments found for this video.")

    df = clean_comments(df)
    df = tokenize_comments(df)
    df = embed_comments(df)
    df = score_sentiment(df)

    fake_dislike = pd.DataFrame([{
        "video_id": meta["video_id"],
        "likes": meta["like_count"],
        "view_count": meta["view_count"]
    }])
    fake_path = "temp_dislike.csv"
    fake_dislike.to_csv(fake_path, index=False)

    features_df = prepare_features(df, fake_path, classification_type, inference_mode=True)

    X = features_df[[  # adapt this to match training features
        'comment_class_negative', 'comment_class_neutral',
        'comment_class_positive', 'like-view'
    ]]

    model = load_final_model(model_path)
    predictions = model.predict(X)

    return pd.Series(predictions).mode()[0]  # Majority prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True, help="YouTube Video ID")
    parser.add_argument("--model", default="data/models/final_model.pkl", help="Path to saved model")
    parser.add_argument("--classification-type", default="binary", choices=["binary", "multiclass"], help="Classification mode")
    args = parser.parse_args()

    main(args.video_id, args.model, args.classification_type)
