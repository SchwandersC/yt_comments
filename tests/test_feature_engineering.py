import pandas as pd
import numpy as np
import tempfile
import os
from src.feature_engineering import prepare_features

def mock_input_df():
    return pd.DataFrame({
        'videoId': ['vid1', 'vid1', 'vid2'],
        'sentiment': ['positive', 'neutral', 'negative'],
        'norm_ratio': [1.0, 2.5, 5.0]
    })

def mock_dislike_csv(path):
    df = pd.DataFrame({
        'video_id': ['vid1', 'vid2'],
        'likes': [100, 50],
        'view_count': [1000, 500]
    })
    df.to_csv(path, index=False)

def test_prepare_features_binary_classification():
    df = mock_input_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "dislikes.csv")
        mock_dislike_csv(csv_path)
        result = prepare_features(df, csv_path, classification_type="binary")

        assert "video_type_clean" in result.columns
        assert result["video_type_clean"].isin([0, 1]).all()

def test_prepare_features_multiclass_classification():
    df = mock_input_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "dislikes.csv")
        mock_dislike_csv(csv_path)
        result = prepare_features(df, csv_path, classification_type="multiclass")

        assert "video_type_clean" in result.columns
        assert set(result["video_type_clean"].dropna().unique()).issubset({-1, 0, 1})

def test_dislike_file_missing():
    df = mock_input_df()
    try:
        prepare_features(df, "nonexistent.csv", classification_type="binary")
    except FileNotFoundError as e:
        assert "Dislike file not found" in str(e)
    else:
        assert False, "Expected FileNotFoundError"

