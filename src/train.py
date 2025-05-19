"""
Model training and evaluation module for YouTube comment sentiment classification.

Supports multiple classifiers, SMOTE oversampling, model saving, and confusion matrix visualization.
"""

import os
import joblib
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    has_xgboost = False

logger = logging.getLogger("yt_pipeline")

def evaluate_model(model, x_test, y_test, name, output_dir="data/models", save_as_final=False):
    """
    Evaluates a trained model, saves it to disk, and plots the confusion matrix.

    Args:
        model: Trained classifier.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        name (str): Model name.
        output_dir (str): Directory to store model and figures.
        save_as_final (bool): If True, also save model as 'final_model.pkl'.

    Returns:
        None
    """
    try:
        y_pred = model.predict(x_test)

        logger.info(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.debug(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")

        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")

        if save_as_final:
            final_path = os.path.join(output_dir, "final_model.pkl")
            joblib.dump(model, final_path)
            logger.info(f"Saved final model as {final_path}")

        display_labels = sorted(y_test.unique().astype(str))
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=display_labels)
        disp.ax_.set_title(name)

        fig_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix to {fig_path}")

    except Exception as e:
        logger.error(f"[evaluate_model] Error evaluating {name}: {e}", exc_info=True)


def train_and_evaluate_all_models(train_df, use_smote=True, selected_models=None):
    """
    Trains and evaluates selected models on sentiment features, saving primary model for later use.

    Args:
        train_df (pd.DataFrame): DataFrame with features + target (`video_type_clean`).
        use_smote (bool): Whether to apply SMOTE to handle class imbalance.
        selected_models (list[str]): Classifier types to run (e.g., ['random_forest', 'xgboost']).

    Returns:
        None
    """
    if selected_models is None:
        selected_models = ["random_forest", "decision_tree", "gradient_boosting"]

    required_cols = [
        'comment_class_negative', 'comment_class_neutral',
        'comment_class_positive', 'like-view', 'video_type_clean'
    ]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing required column in training data: {col}")

    X = train_df[['comment_class_negative', 'comment_class_neutral', 'comment_class_positive', 'like-view']]
    y = train_df['video_type_clean']
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=2023)

    model_defs = {}

    if "decision_tree" in selected_models:
        model_defs["Decision Tree"] = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=2023)
    if "gradient_boosting" in selected_models:
        model_defs["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=80, learning_rate=0.2, max_depth=3, random_state=2023)
    if "random_forest" in selected_models:
        model_defs["Random Forest"] = RandomForestClassifier(n_estimators=1000, random_state=2023)
    if "xgboost" in selected_models and has_xgboost:
        model_defs["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='mlogloss', random_state=2023)

    logger.info("Training models on original data...")
    for idx, (name, model) in enumerate(model_defs.items()):
        logger.info(f"Training {name}...")
        model.fit(x_train, y_train)
        # Save the *first trained model* as final_model.pkl for loading later
        evaluate_model(model, x_test, y_test, name, save_as_final=(idx == 0))

    if use_smote:
        logger.info("Training models with SMOTE oversampling...")
        try:
            smote = SMOTE(random_state=42)
            x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

            for name, model in model_defs.items():
                logger.info(f"Training {name} with SMOTE...")
                model.fit(x_train_resampled, y_train_resampled)
                evaluate_model(model, x_test, y_test, name + " (SMOTE)")
        except Exception as e:
            logger.error(f"[SMOTE] Error during SMOTE training: {e}", exc_info=True)
    else:
        logger.info("SMOTE is disabled via config.")


def load_final_model(path="data/models/final_model.pkl"):
    """
    Loads a previously trained and saved model.

    Args:
        path (str): Path to the saved model pickle file.

    Returns:
        Trained model object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model found at {path}")
    return joblib.load(path)
