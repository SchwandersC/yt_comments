import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE

def evaluate_model(model, x_test, y_test, name, output_dir="data/models"):
    try:
        y_pred = model.predict(x_test)

        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Neg', 'Neu', 'Pos'])
        disp.ax_.set_title(name)
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_confusion_matrix.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix to {fig_path}")

    except Exception as e:
        print(f"[evaluate_model] Error evaluating {name}: {e}")

def train_and_evaluate_all_models(train_df):
    required_cols = [
        'comment_class_negative', 'comment_class_neutral',
        'comment_class_positive', 'like-view', 'video_type_clean'
    ]
    for col in required_cols:
        if col
