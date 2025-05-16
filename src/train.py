from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def evaluate_model(model, x_test, y_test, name, output_dir="data/models"):
    # Predict
    y_pred = model.predict(x_test)

    # Print metrics
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    # Save confusion matrix plot
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Neg', 'Neu', 'Pos'])
    disp.ax_.set_title(name)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {fig_path}")


def train_and_evaluate_all_models(train_df):
    X = train_df[['comment_class_negative', 'comment_class_neutral', 'comment_class_positive', 'like-view']]
    y = train_df['video_type_clean']
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=2023)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=2023),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, learning_rate=0.2, max_depth=3, random_state=2023),
        "Random Forest": RandomForestClassifier(n_estimators=1000, random_state=2023)
    }

    for name, model in models.items():
        model.fit(x_train, y_train)
        evaluate_model(model, x_test, y_test, name)

    # SMOTE
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    for name, model in models.items():
        model.fit(x_train_resampled, y_train_resampled)
        evaluate_model(model, x_test, y_test, name + " (SMOTE)")
