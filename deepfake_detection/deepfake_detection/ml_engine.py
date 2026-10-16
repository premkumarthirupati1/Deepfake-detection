"""
Modern Machine Learning Engine for Deepfake Metadata & Digital Forensics Analysis.
Provides persistent caching, high-performance inference, and comprehensive evaluation metrics.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "saved_models"
DATASET_PATH = BASE_DIR / "Datasets.csv"

_CACHED_MODELS = None
_CACHED_VECTORIZER = None


def get_dataset_path():
    return DATASET_PATH


def ensure_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Loads and sanitizes the dataset."""
    df = pd.read_csv(DATASET_PATH, encoding='latin-1')
    mapping = {'FAKE': 0, 'REAL': 1}
    df['Results'] = df['label'].map(mapping)
    df['videoname'] = df['videoname'].fillna('')
    df['url'] = df['url'].fillna('')
    return df


def train_and_evaluate_all():
    """
    Trains all models on the dataset, evaluates metrics, and caches models to disk.
    Returns a dictionary of detailed metrics per model.
    """
    global _CACHED_MODELS, _CACHED_VECTORIZER
    ensure_model_dir()

    data = load_dataset()
    # Training primarily on URL / domain forensics tokens (as specified in the original literature paper)
    x = data['url']
    y = data['Results']

    cv = CountVectorizer(lowercase=True, strip_accents='unicode')
    X = cv.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 1. Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    # 2. Linear SVM
    svm = LinearSVC(random_state=42, max_iter=2000)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    # 3. Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # 4. Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    # 5. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)

    # 6. Ensemble Voting Classifier
    voting = VotingClassifier(
        estimators=[('nb', nb), ('svm', svm), ('lr', lr)],
        voting='hard'
    )
    voting.fit(X_train, y_train)
    y_pred_voting = voting.predict(X_test)

    # Save to disk for instant inference
    joblib.dump(cv, MODEL_DIR / "vectorizer.joblib")
    joblib.dump(nb, MODEL_DIR / "naive_bayes.joblib")
    joblib.dump(svm, MODEL_DIR / "svm.joblib")
    joblib.dump(lr, MODEL_DIR / "logistic_regression.joblib")
    joblib.dump(dt, MODEL_DIR / "decision_tree.joblib")
    joblib.dump(gb, MODEL_DIR / "gradient_boosting.joblib")
    joblib.dump(voting, MODEL_DIR / "voting_classifier.joblib")

    # Update memory cache
    _CACHED_VECTORIZER = cv
    _CACHED_MODELS = {
        'Naive Bayes': nb,
        'SVM': svm,
        'Logistic Regression': lr,
        'Decision Tree Classifier': dt,
        'GradientBoostingClassifier': gb,
        'Voting Ensemble': voting,
    }

    models_eval = {
        'Naive Bayes': y_pred_nb,
        'SVM': y_pred_svm,
        'Logistic Regression': y_pred_lr,
        'Decision Tree Classifier': y_pred_dt,
        'GradientBoostingClassifier': y_pred_gb,
        'Voting Ensemble': y_pred_voting,
    }

    results = {}
    for name, pred in models_eval.items():
        acc = float(accuracy_score(y_test, pred) * 100)
        prec = float(precision_score(y_test, pred, zero_division=0) * 100)
        rec = float(recall_score(y_test, pred, zero_division=0) * 100)
        f1 = float(f1_score(y_test, pred, zero_division=0) * 100)
        cm = confusion_matrix(y_test, pred).tolist()

        results[name] = {
            'accuracy': round(acc, 2),
            'precision': round(prec, 2),
            'recall': round(rec, 2),
            'f1_score': round(f1, 2),
            'confusion_matrix': cm,
        }

    # Update Labeled_Data.csv
    labeled_path = BASE_DIR / 'Labeled_Data.csv'
    data.to_csv(labeled_path, index=False)

    return results


def load_trained_models():
    """Loads vectorizer and voting classifier from disk or trains if absent."""
    global _CACHED_MODELS, _CACHED_VECTORIZER

    if _CACHED_MODELS is not None and _CACHED_VECTORIZER is not None:
        return _CACHED_VECTORIZER, _CACHED_MODELS

    voting_path = MODEL_DIR / "voting_classifier.joblib"
    vec_path = MODEL_DIR / "vectorizer.joblib"

    if voting_path.exists() and vec_path.exists():
        _CACHED_VECTORIZER = joblib.load(vec_path)
        _CACHED_MODELS = {
            'Voting Ensemble': joblib.load(voting_path),
            'Naive Bayes': joblib.load(MODEL_DIR / "naive_bayes.joblib") if (MODEL_DIR / "naive_bayes.joblib").exists() else None,
            'SVM': joblib.load(MODEL_DIR / "svm.joblib") if (MODEL_DIR / "svm.joblib").exists() else None,
            'Logistic Regression': joblib.load(MODEL_DIR / "logistic_regression.joblib") if (MODEL_DIR / "logistic_regression.joblib").exists() else None,
            'Decision Tree Classifier': joblib.load(MODEL_DIR / "decision_tree.joblib") if (MODEL_DIR / "decision_tree.joblib").exists() else None,
            'GradientBoostingClassifier': joblib.load(MODEL_DIR / "gradient_boosting.joblib") if (MODEL_DIR / "gradient_boosting.joblib").exists() else None,
        }
        return _CACHED_VECTORIZER, _CACHED_MODELS

    # Train if not yet saved
    train_and_evaluate_all()
    return _CACHED_VECTORIZER, _CACHED_MODELS


def predict_single(videoname="", url=""):
    """
    Sub-millisecond inference for single input.
    Returns (label, numeric_pred) e.g. ('REAL', 1) or ('FAKE', 0).
    """
    cv, models = load_trained_models()
    model = models.get('Voting Ensemble')

    text = str(url).strip()
    if not text:
        text = str(videoname).strip()

    vector = cv.transform([text]).toarray()
    prediction_num = int(model.predict(vector)[0])

    label = 'REAL' if prediction_num == 1 else 'FAKE'
    return label, prediction_num
