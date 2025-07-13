import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
)


def run_ml_pipeline(
    df,
    target_col,
    model_option,
    normalization_option=None,
    test_size=0.2,
    random_state=42,
    feature_cols=None
):
    """
    Modular ML pipeline for classification/regression.
    Args:
        df: DataFrame
        target_col: str, target variable
        model_option: str, which model to use
        normalization_option: str or None
        test_size: float
        random_state: int
        feature_cols: list or None
    Returns:
        dict with model, metrics, predictions, etc.
    """
    # Feature/target selection
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle categorical variables
    for col in X.select_dtypes(include=['object', 'category']).columns:
        unique_vals = X[col].nunique()
        if unique_vals == 2:
            # Binary encode
            vals = X[col].unique()
            X[col] = X[col].map({vals[0]: 0, vals[1]: 1})
        else:
            # One-hot encode
            X = pd.get_dummies(X, columns=[col], drop_first=True)

    # Normalization/scaling
    if normalization_option == 'StandardScaler':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif normalization_option in ['l1', 'l2']:
        X = normalize(X, norm=normalization_option)
    else:
        X = X.values if isinstance(X, pd.DataFrame) else X

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Model selection
    model = None
    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_option == "Ridge Classifier":
        model = RidgeClassifier()
    elif model_option == "Lasso Regression":
        model = Lasso()
    elif model_option == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier()
    elif model_option == "Naive Bayes":
        model = GaussianNB()
    elif model_option == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_option == "Random Forest":
        model = RandomForestClassifier()
    elif model_option == "Gradient Boosting Machines (GBMs)":
        model = GradientBoostingClassifier()
    elif model_option == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif model_option == "Linear Discriminant Analysis (LDA)":
        model = LinearDiscriminantAnalysis()
    elif model_option == "Support Vector Machines (SVMs)":
        model = svm.SVC(probability=True)
    elif model_option == "Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu")
    else:
        raise ValueError(f"Unknown model option: {model_option}")

    # Fit model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = predictions

    # Metrics
    metrics = {}
    if len(np.unique(y_train)) == 2:
        # Binary classification metrics
        metrics['accuracy'] = accuracy_score(y_test, predictions)
        metrics['f1'] = f1_score(y_test, predictions)
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_scores)
                precision, recall, _ = precision_recall_curve(y_test, y_scores)
                metrics['pr_auc'] = auc(recall, precision)
            except Exception:
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        metrics['confusion_matrix'] = confusion_matrix(y_test, predictions)
    else:
        # Regression metrics (add more as needed)
        metrics['accuracy'] = model.score(X_test, y_test)

    return {
        'model': model,
        'metrics': metrics,
        'predictions': predictions,
        'y_test': y_test,
        'y_scores': y_scores
    }
