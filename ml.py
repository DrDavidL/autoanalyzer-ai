import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
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
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
import matplotlib.pyplot as plt
import shap
from sklearn.impute import SimpleImputer


def display_metrics(y_true, y_pred, y_scores, set_name="Test"):
    # Check if this is a regression model (continuous outputs) or classification model
    is_regression = False
    try:
        # Try to compute classification metrics
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        # Handle precision-recall curve with proper pos_label
        unique_labels = np.unique(y_true)
        if len(unique_labels) == 2:
            # For binary classification, determine the positive label
            # If labels are numeric (0, 1), use 1 as positive
            # If labels are strings, use the second label alphabetically as positive
            if np.issubdtype(y_true.dtype, np.number):
                pos_label = max(unique_labels)
            else:
                pos_label = sorted(unique_labels)[1]  # Use second label alphabetically
            
            precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
        else:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        pr_auc = auc(recall, precision)
        
        # Display classification metrics
        st.info(
            f"**Your Model Metrics ({set_name} Set):** F1 score: {f1:.2f}, Accuracy: {accuracy:.2f}, ROC AUC: {roc_auc:.2f}, PR AUC: {pr_auc:.2f}"
        )
    except ValueError as e:
        # If we get a ValueError, it's likely because we're using a regression model
        is_regression = True
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Compute regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Display regression metrics
        st.info(
            f"**Your Model Metrics ({set_name} Set):** RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}"
        )
    
    with st.expander("Explanations for the Metrics"):
        if is_regression:
            st.write(
                """
### Explanation of Regression Metrics
- **RMSE** (Root Mean Squared Error) measures the average magnitude of the errors. It gives higher weight to larger errors.
- **MAE** (Mean Absolute Error) measures the average magnitude of the errors without considering their direction.
- **R²** (R-squared) represents the proportion of variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values indicating better fit.
"""
            )
        else:
            st.write(
                """
### Explanation of Classification Metrics
- **F1 score** is the harmonic mean of precision and recall, and it tries to balance the two. It is a good metric when you have imbalanced classes.
- **Accuracy** is the ratio of correct predictions to the total number of predictions. It can be misleading if the classes are imbalanced.
- **ROC AUC** (Receiver Operating Characteristic Area Under Curve) represents the likelihood of the classifier distinguishing between a positive sample and a negative sample. It's equal to 0.5 for random predictions and 1.0 for perfect predictions.
- **PR AUC** (Precision-Recall Area Under Curve) is another way of summarizing the trade-off between precision and recall, and it gives more weight to precision. It's useful when the classes are imbalanced.
"""
            )
    # Confusion matrix, ROC, and PR curves are now shown in the main ML tab for clarity.
    # ROC curve is now shown only once per set, outside this function.
    
    return is_regression


def plot_pr_curve(y_true, y_scores):
    # Handle precision-recall curve with proper pos_label
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 2:
        # For binary classification, determine the positive label
        # If labels are numeric (0, 1), use 1 as positive
        # If labels are strings, use the second label alphabetically as positive
        if np.issubdtype(y_true.dtype, np.number):
            pos_label = max(unique_labels)
        else:
            pos_label = sorted(unique_labels)[1]  # Use second label alphabetically
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower right")
    return fig


def get_categorical_and_numerical_cols(df):
    # Initialize empty lists for categorical and numerical columns
    categorical_cols = []
    numeric_cols = []

    # Go through each column in the dataframe
    for col in df.columns:
        # If the column data type is numerical and has more than two unique values, add it to the numeric list
        if np.issubdtype(df[col].dtype, np.number) and len(df[col].unique()) > 2:
            numeric_cols.append(col)
        # Otherwise, add it to the categorical list
        else:
            categorical_cols.append(col)

    # Sort the lists
    numeric_cols.sort()
    categorical_cols.sort()

    return numeric_cols, categorical_cols


def plot_confusion_matrix(y_true, y_pred):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create the ConfusionMatrixDisplay object
    cmd = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Class 0", "Class 1"]
    )

    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(dpi=100)

    # Plot the confusion matrix using the `plot` method
    cmd.plot(ax=ax, cmap="Blues", values_format="d")

    # Customize the plot if needed
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    return fig


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.legend(loc="lower right")

    return fig


def preprocess(df, target_col):
    included_cols = []
    excluded_cols = []

    for col in df.columns:
        if col != target_col:  # Exclude target column from preprocessing
            if df[col].dtype == "object":
                if len(df[col].unique()) == 2:  # Bivariate case
                    most_freq = df[col].value_counts().idxmax()
                    least_freq = df[col].value_counts().idxmin()

                    # Update the mapping to include 'F' as 0 and 'M' as 1
                    df[col] = df[col].map({most_freq: 0, least_freq: 1, "F": 0})

                    included_cols.append(col)
                else:  # Multivariate case
                    excluded_cols.append(col)
            elif df[col].dtype in ["int64", "float64"]:  # Numerical case
                if df[col].isnull().values.any():
                    mean_imputer = SimpleImputer(strategy="mean")
                    df[col] = mean_imputer.fit_transform(df[[col]])
                    st.write(f"Imputed missing values in {col} with mean.")

                included_cols.append(col)

    return df[included_cols], included_cols, excluded_cols


def preprocess_old(df, target_col):
    included_cols = []
    excluded_cols = []

    for col in df.columns:
        if col != target_col:  # Exclude target column from preprocessing
            if df[col].dtype == "object":
                if len(df[col].unique()) == 2:  # Bivariate case
                    most_freq = df[col].value_counts().idxmax()
                    least_freq = df[col].value_counts().idxmin()
                    df[col] = df[col].map({most_freq: 0, least_freq: 1})
                    included_cols.append(col)
                else:  # Multivariate case
                    excluded_cols.append(col)
            elif df[col].dtype in ["int64", "float64"]:  # Numerical case
                if df[col].isnull().values.any():
                    mean_imputer = SimpleImputer(strategy="mean")
                    df[col] = mean_imputer.fit_transform(df[[col]])
                    st.write(f"Imputed missing values in {col} with mean.")
                included_cols.append(col)


def run_ml_pipeline(
    df,
    target_col,
    model_option,
    normalization_option=None,
    test_size=0.2,
    random_state=42,
    feature_cols=None,
    perform_shapley=False
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
    
    # Handle target variable encoding if it's categorical
    target_label_mapping = None
    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
        unique_labels = y.unique()
        if len(unique_labels) == 2:
            # Binary classification - create mapping
            target_label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
            y = y.map(target_label_mapping)
        else:
            # Multi-class classification - use label encoding
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            target_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    # Handle categorical variables in features
    for col in X.select_dtypes(include=['object', 'category']).columns:
        unique_vals = X[col].nunique()
        if unique_vals == 2:
            # Binary encode
            vals = X[col].unique()
            X[col] = X[col].map({vals[0]: 0, vals[1]: 1})
        else:
            # One-hot encode
            X = pd.get_dummies(X, columns=[col], drop_first=True)

    # Store feature names before any transformations that might convert to numpy arrays
    feature_names = list(X.columns)
    
    # Normalization/scaling
    if normalization_option == 'StandardScaler':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Keep as DataFrame to preserve column names for SHAP
        X = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    elif normalization_option in ['l1', 'l2']:
        X_normalized = normalize(X, norm=normalization_option)
        # Keep as DataFrame to preserve column names for SHAP
        X = pd.DataFrame(X_normalized, columns=feature_names, index=X.index)
    # If no scaling/normalization, X remains as DataFrame

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

    # Metrics & SHAP
    metrics = {}
    shap_values = None
    explainer = None

    if len(np.unique(y_train)) == 2:
        # Binary classification metrics
        metrics['accuracy'] = accuracy_score(y_test, predictions)
        metrics['f1'] = f1_score(y_test, predictions)
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_scores)
                
                # Handle precision-recall curve with proper pos_label
                unique_labels = np.unique(y_test)
                if np.issubdtype(y_test.dtype, np.number):
                    pos_label = max(unique_labels)
                else:
                    pos_label = sorted(unique_labels)[1]  # Use second label alphabetically
                
                precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=pos_label)
                metrics['pr_auc'] = auc(recall, precision)
            except Exception:
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        metrics['confusion_matrix'] = confusion_matrix(y_test, predictions)
    else:
        # Regression metrics (add more as needed)
        metrics['accuracy'] = model.score(X_test, y_test)

    if perform_shapley:
        # Reduce background sample size to improve performance
        # Use shap.sample to get a smaller representative sample
        if X_train.shape[0] > 100:
            background_sample = shap.sample(X_train, 100)  # Use 100 samples instead of 50 for better representation
        else:
            background_sample = X_train
            
        if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            # For non-tree models, use KernelExplainer
            # Remove the l1_reg limitation to allow all features to be explained
            try:
                # First try with predict_proba for classification models
                if hasattr(model, 'predict_proba') and len(np.unique(y_train)) == 2:
                    explainer = shap.KernelExplainer(model.predict_proba, background_sample)
                    shap_values = explainer.shap_values(X_test)
                    # For binary classification, take the positive class SHAP values
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]
                else:
                    # For regression or models without predict_proba
                    explainer = shap.KernelExplainer(model.predict, background_sample)
                    shap_values = explainer.shap_values(X_test)
            except Exception as e:
                st.warning(f"SHAP analysis failed: {e}. Trying with reduced sample size.")
                try:
                    # Fallback: use smaller sample and limit features if needed
                    smaller_background = shap.sample(X_train, 25)
                    explainer = shap.KernelExplainer(model.predict, smaller_background)
                    shap_values = explainer.shap_values(X_test[:10])  # Analyze only first 10 test samples
                except Exception as e2:
                    st.warning(f"SHAP analysis failed completely: {e2}")
                    explainer = None
                    shap_values = None

    return {
        'model': model,
        'metrics': metrics,
        'predictions': predictions,
        'y_test': y_test,
        'y_scores': y_scores,
        'X_test': X_test,
        'explainer': explainer,
        'shap_values': shap_values,
        'feature_names': feature_names
    }
