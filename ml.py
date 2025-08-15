import numpy as np
import pandas as pd
import streamlit as st
import data_validation
import monitoring
import utils
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
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
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
import matplotlib.pyplot as plt
import shap
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

# Import SMOTE for handling imbalanced datasets
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    st.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Additional imports for sensitivity optimization
from sklearn.metrics import precision_recall_curve, recall_score, precision_score


def detect_imbalance(y, threshold=0.1):
    """
    Detect if the target variable is imbalanced.
    
    Args:
        y: Target variable
        threshold: Minimum proportion for the minority class (default 10%)
    
    Returns:
        dict: Contains imbalance info including is_imbalanced, minority_ratio, class_counts
    """
    if len(np.unique(y)) != 2:
        return {
            "is_imbalanced": False,
            "minority_ratio": None,
            "class_counts": None,
            "message": "Imbalance detection only supports binary classification"
        }
    
    class_counts = pd.Series(y).value_counts()
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    total_count = len(y)
    
    minority_ratio = minority_count / total_count
    is_imbalanced = minority_ratio < threshold
    
    return {
        "is_imbalanced": is_imbalanced,
        "minority_ratio": minority_ratio,
        "class_counts": class_counts.to_dict(),
        "minority_count": minority_count,
        "majority_count": majority_count,
        "total_count": total_count,
        "message": f"Minority class represents {minority_ratio:.1%} of data"
    }


def apply_imbalance_handling(X, y, method="smote", random_state=42):
    """
    Apply imbalance handling techniques to the dataset.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: Method to use ('smote', 'undersampling', 'smote_tomek', 'class_weight')
        random_state: Random state for reproducibility
    
    Returns:
        tuple: (X_resampled, y_resampled, class_weights, info_message)
    """
    if not IMBALANCED_LEARN_AVAILABLE and method in ['smote', 'undersampling', 'smote_tomek']:
        return X, y, None, "imbalanced-learn package not available. Using original data."
    
    info_message = ""
    class_weights = None
    
    if method == "smote":
        try:
            # Use SMOTE to oversample minority class
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, len(X)-1))
            X_resampled, y_resampled = smote.fit_resample(X, y)
            info_message = f"SMOTE applied: {len(X)} → {len(X_resampled)} samples"
            return X_resampled, y_resampled, class_weights, info_message
        except Exception as e:
            st.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y, class_weights, f"SMOTE failed: {e}"
    
    elif method == "undersampling":
        try:
            # Use random undersampling to balance classes
            rus = RandomUnderSampler(random_state=random_state)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            info_message = f"Random undersampling applied: {len(X)} → {len(X_resampled)} samples"
            return X_resampled, y_resampled, class_weights, info_message
        except Exception as e:
            st.warning(f"Undersampling failed: {e}. Using original data.")
            return X, y, class_weights, f"Undersampling failed: {e}"
    
    elif method == "smote_tomek":
        try:
            # Use SMOTE + Tomek links for both over and under sampling
            smote_tomek = SMOTETomek(random_state=random_state)
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
            info_message = f"SMOTE + Tomek links applied: {len(X)} → {len(X_resampled)} samples"
            return X_resampled, y_resampled, class_weights, info_message
        except Exception as e:
            st.warning(f"SMOTE + Tomek failed: {e}. Using original data.")
            return X, y, class_weights, f"SMOTE + Tomek failed: {e}"
    
    elif method == "class_weight":
        # Compute class weights for algorithms that support it
        unique_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight_dict = dict(zip(unique_classes, class_weights))
        info_message = f"Class weights computed: {class_weight_dict}"
        return X, y, class_weight_dict, info_message
    
    else:
        # No imbalance handling
        return X, y, class_weights, "No imbalance handling applied"


def get_model_with_class_weights(model_option, class_weights=None):
    """
    Get a model instance with class weights if supported.
    
    Args:
        model_option: Name of the model
        class_weights: Dictionary of class weights
    
    Returns:
        model: Initialized model with class weights if supported
    """
    if model_option == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight=class_weights)
    elif model_option == "Ridge Classifier":
        return RidgeClassifier(class_weight=class_weights)
    elif model_option == "Decision Tree":
        return DecisionTreeClassifier(class_weight=class_weights)
    elif model_option == "Random Forest":
        return RandomForestClassifier(class_weight=class_weights)
    elif model_option == "Gradient Boosting Machines (GBMs)":
        return GradientBoostingClassifier()  # GBM doesn't support class_weight directly
    elif model_option == "XGBoost":
        if class_weights:
            # XGBoost uses scale_pos_weight for binary classification
            if len(class_weights) == 2:
                scale_pos_weight = class_weights[0] / class_weights[1]
                return XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                   scale_pos_weight=scale_pos_weight)
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif model_option == "Support Vector Machines (SVMs)":
        return svm.SVC(probability=True, class_weight=class_weights)
    else:
        # For models that don't support class weights, return the original implementation
        if model_option == "Lasso Regression":
            return Lasso()
        elif model_option == "K-Nearest Neighbors (KNN)":
            return KNeighborsClassifier()
        elif model_option == "Naive Bayes":
            return GaussianNB()
        elif model_option == "Linear Discriminant Analysis (LDA)":
            return LinearDiscriminantAnalysis()
        elif model_option == "Neural Network":
            return MLPClassifier(hidden_layer_sizes=(100,), activation="relu")
        else:
            raise ValueError(f"Unknown model option: {model_option}")


def optimize_for_sensitivity(y_true, y_scores, min_sensitivity=0.90, min_precision=0.10):
    """
    Find the optimal threshold that maximizes sensitivity while maintaining minimum precision.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities for the positive class
        min_sensitivity: Minimum sensitivity (recall) to achieve (default 90%)
        min_precision: Minimum precision to maintain (default 10%)
    
    Returns:
        dict: Contains optimal threshold, sensitivity, precision, and other metrics
    """
    if y_scores is None:
        return None
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Find thresholds that meet minimum sensitivity requirement
    high_sensitivity_mask = recall >= min_sensitivity
    
    if not np.any(high_sensitivity_mask):
        # If we can't achieve minimum sensitivity, find the best achievable
        best_recall_idx = np.argmax(recall)
        optimal_threshold = thresholds[best_recall_idx] if best_recall_idx < len(thresholds) else 0.5
        return {
            "optimal_threshold": optimal_threshold,
            "sensitivity": recall[best_recall_idx],
            "precision": precision[best_recall_idx],
            "f1_score": 2 * (precision[best_recall_idx] * recall[best_recall_idx]) /
                       (precision[best_recall_idx] + recall[best_recall_idx]) if (precision[best_recall_idx] + recall[best_recall_idx]) > 0 else 0,
            "achieved_target": False,
            "message": f"Could not achieve {min_sensitivity:.0%} sensitivity. Best achievable: {recall[best_recall_idx]:.1%}"
        }
    
    # Among high sensitivity options, find the one with best precision
    high_sensitivity_precision = precision[high_sensitivity_mask]
    high_sensitivity_recall = recall[high_sensitivity_mask]
    high_sensitivity_thresholds = thresholds[high_sensitivity_mask[:-1]]  # thresholds is one element shorter
    
    # Find the threshold that gives the best precision while maintaining high sensitivity
    if len(high_sensitivity_precision) > 0:
        best_precision_idx = np.argmax(high_sensitivity_precision)
        optimal_threshold = high_sensitivity_thresholds[best_precision_idx] if best_precision_idx < len(high_sensitivity_thresholds) else 0.5
        optimal_sensitivity = high_sensitivity_recall[best_precision_idx]
        optimal_precision = high_sensitivity_precision[best_precision_idx]
    else:
        optimal_threshold = 0.5
        optimal_sensitivity = recall_score(y_true, y_scores > optimal_threshold)
        optimal_precision = precision_score(y_true, y_scores > optimal_threshold)
    
    # Calculate F1 score
    f1 = 2 * (optimal_precision * optimal_sensitivity) / (optimal_precision + optimal_sensitivity) if (optimal_precision + optimal_sensitivity) > 0 else 0
    
    return {
        "optimal_threshold": optimal_threshold,
        "sensitivity": optimal_sensitivity,
        "precision": optimal_precision,
        "f1_score": f1,
        "achieved_target": optimal_sensitivity >= min_sensitivity and optimal_precision >= min_precision,
        "message": f"Optimized for {min_sensitivity:.0%} sensitivity. Achieved: {optimal_sensitivity:.1%} sensitivity, {optimal_precision:.1%} precision"
    }


def apply_sensitivity_threshold(y_scores, threshold):
    """
    Apply a sensitivity-optimized threshold to probability scores.
    
    Args:
        y_scores: Predicted probabilities
        threshold: Optimal threshold from sensitivity optimization
    
    Returns:
        np.array: Binary predictions using the optimal threshold
    """
    if y_scores is None or threshold is None:
        return None
    return (y_scores >= threshold).astype(int)


def explain_sensitivity_optimization():
    """
    Provide educational content about sensitivity optimization.
    
    Returns:
        str: Explanation text for sensitivity optimization
    """
    return """
    ### 🎯 What is Sensitivity Optimization?
    
    **Sensitivity (also called Recall or True Positive Rate)** measures how well your model detects positive cases:
    - **Sensitivity = True Positives / (True Positives + False Negatives)**
    - **High Sensitivity = Fewer missed cases (false negatives)**
    
    ### 🏥 Why Optimize for Sensitivity in Healthcare?
    
    In medical applications, **missing a positive case can be dangerous**:
    - ❌ **False Negative**: Missing a patient who has a disease
    - ✅ **False Positive**: Flagging a healthy patient for further testing
    
    **Examples where high sensitivity is crucial:**
    - Cancer screening (better to over-test than miss cancer)
    - Infectious disease detection (prevent spread)
    - Emergency triage (don't miss critical patients)
    
    ### ⚖️ The Trade-off: Sensitivity vs Precision
    
    - **↑ Higher Sensitivity** = Catch more positive cases, but more false alarms
    - **↑ Higher Precision** = Fewer false alarms, but might miss some cases
    
    **Sensitivity optimization finds the threshold that maximizes sensitivity while maintaining acceptable precision.**
    """


def display_sensitivity_results(sensitivity_results, y_true, y_pred_original, y_pred_optimized):
    """
    Display sensitivity optimization results with visualizations.
    
    Args:
        sensitivity_results: Results from optimize_for_sensitivity()
        y_true: True labels
        y_pred_original: Original predictions (0.5 threshold)
        y_pred_optimized: Sensitivity-optimized predictions
    """
    if sensitivity_results is None:
        st.warning("Sensitivity optimization not available for this model type.")
        return
    
    st.subheader("🎯 Sensitivity Optimization Results")
    
    # Display the main results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🎯 Optimal Threshold",
            f"{sensitivity_results['optimal_threshold']:.3f}",
            help="Threshold that maximizes sensitivity while maintaining precision"
        )
    
    with col2:
        st.metric(
            "📈 Sensitivity (Recall)",
            f"{sensitivity_results['sensitivity']:.1%}",
            help="Percentage of positive cases correctly identified"
        )
    
    with col3:
        st.metric(
            "🎯 Precision",
            f"{sensitivity_results['precision']:.1%}",
            help="Percentage of predicted positives that are actually positive"
        )
    
    # Status message
    if sensitivity_results['achieved_target']:
        st.success(f"✅ {sensitivity_results['message']}")
    else:
        st.warning(f"⚠️ {sensitivity_results['message']}")
    
    # Comparison table
    st.subheader("📊 Standard vs Sensitivity-Optimized Performance")
    
    # Calculate metrics for both approaches
    original_sensitivity = recall_score(y_true, y_pred_original)
    original_precision = precision_score(y_true, y_pred_original)
    original_f1 = f1_score(y_true, y_pred_original)
    
    optimized_sensitivity = recall_score(y_true, y_pred_optimized)
    optimized_precision = precision_score(y_true, y_pred_optimized)
    optimized_f1 = f1_score(y_true, y_pred_optimized)
    
    comparison_data = {
        "Metric": ["Sensitivity (Recall)", "Precision", "F1 Score"],
        "Standard (0.5 threshold)": [
            f"{original_sensitivity:.1%}",
            f"{original_precision:.1%}",
            f"{original_f1:.3f}"
        ],
        "Sensitivity-Optimized": [
            f"{optimized_sensitivity:.1%}",
            f"{optimized_precision:.1%}",
            f"{optimized_f1:.3f}"
        ],
        "Change": [
            f"{(optimized_sensitivity - original_sensitivity):.1%}" if optimized_sensitivity >= original_sensitivity else f"{(optimized_sensitivity - original_sensitivity):.1%}",
            f"{(optimized_precision - original_precision):.1%}" if optimized_precision >= original_precision else f"{(optimized_precision - original_precision):.1%}",
            f"{(optimized_f1 - original_f1):+.3f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    # Create confusion matrix comparison
    from sklearn.metrics import confusion_matrix
    
    st.subheader("🔍 Confusion Matrix Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Standard Threshold (0.5)**")
        cm_original = confusion_matrix(y_true, y_pred_original)
        
        cm_df_original = pd.DataFrame(
            cm_original,
            columns=["Predicted Negative", "Predicted Positive"],
            index=["Actually Negative", "Actually Positive"]
        )
        st.dataframe(cm_df_original)
        
        # Calculate rates
        tn, fp, fn, tp = cm_original.ravel()
        st.write(f"- **True Positives**: {tp} (correctly identified cases)")
        st.write(f"- **False Negatives**: {fn} (missed cases) ⚠️")
        st.write(f"- **False Positives**: {fp} (false alarms)")
        st.write(f"- **True Negatives**: {tn} (correctly identified non-cases)")
    
    with col2:
        st.write("**Sensitivity-Optimized Threshold**")
        cm_optimized = confusion_matrix(y_true, y_pred_optimized)
        
        cm_df_optimized = pd.DataFrame(
            cm_optimized,
            columns=["Predicted Negative", "Predicted Positive"],
            index=["Actually Negative", "Actually Positive"]
        )
        st.dataframe(cm_df_optimized)
        
        # Calculate rates
        tn, fp, fn, tp = cm_optimized.ravel()
        st.write(f"- **True Positives**: {tp} (correctly identified cases)")
        st.write(f"- **False Negatives**: {fn} (missed cases) ⚠️")
        st.write(f"- **False Positives**: {fp} (false alarms)")
        st.write(f"- **True Negatives**: {tn} (correctly identified non-cases)")
    
    # Show the improvement in missed cases
    original_fn = confusion_matrix(y_true, y_pred_original)[1, 0]
    optimized_fn = confusion_matrix(y_true, y_pred_optimized)[1, 0]
    
    if optimized_fn < original_fn:
        st.success(f"🎉 **Improvement**: Reduced missed cases from {original_fn} to {optimized_fn} (saved {original_fn - optimized_fn} cases from being missed)")
    elif optimized_fn == original_fn:
        st.info("ℹ️ No change in missed cases between standard and optimized thresholds")
    else:
        st.warning(f"⚠️ **Trade-off**: Missed cases increased from {original_fn} to {optimized_fn}, but overall sensitivity strategy may still be beneficial")


def display_imbalance_warning(imbalance_info):
    """
    Display warning and advice for imbalanced datasets.
    
    Args:
        imbalance_info: Dictionary with imbalance information
    """
    if imbalance_info["is_imbalanced"]:
        st.warning(f"""
        ⚠️ **Imbalanced Dataset Detected!**
        
        Your target variable is imbalanced:
        - Minority class: {imbalance_info['minority_count']} samples ({imbalance_info['minority_ratio']:.1%})
        - Majority class: {imbalance_info['majority_count']} samples ({(1-imbalance_info['minority_ratio']):.1%})
        
        **Recommendations:**
        1. **Enable imbalance handling** using the options below
        2. **Focus on F1-score and PR AUC** rather than accuracy
        3. **Consider stratified sampling** to maintain class distribution in train/test splits
        
        **Why this matters:** Models trained on imbalanced data often achieve high accuracy by simply predicting the majority class, but perform poorly on the minority class (which is often the class of interest in medical data).
        """)
        
        # Display class distribution visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        class_counts = pd.Series(imbalance_info["class_counts"])
        bars = ax.bar(range(len(class_counts)), class_counts.values,
                     color=['lightcoral', 'lightblue'])
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution in Target Variable')
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels([f'Class {k}' for k in class_counts.index])
        
        # Add count labels on bars
        for bar, count in zip(bars, class_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.success(f"""
        ✅ **Balanced Dataset**
        
        Your target variable appears balanced:
        - Minority class: {imbalance_info['minority_count']} samples ({imbalance_info['minority_ratio']:.1%})
        - Majority class: {imbalance_info['majority_count']} samples ({(1-imbalance_info['minority_ratio']):.1%})
        
        No special handling for class imbalance is required.
        """)


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

            precision, recall, _ = precision_recall_curve(
                y_true, y_scores, pos_label=pos_label
            )
        else:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)

        pr_auc = auc(recall, precision)

        # Display classification metrics
        st.info(
            f"**Your Model Metrics ({set_name} Set):** F1 score: {f1:.2f}, Accuracy: {accuracy:.2f}, ROC AUC: {roc_auc:.2f}, PR AUC: {pr_auc:.2f}"
        )
    except ValueError:
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

        precision, recall, _ = precision_recall_curve(
            y_true, y_scores, pos_label=pos_label
        )
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
                    df.loc[:, col] = df[col].map({most_freq: 0, least_freq: 1, "F": 0})

                    included_cols.append(col)
                else:  # Multivariate case
                    excluded_cols.append(col)
            elif df[col].dtype in ["int64", "float64"]:  # Numerical case
                if df[col].isnull().values.any():
                    mean_imputer = SimpleImputer(strategy="mean")
                    df.loc[:, col] = mean_imputer.fit_transform(df[[col]]).ravel()
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
                    df.loc[:, col] = df[col].map({most_freq: 0, least_freq: 1})
                    included_cols.append(col)
                else:  # Multivariate case
                    excluded_cols.append(col)
            elif df[col].dtype in ["int64", "float64"]:  # Numerical case
                if df[col].isnull().values.any():
                    mean_imputer = SimpleImputer(strategy="mean")
                    df.loc[:, col] = mean_imputer.fit_transform(df[[col]]).ravel()
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
    perform_shapley=False,
    imbalance_method=None,
    stratify_split=False,
    optimize_sensitivity=False,
    target_sensitivity=0.90,
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
        perform_shapley: bool, whether to perform SHAP analysis
    Returns:
        dict with model, metrics, predictions, etc.
    """
    # Validate input data
    is_valid, message = data_validation.check_data_size(df, "ml_input_data")
    if not is_valid:
        st.error(f"ML pipeline data validation failed: {message}")
        return None
    
    # Log data size for monitoring
    monitoring.log_dataframe_size(df, "ml_input_data")
    
    # Feature/target selection updates
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle target variable encoding if it's categorical
    target_label_mapping = None
    if y.dtype == "object" or isinstance(y.dtype, pd.CategoricalDtype):
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
            target_label_mapping = dict(
                zip(
                    label_encoder.classes_,
                    label_encoder.transform(label_encoder.classes_),
                )
            )

    # Detect imbalance in target variable
    imbalance_info = detect_imbalance(y)
    
    # Apply imbalance handling if requested
    class_weights = None
    imbalance_info_message = "No imbalance handling applied"
    
    if imbalance_method and imbalance_method != "none":
        X_processed, y_processed, class_weights, imbalance_info_message = apply_imbalance_handling(
            X, y, method=imbalance_method, random_state=random_state
        )
        # Update X and y with processed versions
        X = X_processed
        y = y_processed

    # Handle categorical variables in features
    # Process categorical variables using the utility function
    X, mapping_definitions = utils.preprocess_categorical_vars(X, track_mapping=True)
    
    # Handle remaining categorical variables (non-binary)
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        # One-hot encode
        X = pd.get_dummies(X, columns=[col], drop_first=True)

    # Store feature names before any transformations that might convert to numpy arrays
    feature_names = list(X.columns)

    # Normalization/scaling
    if normalization_option == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Keep as DataFrame to preserve column names for SHAP
        X = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    elif normalization_option in ["l1", "l2"]:
        X_normalized = normalize(X, norm=normalization_option)
        # Keep as DataFrame to preserve column names for SHAP
        X = pd.DataFrame(X_normalized, columns=feature_names, index=X.index)
    # If no scaling/normalization, X remains as DataFrame

    # Train/test split with optional stratification
    if stratify_split and len(np.unique(y)) > 1:
        # Use stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # Model selection with class weights support
    if class_weights and imbalance_method == "class_weight":
        model = get_model_with_class_weights(model_option, class_weights)
    else:
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
        y_scores = (
            model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None
        )
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = predictions

    # Sensitivity optimization
    sensitivity_results = None
    predictions_optimized = None
    if optimize_sensitivity and len(np.unique(y_train)) == 2 and y_scores is not None:
        sensitivity_results = optimize_for_sensitivity(
            y_test, y_scores, min_sensitivity=target_sensitivity, min_precision=0.10
        )
        if sensitivity_results and sensitivity_results["optimal_threshold"] is not None:
            predictions_optimized = apply_sensitivity_threshold(
                y_scores, sensitivity_results["optimal_threshold"]
            )

    # Metrics & SHAP
    metrics = {}
    shap_values = None
    explainer = None

    if len(np.unique(y_train)) == 2:
        # Binary classification metrics
        metrics["accuracy"] = accuracy_score(y_test, predictions)
        metrics["f1"] = f1_score(y_test, predictions)
        if y_scores is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_test, y_scores)

                # Handle precision-recall curve with proper pos_label
                unique_labels = np.unique(y_test)
                if np.issubdtype(y_test.dtype, np.number):
                    pos_label = max(unique_labels)
                else:
                    pos_label = sorted(unique_labels)[
                        1
                    ]  # Use second label alphabetically

                precision, recall, _ = precision_recall_curve(
                    y_test, y_scores, pos_label=pos_label
                )
                metrics["pr_auc"] = auc(recall, precision)
            except Exception:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
        metrics["confusion_matrix"] = confusion_matrix(y_test, predictions)
    else:
        # Regression metrics (add more as needed)
        metrics["accuracy"] = model.score(X_test, y_test)

    if perform_shapley:
        # Reduce background sample size to improve performance
        # Use shap.sample to get a smaller representative sample
        if X_train.shape[0] > 100:
            background_sample = shap.sample(
                X_train, 100
            )  # Use 100 samples instead of 50 for better representation
        else:
            background_sample = X_train

        if isinstance(
            model,
            (
                DecisionTreeClassifier,
                RandomForestClassifier,
                GradientBoostingClassifier,
                XGBClassifier,
            ),
        ):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            # For non-tree models, use KernelExplainer
            # Remove the l1_reg limitation to allow all features to be explained
            try:
                # First try with predict_proba for classification models
                if hasattr(model, "predict_proba") and len(np.unique(y_train)) == 2:
                    explainer = shap.KernelExplainer(
                        model.predict_proba, background_sample
                    )
                    shap_values = explainer.shap_values(X_test)
                    # For binary classification, take the positive class SHAP values
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]
                else:
                    # For regression or models without predict_proba
                    explainer = shap.KernelExplainer(model.predict, background_sample)
                    shap_values = explainer.shap_values(X_test)
            except Exception as e:
                st.warning(
                    f"SHAP analysis failed: {e}. Trying with reduced sample size."
                )
                try:
                    # Fallback: use smaller sample and limit features if needed
                    smaller_background = shap.sample(X_train, 25)
                    explainer = shap.KernelExplainer(model.predict, smaller_background)
                    shap_values = explainer.shap_values(
                        X_test[:10]
                    )  # Analyze only first 10 test samples
                except Exception as e2:
                    st.warning(f"SHAP analysis failed completely: {e2}")
                    explainer = None
                    shap_values = None

    # Log results for monitoring
    monitoring.log_dataframe_size(X_test, "ml_test_features")
    if y_scores is not None:
        monitoring.log_dataframe_size(pd.DataFrame(y_scores), "ml_predictions")
    
    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions,
        "y_test": y_test,
        "y_scores": y_scores,
        "X_test": X_test,
        "explainer": explainer,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "imbalance_info": imbalance_info,
        "imbalance_method": imbalance_method,
        "imbalance_message": imbalance_info_message,
        "class_weights": class_weights,
        "sensitivity_results": sensitivity_results,
        "predictions_optimized": predictions_optimized,
        "optimize_sensitivity": optimize_sensitivity,
        "target_sensitivity": target_sensitivity,
    }
