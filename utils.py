# Utility functions for AutoAnalyzer Streamlit app

import base64
import tempfile
import streamlit as st
import monitoring
import openai
import io
import pandas as pd
import numpy as np


def is_valid_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.Completion.create(model="text-davinci-003", prompt="Hello world")[
            "choices"
        ][0]["text"]
        return True
    except Exception:
        return False


def is_bytes_like(obj):
    return isinstance(obj, (bytes, bytearray, memoryview))


def get_output_path():
    # Create a monitored temporary directory
    tmpdirname = tempfile.mkdtemp(prefix="output_")
    monitoring.monitor.register_temp_dir(tmpdirname)
    return tmpdirname


def get_download_link(file_path, file_type):
    try:
        with open(file_path, "rb") as file:
            contents = file.read()
        base64_data = base64.b64encode(contents).decode("utf-8")
        download_link = f'<a href="data:application/octet-stream;base64,{base64_data}" download="tableone_results.{file_type}">Click here to download the TableOne results in {file_type} format.</a>'
        return download_link
    except Exception as e:
        st.warning(f"Could not create download link: {e}")
        return ""


def save_image(plot, filename):
    try:
        if is_bytes_like(plot):
            img = io.BytesIO(plot)
        else:
            img = io.BytesIO()
            plot.savefig(img, format="png")
        img.seek(0)
        st.download_button(
            label="Download your plot.",
            data=img,
            file_name=filename,
            mime="image/png",
        )
    except Exception as e:
        st.warning(f"Could not save or download plot: {e}")


def generate_regression_equation(intercept, coef, x_cols):
    """
    Returns a string representing the regression equation.
    intercept: float
    coef: array-like
    x_cols: list of str
    """
    terms = [f"{coef[i]:.4f}*{x}" for i, x in enumerate(x_cols)]
    equation = f"y = {' + '.join(terms)} + {intercept:.4f}"
    return equation


def create_sampled_header(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Creates a DataFrame with the original headers but with randomly sampled
    values from each corresponding column, completely decorrelating the data.
    
    For numerical columns, values are altered by -20% to +20% to enhance privacy.
    For categorical columns, values are sent as is.

    This is useful for providing a privacy-preserving sample of a dataset's
    structure and value types to an LLM.

    Args:
        df (pd.DataFrame): The original DataFrame.
        n (int): The number of random samples to draw from each column.

    Returns:
        pd.DataFrame: A new DataFrame where each column contains 'n' random
                      samples from the original column.
    """
    # Create a dictionary to hold the sampled data
    sampled_data = {}

    # Iterate over each column in the original DataFrame
    for col in df.columns:
        # Drop missing values and get the unique values from the column
        unique_vals = df[col].dropna().unique()

        # If the column has no non-null unique values, fill with NaN
        if len(unique_vals) == 0:
            samples = [np.nan] * n
        else:
            # Randomly choose 'n' values from the unique values.
            # 'replace=True' allows sampling even if n > number of unique values.
            samples = np.random.choice(unique_vals, size=n, replace=True)
            
            # For numerical columns, add privacy by altering values by -20% to +20%
            if pd.api.types.is_numeric_dtype(df[col]):
                # Apply random alteration between -20% and +20%
                alteration_factors = np.random.uniform(0.8, 1.2, size=len(samples))
                samples = samples * alteration_factors

        sampled_data[col] = samples

    # Create and return the new DataFrame from the sampled data
    return pd.DataFrame(sampled_data)


def map_binary_categorical_vars(df, track_mapping=True):
    """
    Map binary categorical variables to 0 (most frequent) and 1 (least frequent).
    
    Args:
        df (pd.DataFrame): DataFrame to process
        track_mapping (bool): Whether to track and return the mapping definitions
    
    Returns:
        pd.DataFrame: Processed DataFrame with binary variables mapped
        dict (optional): Mapping definitions if track_mapping=True
    """
    df_processed = df.copy()
    binary_mapping = {}
    
    # Identify binary categorical columns
    for col in df_processed.columns:
        # Check if column is categorical or object type with exactly 2 unique values
        if (df_processed[col].dtype == 'object' or
            isinstance(df_processed[col].dtype, pd.CategoricalDtype)) and \
           df_processed[col].nunique() == 2:
            
            # Get value counts to determine most and least frequent values
            value_counts = df_processed[col].value_counts()
            most_frequent_value = value_counts.idxmax()
            least_frequent_value = value_counts.idxmin()
            
            # Create mapping dictionary
            mapping = {most_frequent_value: 0, least_frequent_value: 1}
            
            # Apply mapping to the column
            df_processed.loc[:, col] = df_processed[col].map(mapping)
            
            # Track mapping if requested
            if track_mapping:
                binary_mapping[col] = mapping
    
    # Return processed DataFrame and mapping if tracking is enabled
    if track_mapping:
        return df_processed, binary_mapping
    else:
        return df_processed


def preprocess_categorical_vars(df, max_onehot_unique=6, track_mapping=True):
    """
    Preprocess categorical variables by:
    1. Mapping binary categorical variables to 0/1
    2. One-hot encoding categorical variables with up to max_onehot_unique unique values
    
    Args:
        df (pd.DataFrame): DataFrame to process
        max_onehot_unique (int): Maximum number of unique values for one-hot encoding
        track_mapping (bool): Whether to track and return the mapping definitions
    
    Returns:
        pd.DataFrame: Processed DataFrame with categorical variables handled
        dict (optional): Mapping definitions if track_mapping=True
    """
    df_processed = df.copy()
    mapping_definitions = {}
    
    # Handle binary categorical variables
    df_processed, binary_mapping = map_binary_categorical_vars(df_processed, track_mapping=True)
    if track_mapping and binary_mapping:
        mapping_definitions['binary'] = binary_mapping
    
    # Identify categorical columns for potential one-hot encoding
    categorical_cols = []
    for col in df_processed.columns:
        if (df_processed[col].dtype == 'object' or
            isinstance(df_processed[col].dtype, pd.CategoricalDtype)):
            categorical_cols.append(col)
    
    # One-hot encode categorical variables with appropriate number of unique values
    onehot_mapping = {}
    for col in categorical_cols:
        unique_count = df_processed[col].nunique()
        if 2 < unique_count <= max_onehot_unique:  # More than binary but within limit
            # One-hot encode the column
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            # Remove the original column
            df_processed = df_processed.drop(columns=[col])
            # Add the one-hot encoded columns
            df_processed = pd.concat([df_processed, dummies], axis=1)
            
            # Track one-hot encoding if requested
            if track_mapping:
                onehot_mapping[col] = list(dummies.columns)
    
    # Add one-hot encoding information to mapping definitions
    if track_mapping and onehot_mapping:
        mapping_definitions['onehot'] = onehot_mapping
    
    # Return processed DataFrame and mapping if tracking is enabled
    if track_mapping:
        return df_processed, mapping_definitions
    else:
        return df_processed
