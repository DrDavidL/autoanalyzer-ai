# Utility functions for AutoAnalyzer Streamlit app

import base64
import tempfile
import streamlit as st
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
    tmpdirname = tempfile.mkdtemp(prefix="output_")
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

        sampled_data[col] = samples

    # Create and return the new DataFrame from the sampled data
    return pd.DataFrame(sampled_data)
