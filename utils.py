# Utility functions for AutoAnalyzer Streamlit app

import os
import base64
import tempfile
import streamlit as st
import openai
import random
import io

def is_valid_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.Completion.create(
            model="text-davinci-003", prompt="Hello world"
        )["choices"][0]["text"]
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