# Data processing functions for AutoAnalyzer

import numpy as np
import streamlit as st


def all_categorical(df):
    categ_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [
        col
        for col in df.columns
        if df[col].nunique() == 2 and df[col].dtype != "object"
    ]
    filtered_categorical_cols = [col for col in categ_cols if df[col].nunique() <= 15]
    all_categ = filtered_categorical_cols + numeric_cols
    return all_categ


def all_numerical(df):
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() == 2:
            unique_values = df[col].unique()
            if 0 in unique_values and 1 in unique_values:
                continue
            value_counts = df[col].value_counts()
            most_frequent_value = value_counts.idxmax()
            least_frequent_value = value_counts.idxmin()
            if most_frequent_value != 0 and least_frequent_value != 1:
                df[col] = np.where(df[col] == most_frequent_value, 0, 1)
                st.write(
                    f"Replaced most frequent value '{most_frequent_value}' with 0 and least frequent value '{least_frequent_value}' with 1 in column '{col}'."
                )
                numerical_cols.append(col)
    return numerical_cols


def filter_dataframe(df):
    columns = df.columns
    dtypes = df.dtypes
    excluded_columns = st.multiselect("Exclude Columns", columns)
    filtered_df = df.copy()
    filtered_df = filtered_df.drop(excluded_columns, axis=1)
    filtered_columns = filtered_df.columns
    filtered_dtypes = filtered_df.dtypes
    numerical_columns = [
        col
        for col, dtype in zip(filtered_columns, filtered_dtypes)
        if dtype in ["int64", "float64"]
    ]
    for col in numerical_columns:
        min_val = filtered_df[col].min()
        max_val = filtered_df[col].max()
        st.write(f"**{col}**")
        min_range, max_range = st.slider(
            "", min_val, max_val, (min_val, max_val), key=col
        )
        if min_range > min_val or max_range < max_val:
            filtered_df = filtered_df[
                (filtered_df[col] >= min_range) & (filtered_df[col] <= max_range)
            ]
    categorical_columns = [
        col
        for col, dtype in zip(filtered_columns, filtered_dtypes)
        if dtype == "object"
    ]
    for col in categorical_columns:
        unique_values = filtered_df[col].unique()
        selected_values = st.multiselect(col, unique_values, unique_values)
        if len(selected_values) < len(unique_values):
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    return filtered_df
