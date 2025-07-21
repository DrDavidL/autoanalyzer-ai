# Streamlit UI helper functions for AutoAnalyzer

import streamlit as st
import pandas as pd

def df_download_options(df, report_type):
    import random
    format = st.radio(
        "Select the format for your report:",
        ("csv", "json", "html"),
        key=f"report_format_{report_type}",
        horizontal=True,
    )
    file_name = f"{report_type}.{format}"
    try:
        if format == "csv":
            data = df.to_csv(index=True)
            mime = "text/csv"
        elif format == "json":
            data = df.to_json(orient="records")
            mime = "application/json"
        else:  # html
            data = df.to_html()
            mime = "text/html"
        st.download_button(
            label="Download your report.",
            data=data,
            file_name=file_name,
            mime=mime,
        )
    except Exception as e:
        st.warning(f"Could not generate download file: {e}")
