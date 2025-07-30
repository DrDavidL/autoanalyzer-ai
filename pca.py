import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import category_encoders as ce


def preprocess_for_pca(df):
    included_cols = []
    excluded_cols = []
    binary_mapping = {}  # initialize empty dict for binary mapping
    binary_encoded_vars = []  # initialize empty list for binary encoded vars

    # Create a binary encoder
    bin_encoder = ce.BinaryEncoder()

    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == "object":
            unique = df[col].nunique()

            # For binary categorical columns
            if unique == 2:
                most_freq = df[col].value_counts().idxmax()
                least_freq = df[col].value_counts().idxmin()
                df[col] = df[col].map({most_freq: 0, least_freq: 1})
                binary_mapping[col] = {
                    most_freq: 0,
                    least_freq: 1,
                }  # add mapping to dict
                included_cols.append(col)

            # For categorical columns with less than 15 unique values
            elif 2 < unique <= 15:
                try:
                    # Perform binary encoding
                    df_transformed = bin_encoder.fit_transform(df[col])
                    # Drop the original column from df
                    df.drop(columns=[col], inplace=True)
                    # Join the transformed data to df
                    df = pd.concat([df, df_transformed], axis=1)
                    # Add transformed columns to binary encoded vars list and included_cols
                    transformed_cols = df_transformed.columns.tolist()
                    binary_encoded_vars.extend(transformed_cols)
                    included_cols.extend(transformed_cols)
                except Exception as e:
                    st.write(f"Failure in encoding {col} due to {str(e)}")
                    excluded_cols.append(col)
            else:
                excluded_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            included_cols.append(col)
        else:
            excluded_cols.append(col)

    # Display binary mappings and binary encoded variables in streamlit
    if binary_mapping:
        st.write("Binary Mappings: ", binary_mapping)
    if binary_encoded_vars:
        st.write("Binary Encoded Variables: ", binary_encoded_vars)

    return df[included_cols], included_cols, excluded_cols


def create_scree_plot(df):
    temp_df_pca, included_cols, excluded_cols = preprocess_for_pca(df)

    # Standardize the features
    x = StandardScaler().fit_transform(temp_df_pca)

    # Create a PCA instance: n_components should be None so variance is preserved from all initial features
    pca = PCA(n_components=None)
    pca.fit_transform(x)

    # Scree plot
    fig, ax = plt.subplots()
    ax.plot(
        np.arange(1, len(pca.explained_variance_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
    )
    ax.set_title("Cumulative Explained Variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    st.pyplot(fig)
    return fig


def perform_pca_plot(df):
    st.write(
        "Note: For this PCA analysis, categorical columns with 2 values are mapped to 1 and 0. Categories with more than 2 values have been binary encoded."
    )
    temp_df_pca, included_cols, excluded_cols = preprocess_for_pca(df)

    # Standardize the features
    x = StandardScaler().fit_transform(temp_df_pca)

    # Select the target column for PCA
    cols_2_15_unique_vals = [
        col for col in included_cols if 2 <= df[col].nunique() <= 15
    ]
    target_col_pca = st.selectbox(
        "Select the target column for PCA", cols_2_15_unique_vals
    )

    num_unique_targets = df[
        target_col_pca
    ].nunique()  # Calculate the number of unique targets

    # Ask the user to request either 2 or 3 component PCA
    n_components = st.selectbox("Select the number of PCA components (2 or 3)", [2, 3])

    # Create a PCA instance
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)

    # Depending on user choice, plot the appropriate PCA
    if n_components == 2:
        principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])
    else:
        principalDf = pd.DataFrame(
            data=principalComponents, columns=["PC1", "PC2", "PC3"]
        )

    finalDf = pd.concat([principalDf, df[[target_col_pca]]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    if n_components == 2:
        ax = fig.add_subplot(111)
    else:
        # ax = Axes3D(fig)
        ax = plt.axes(projection="3d")
        ax.set_zlabel("Principal Component 3", fontsize=15)

    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)

    ax.set_title(f"{n_components} component PCA", fontsize=20)

    targets = finalDf[target_col_pca].unique().tolist()
    colors = sns.color_palette("husl", n_colors=num_unique_targets)
    # finalDf

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf[target_col_pca] == target
        if n_components == 2:
            ax.scatter(
                finalDf.loc[indicesToKeep, "PC1"],
                finalDf.loc[indicesToKeep, "PC2"],
                c=[color],
                s=50,
            )
        else:
            ax.scatter(
                finalDf.loc[indicesToKeep, "PC1"],
                finalDf.loc[indicesToKeep, "PC2"],
                finalDf.loc[indicesToKeep, "PC3"],
                c=[color],
                s=50,
            )

    ax.legend(targets)

    # Make a scree plot

    # Display the plot using Streamlit
    st.pyplot(fig)
    st.subheader("Use the PCA Updated Dataset for Machine Learning")
    st.write(
        "Download the current plot if you'd like to save it! Then, follow steps to apply machine learning to your PCA modified dataset."
    )
    st.info(
        "Step 1. Click Button to use the PCA Dataset for ML. Step 2. Select Modified Dataframe on left sidebar and switch to the Machine Learning tab. (You'll overfit if you click below again!)"
    )
    if st.button("Use PCA Updated dataset on Machine Learning Tab"):
        st.session_state.modified_df = finalDf

    return fig
