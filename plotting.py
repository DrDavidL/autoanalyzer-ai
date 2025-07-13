# Plotting functions for AutoAnalyzer

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_boxplot(df, numeric_col, categorical_col, show_points=False):
    fig, ax = plt.subplots()
    sns.boxplot(x=categorical_col, y=numeric_col, data=df, ax=ax)
    if show_points:
        sns.stripplot(x=categorical_col, y=numeric_col, data=df, color='black', alpha=0.3, ax=ax)
    st.pyplot(fig)
    return fig

def create_violinplot(df, numeric_col, categorical_col):
    fig, ax = plt.subplots()
    sns.violinplot(x=categorical_col, y=numeric_col, data=df, ax=ax)
    st.pyplot(fig)
    return fig

def create_scatterplot(df, scatter_x, scatter_y):
    fig, ax = plt.subplots()
    sns.scatterplot(x=scatter_x, y=scatter_y, data=df, ax=ax)
    st.pyplot(fig)

def plot_corr(df):
    fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    return fig

def plot_pie(df, col_name):
    fig, ax = plt.subplots()
    df[col_name].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)
    return fig

def plot_categorical(df, col_name):
    fig, ax = plt.subplots()
    df[col_name].value_counts().plot.bar(ax=ax)
    st.pyplot(fig)
    return fig

def plot_numeric(df, col_name):
    fig, ax = plt.subplots()
    df[col_name].plot.hist(ax=ax)
    st.pyplot(fig)
    return fig

def plot_missing_data(df):
    import missingno as msno
    fig = msno.matrix(df)
    st.pyplot(fig.figure)


def plot_survival_curve(df, duration_col, event_col, group_col=None):
    """
    Plot Kaplan-Meier survival curves using lifelines.
    Args:
        df: pandas DataFrame
        duration_col: column name for time-to-event or censoring
        event_col: column name for event occurrence (1=event, 0=censored)
        group_col: optional column name for stratified curves
    """
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()
    if group_col and group_col in df.columns:
        for name, grouped_df in df.groupby(group_col):
            kmf.fit(grouped_df[duration_col], grouped_df[event_col], label=str(name))
            kmf.plot_survival_function(ax=ax)
    else:
        kmf.fit(df[duration_col], df[event_col])
        kmf.plot_survival_function(ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Kaplan-Meier Survival Curve')
    st.pyplot(fig)
    return fig

def plot_cox_proportional_hazards(df, duration_col, event_col, covariate_cols):
    """
    Fit and plot Cox Proportional Hazards model using lifelines.
    Args:
        df: pandas DataFrame
        duration_col: column name for time-to-event or censoring
        event_col: column name for event occurrence (1=event, 0=censored)
        covariate_cols: list of covariate column names
    """
    from lifelines import CoxPHFitter
    import matplotlib.pyplot as plt
    import streamlit as st
    cph = CoxPHFitter()
    # Prepare the dataframe for CoxPHFitter
    data = df[[duration_col, event_col] + covariate_cols].dropna()
    cph.fit(data, duration_col=duration_col, event_col=event_col)
    st.write("Cox Proportional Hazards Model Summary:")
    st.write(cph.summary)
    fig, ax = plt.subplots()
    cph.plot(ax=ax)
    st.pyplot(fig)
    return cph, fig

def plot_multiple_linear_regression(df, x_cols, y_col):
    """
    Plot actual vs predicted for multiple linear regression and display coefficients and summary.
    Args:
        df: pd.DataFrame
        x_cols: list of str, feature columns
        y_col: str, target column
    """
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression

    X = df[x_cols]
    y = df[y_col]
    # Fit model with sklearn
    regr = LinearRegression()
    regr.fit(X, y)
    y_pred = regr.predict(X)

    # Plot actual vs predicted
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Multiple Linear Regression: Actual vs Predicted')
    st.pyplot(fig)

    # Show coefficients
    st.write('Intercept:', regr.intercept_)
    st.write('Coefficients:')
    st.write(dict(zip(x_cols, regr.coef_)))

    # Statsmodels summary
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    st.write(model.summary())

    return fig, regr.intercept_, regr.coef_, model.summary()
