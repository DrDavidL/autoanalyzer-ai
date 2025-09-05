# Plotting functions for AutoAnalyzer

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import utils


def create_boxplot(df, numeric_col, categorical_col, show_points=False):
    fig, ax = plt.subplots()
    sns.boxplot(x=categorical_col, y=numeric_col, data=df, ax=ax)
    if show_points:
        sns.stripplot(
            x=categorical_col, y=numeric_col, data=df, color="black", alpha=0.3, ax=ax
        )
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
    # Process categorical variables using the utility function
    df_processed = utils.preprocess_categorical_vars(df, track_mapping=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))  # Set figure size to control overall size
    corr = df_processed.corr()
    
    # Create heatmap with smaller font size to prevent overlap
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                annot_kws={"size": 8},  # Smaller font size for annotations
                cbar_kws={"shrink": 0.8})  # Smaller color bar
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    st.pyplot(fig)
    return fig


def plot_pie(df, col_name):
    fig, ax = plt.subplots()
    df[col_name].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
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
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Kaplan-Meier Survival Curve")
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
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Multiple Linear Regression: Actual vs Predicted")
    st.pyplot(fig)

    # Show coefficients
    st.write("Intercept:", regr.intercept_)
    st.write("Coefficients:")
    st.write(dict(zip(x_cols, regr.coef_)))

    # Statsmodels summary
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    st.write(model.summary())

    return fig, regr.intercept_, regr.coef_, model.summary()


def create_3d_scatter(df, x_col, y_col, z_col, color_col=None, size_col=None):
    """
    Create an interactive 3D scatter plot using Plotly.
    Args:
        df: pandas DataFrame
        x_col: str, column for x-axis
        y_col: str, column for y-axis  
        z_col: str, column for z-axis
        color_col: str, optional column for color coding
        size_col: str, optional column for size mapping
    """
    # Define better color schemes
    color_schemes = {
        'categorical': 'Set3',  # Good categorical separation
        'continuous': 'Viridis'  # Good continuous color scale
    }
    
    # Determine if color column is categorical or continuous
    color_discrete_map = None
    color_continuous_scale = None
    
    if color_col:
        if df[color_col].dtype == 'object' or df[color_col].nunique() < 10:
            color_discrete_map = color_schemes['categorical']
        else:
            color_continuous_scale = color_schemes['continuous']
    
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, 
                       color=color_col, size=size_col,
                       hover_data=[col for col in df.columns if col not in [x_col, y_col, z_col]],
                       title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
                       color_discrete_sequence=px.colors.qualitative.Set3 if color_col and (df[color_col].dtype == 'object' or df[color_col].nunique() < 10) else None,
                       color_continuous_scale=color_continuous_scale)
    
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        height=600,
        width=800
    )
    
    # Make markers larger and more visible
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white')))
    
    st.plotly_chart(fig, use_container_width=True)
    return fig


def create_3d_surface(df, x_col, y_col, z_col):
    """
    Create a 3D surface plot using Plotly.
    Args:
        df: pandas DataFrame
        x_col: str, column for x-axis
        y_col: str, column for y-axis
        z_col: str, column for z-axis (height)
    """
    import numpy as np
    from scipy.interpolate import griddata
    
    try:
        # Remove any null values
        df_clean = df[[x_col, y_col, z_col]].dropna()
        
        if len(df_clean) < 4:
            st.error("Need at least 4 data points for surface plot after removing null values.")
            return None
            
        # Get the data points
        x = df_clean[x_col].values
        y = df_clean[y_col].values
        z = df_clean[z_col].values
        
        # Create a grid for interpolation
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        X, Y = np.meshgrid(xi, yi)
        
        # Interpolate Z values on the grid
        Z = griddata((x, y), z, (X, Y), method='linear')
        
        # Replace NaN values with nearest interpolation where linear fails
        mask = np.isnan(Z)
        if mask.any():
            Z_nearest = griddata((x, y), z, (X, Y), method='nearest')
            Z[mask] = Z_nearest[mask]
        
        fig = go.Figure(data=[go.Surface(
            z=Z, 
            x=xi, 
            y=yi,
            colorscale='Viridis',
            showscale=True
        )])
        
        fig.update_layout(
            title=f"3D Surface Plot: {z_col} by {x_col} and {y_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return fig
    except Exception as e:
        st.error(f"Could not create 3D surface plot: {e}")
        st.info("Surface plots work best with sufficient data points distributed across the X-Y plane. Try using columns with good coverage of the coordinate space.")
        return None


def create_3d_line(df, x_col, y_col, z_col, color_col=None):
    """
    Create a 3D line plot (trajectory) using Plotly.
    Args:
        df: pandas DataFrame
        x_col: str, column for x-axis
        y_col: str, column for y-axis
        z_col: str, column for z-axis
        color_col: str, optional column for color gradient along line
    """
    if color_col and color_col in df.columns:
        # Use px.line_3d for colored lines
        fig = px.line_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                        title=f"3D Trajectory: {x_col}, {y_col}, {z_col}",
                        color_discrete_sequence=px.colors.qualitative.Set3)
    else:
        # Create a gradient colored line based on sequence
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x_col], y=df[y_col], z=df[z_col],
            mode='lines+markers',
            line=dict(
                width=6,
                color=list(range(len(df))),  # Color by sequence
                colorscale='Rainbow',
                showscale=True,
                colorbar=dict(title="Sequence")
            ),
            marker=dict(
                size=4,
                color=list(range(len(df))),
                colorscale='Rainbow',
                showscale=False
            )
        )])
        
        fig.update_layout(
            title=f"3D Trajectory: {x_col}, {y_col}, {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            )
        )
    
    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig, use_container_width=True)
    return fig


def create_3d_mesh(df, x_col, y_col, z_col, intensity_col=None):
    """
    Create a 3D mesh plot using Plotly.
    Args:
        df: pandas DataFrame
        x_col: str, column for x-axis
        y_col: str, column for y-axis
        z_col: str, column for z-axis
        intensity_col: str, optional column for color intensity
    """
    try:
        # Clean data
        df_clean = df[[x_col, y_col, z_col]].dropna()
        if intensity_col:
            df_clean = df[[x_col, y_col, z_col, intensity_col]].dropna()
        
        if len(df_clean) < 4:
            st.error("Need at least 4 data points for mesh plot after removing null values.")
            return None
        
        if intensity_col and intensity_col in df_clean.columns:
            fig = go.Figure(data=[go.Mesh3d(
                x=df_clean[x_col], y=df_clean[y_col], z=df_clean[z_col],
                intensity=df_clean[intensity_col],
                colorscale='Plasma',
                opacity=0.7,
                showscale=True,
                colorbar=dict(title=intensity_col)
            )])
        else:
            # Use a nice default colorscale for mesh
            fig = go.Figure(data=[go.Mesh3d(
                x=df_clean[x_col], y=df_clean[y_col], z=df_clean[z_col],
                colorscale='Turbo',
                opacity=0.7,
                showscale=True
            )])
        
        fig.update_layout(
            title=f"3D Mesh: {x_col}, {y_col}, {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor="rgba(0,0,0,0)"
            ),
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return fig
    except Exception as e:
        st.error(f"Could not create 3D mesh plot: {e}")
        st.info("Mesh plots work best with coordinate data that forms connected surfaces. Try using datasets with spatial relationships between points.")
        return None
