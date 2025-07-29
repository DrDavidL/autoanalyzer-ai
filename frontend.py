import streamlit as st
import requests
import pandas as pd
import json
import time
import os

# Configure Streamlit page
st.set_page_config(
    page_title="AutoAnalyzer AI",
    layout="centered",
    page_icon="📊",
    initial_sidebar_state="auto",
)

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "current_data" not in st.session_state:
    st.session_state.current_data = None
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

def create_session():
    """Create a new session with the backend"""
    try:
        response = requests.post(f"{BACKEND_URL}/api/data/sessions")
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data["session_id"]
            return data["session_id"]
        else:
            st.error(f"Failed to create session: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return None

def load_demo_dataset(dataset_id):
    """Load a demo dataset"""
    if not st.session_state.session_id:
        if not create_session():
            return None
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/data/demo-datasets/{dataset_id}/load",
            params={"session_id": st.session_state.session_id}
        )
        if response.status_code == 200:
            data = response.json()
            st.success(f"Loaded {dataset_id}: {data['rows']} rows, {data['columns']} columns")
            return data
        else:
            error_data = response.json()
            st.error(f"Failed to load demo dataset: {error_data.get('detail', {}).get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def get_session_data():
    """Get data from current session"""
    if not st.session_state.session_id:
        return None
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/data/sessions/{st.session_state.session_id}/data")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error getting session data: {e}")
        return None

def get_data_summary():
    """Get data summary from current session"""
    if not st.session_state.session_id:
        return None
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/data/sessions/{st.session_state.session_id}/summary")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error getting data summary: {e}")
        return None

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 3rem !important;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .highlight-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown("<h1 class='main-title'>📊 AutoAnalyzer AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Interactive data exploration and machine learning for healthcare data</p>", unsafe_allow_html=True)

# Create session if not exists
if not st.session_state.session_id:
    with st.spinner("Creating new session..."):
        create_session()
    if st.session_state.session_id:
        st.success(f"Created new session: {st.session_state.session_id[:8]}...")

# Sidebar for session info and steps
if st.session_state.session_id:
    st.sidebar.success(f"Session: {st.session_state.session_id[:8]}...")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🧠 Machine Learning", "🤖 Analyze with GPT"])

with tab1:
    # Sidebar content for Data Exploration
    st.sidebar.markdown("<div class='step-header'>Step 1: Upload your data or view a demo dataset</div>", unsafe_allow_html=True)
    
    demo_or_custom = st.sidebar.selectbox(
        "Upload a CSV or Excel file. NO PHI - use only anonymized data",
        (
            "🩸 Demo 1 (diabetes)",
            "🔬 Demo 2 (breast cancer)",
            "🧠 Demo 3 (stroke)",
            "📁 CSV or Excel Upload",
        ),
        index=0,
    )
    
    # Dataset mapping to correct backend IDs
    dataset_mapping = {
        "🩸 Demo 1 (diabetes)": "diabetes",
        "🔬 Demo 2 (breast cancer)": "breast_cancer",  # Correct backend ID
        "🧠 Demo 3 (stroke)": "stroke"
    }
    
    # Handle dataset loading
    if demo_or_custom == "📁 CSV or Excel Upload":
        st.info("Please use the file uploader that appeared in the sidebar on the left.")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file:
            st.info("File upload functionality will be implemented with the backend")
    
    elif demo_or_custom in dataset_mapping:
        dataset_id = dataset_mapping[demo_or_custom]
        if st.sidebar.button("Load Dataset"):
            result = load_demo_dataset(dataset_id)
            if result:
                st.session_state.current_data = result
    
    # Show data exploration if data is loaded
    if st.session_state.current_data:
        st.markdown("<div class='step-header'>Step 2: Explore Your Data</div>", unsafe_allow_html=True)
        
        # Get and display data summary
        summary = get_data_summary()
        if summary:
            # Basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", summary.get("shape", {}).get("rows", "N/A"))
            with col2:
                st.metric("Columns", summary.get("shape", {}).get("columns", "N/A"))
            with col3:
                st.metric("Missing Values", summary.get("missing_total", "N/A"))
            
            # Data preview
            session_data = get_session_data()
            if session_data and session_data.get("data"):
                st.subheader("Data Preview")
                df = pd.DataFrame(session_data["data"])
                st.dataframe(df.head())
                
                # Store dataframe in session state for compatibility
                st.session_state.df = df
                
                # Column info
                if "dtypes" in session_data:
                    st.subheader("Column Information")
                    col_info = pd.DataFrame([
                        {"Column": col, "Type": dtype} 
                        for col, dtype in session_data["dtypes"].items()
                    ])
                    st.dataframe(col_info)
        
        # Sidebar tools
        st.sidebar.markdown("<div class='step-header'>Step 2: Assess Data Readiness</div>", unsafe_allow_html=True)
        
        check_preprocess = st.sidebar.checkbox("🔍 Assess dataset readiness")
        needs_preprocess = st.sidebar.checkbox("🛠️ Select if dataset fails readiness")
        
        st.sidebar.markdown("<div class='step-header'>Step 3: Tools for Analysis</div>", unsafe_allow_html=True)
        
        # Basic analysis tools in sidebar
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            header = st.checkbox("📋 Show header")
            summary_stats = st.checkbox("📊 Summary stats")
            show_corr = st.checkbox("🔥 Correlation heatmap")
            
        with col2:
            histogram = st.checkbox("📊 Histogram")
            scatter = st.checkbox("📈 Scatterplot")
            box_plot = st.checkbox("📦 Box plot")
        
        # Display analysis results
        if header and not st.session_state.df.empty:
            st.subheader("First 5 Rows of Data")
            st.dataframe(st.session_state.df.head())
        
        if summary_stats and not st.session_state.df.empty:
            st.subheader("Summary Statistics")
            st.dataframe(st.session_state.df.describe())
        
        if show_corr and not st.session_state.df.empty:
            st.subheader("Correlation Heatmap")
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = st.session_state.df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis")
        
        if histogram and not st.session_state.df.empty:
            st.subheader("Histogram")
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Choose a column for histogram:", numeric_cols)
                if selected_col:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    st.session_state.df[selected_col].hist(bins=30, ax=ax)
                    ax.set_title(f"Distribution of {selected_col}")
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
        
        if scatter and not st.session_state.df.empty:
            st.subheader("Scatterplot")
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox("Y-axis:", numeric_cols, index=1, key="scatter_y")
                
                if x_col and y_col:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=st.session_state.df, x=x_col, y=y_col, ax=ax)
                    ax.set_title(f"{y_col} vs {x_col}")
                    st.pyplot(fig)
        
        if box_plot and not st.session_state.df.empty:
            st.subheader("Box Plot")
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols and categorical_cols:
                col1, col2 = st.columns(2)
                with col1:
                    num_col = st.selectbox("Numeric column:", numeric_cols, key="box_num")
                with col2:
                    cat_col = st.selectbox("Grouping column:", categorical_cols, key="box_cat")
                
                if num_col and cat_col:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    fig, ax = plt.subplots()
                    sns.boxplot(data=st.session_state.df, x=cat_col, y=num_col, ax=ax)
                    ax.set_title(f"{num_col} by {cat_col}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
    else:
        st.info("Please select a demo dataset or upload a file to begin analysis.")

with tab2:
    st.header("🧠 Machine Learning")
    
    if not st.session_state.current_data:
        st.warning("Please load a dataset first in the Data Exploration tab")
    else:
        st.info("Machine learning functionality will connect to the backend ML endpoints")
        
        # Get session data for ML
        session_data = get_session_data()
        if session_data and session_data.get("data"):
            df = pd.DataFrame(session_data["data"])
            
            # Basic ML interface
            st.subheader("Model Configuration")
            
            # Target selection
            all_columns = df.columns.tolist()
            target_col = st.selectbox("Select target column:", all_columns)
            
            # Feature selection
            feature_cols = st.multiselect(
                "Select feature columns:",
                [col for col in all_columns if col != target_col],
                default=[col for col in all_columns if col != target_col][:5]
            )
            
            # Model type
            model_type = st.selectbox(
                "Select model type:",
                ["Logistic Regression", "Random Forest", "XGBoost"]
            )
            
            if st.button("Train Model"):
                st.info("Model training will be implemented with backend ML endpoints")

with tab3:
    st.header("🤖 Analyze with GPT")
    
    if not st.session_state.current_data:
        st.warning("Please load a dataset first in the Data Exploration tab")
    else:
        st.markdown("""
        <div class="highlight-box">
            <h4>AI-Powered Data Analysis</h4>
            <p>Ask any question about your data in plain English. The AI will analyze your data and provide insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Question input
        question = st.text_area(
            "Ask a question about your data:",
            placeholder="Example: What are the key factors that predict diabetes in this dataset?",
            height=100
        )
        
        if st.button("🚀 Analyze My Data"):
            if not question.strip():
                st.error("Please enter a question")
            else:
                st.info("GPT analysis will be implemented with backend GPT endpoints")
