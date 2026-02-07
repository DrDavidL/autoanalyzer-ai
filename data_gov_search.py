"""
Data.gov search functionality for AutoAnalyzer
"""

import pandas as pd
import requests
import streamlit as st
from typing import List, Dict, Optional, Tuple
import io


def search_data_gov(search_term: str, max_results: int = 100) -> Optional[List[Dict]]:
    """
    Search data.gov for CSV datasets
    
    Args:
        search_term: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of dataset dictionaries or None if error
    """
    base_url = "https://catalog.data.gov/api/3/action/package_search"
    
    query_params = {
        "q": search_term,
        "fq": "res_format:CSV",
        "rows": min(max_results, 1000),  # API limit is 1000
    }
    
    try:
        response = requests.get(base_url, params=query_params, timeout=30)
        response.raise_for_status()
        
        datasets_json = response.json()
        
        if not datasets_json.get('success', False):
            st.error("API request was not successful")
            return None
            
        results = datasets_json.get('result', {}).get('results', [])
        
        data = []
        for dataset in results:
            title = dataset.get('title', 'No Title')
            notes = dataset.get('notes', 'No Description')
            organization = dataset.get('organization', {}).get('title', 'Unknown')
            
            # Find CSV resources
            for resource in dataset.get('resources', []):
                if resource.get('format', '').upper() == 'CSV':
                    url = resource.get('url')
                    if url:
                        data.append({
                            'Title': title,
                            'Description': notes,
                            'Organization': organization,
                            'Download Link': url,
                            'Resource ID': resource.get('id', ''),
                            'Last Modified': resource.get('last_modified', 'Unknown')
                        })
                        break  # Only take the first CSV resource per dataset
        
        return data
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error searching data.gov: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


def download_csv_from_url(url: str) -> Optional[pd.DataFrame]:
    """
    Download and parse CSV from URL
    
    Args:
        url: URL to CSV file
        
    Returns:
        DataFrame or None if error
    """
    try:
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        # Try to detect encoding
        encoding = response.encoding if response.encoding else 'utf-8'
        
        # Read CSV into DataFrame
        csv_content = io.StringIO(response.content.decode(encoding, errors='replace'))
        df = pd.read_csv(csv_content)
        
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading file: {str(e)}")
        return None
    except pd.errors.EmptyDataError:
        st.error("The downloaded file is empty or not a valid CSV.")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error downloading file: {str(e)}")
        return None


def display_search_results(datasets: List[Dict]) -> Optional[str]:
    """
    Display search results and allow user to select one
    
    Args:
        datasets: List of dataset dictionaries
        
    Returns:
        Selected download URL or None
    """
    if not datasets:
        st.warning("No datasets found.")
        return None
    
    st.success(f"Found {len(datasets)} datasets with CSV files.")
    
    # Create a DataFrame for display
    display_df = pd.DataFrame(datasets)
    
    # Truncate long descriptions for better display
    display_df['Description'] = display_df['Description'].apply(
        lambda x: (x[:200] + '...') if len(str(x)) > 200 else x
    )
    
    # Display the results table
    st.dataframe(
        display_df[['Title', 'Organization', 'Description', 'Last Modified']], 
        height=400,
        use_container_width=True
    )
    
    # Let user select a dataset
    dataset_options = [f"{i+1}. {row['Title'][:80]}..." if len(row['Title']) > 80 
                      else f"{i+1}. {row['Title']}" 
                      for i, row in enumerate(datasets)]
    
    selected_idx = st.selectbox(
        "Select a dataset to download:",
        range(len(dataset_options)),
        format_func=lambda x: dataset_options[x],
        key="data_gov_selection"
    )
    
    if st.button("Download Selected Dataset", key="download_data_gov"):
        selected_dataset = datasets[selected_idx]
        st.info(f"Downloading: {selected_dataset['Title']}")
        
        with st.spinner("Downloading dataset..."):
            df = download_csv_from_url(selected_dataset['Download Link'])
            
        if df is not None:
            st.success(f"Successfully downloaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
            
            # Show preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show basic info
            st.subheader("Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Show column info
            with st.expander("Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(col_info, use_container_width=True)
            
            return df
    
    return None


def data_gov_search_interface():
    """
    Main interface for data.gov search functionality
    
    Returns:
        DataFrame if dataset selected and downloaded, None otherwise
    """
    st.markdown("### 🏛️ Search Data.gov for Datasets")
    st.markdown("""
    Search the U.S. government's open data catalog for CSV datasets. 
    You can search by topic, agency, or keywords.
    """)
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input(
            "Enter search terms:",
            placeholder="e.g., heart disease, COVID-19, census, climate",
            help="Use keywords related to your research topic"
        )
    
    with col2:
        max_results = st.number_input(
            "Max results:",
            min_value=10,
            max_value=100,
            value=20,
            step=10
        )
    
    # Example searches
    st.markdown("**Example searches:**")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("🫀 Heart Disease", key="example_heart"):
            search_term = "heart disease OR cardiovascular"
    
    with example_col2:
        if st.button("🦠 COVID-19", key="example_covid"):
            search_term = "COVID-19 OR coronavirus"
    
    with example_col3:
        if st.button("🌡️ Climate", key="example_climate"):
            search_term = "climate change OR temperature"
    
    # Search button
    if st.button("🔍 Search Data.gov", disabled=not search_term.strip()):
        if search_term.strip():
            with st.spinner(f"Searching data.gov for '{search_term}'..."):
                datasets = search_data_gov(search_term.strip(), max_results)
            
            if datasets:
                # Store results in session state
                st.session_state.data_gov_results = datasets
                st.session_state.data_gov_search_term = search_term
    
    # Display results if available
    if hasattr(st.session_state, 'data_gov_results') and st.session_state.data_gov_results:
        st.markdown(f"### Search Results for: '{st.session_state.data_gov_search_term}'")
        df = display_search_results(st.session_state.data_gov_results)
        return df
    
    return None
