"""
Data.gov API integration for AutoAnalyzer
Provides functionality to search, browse, and download CSV datasets from Data.gov
"""

import requests
import pandas as pd
import streamlit as st
import os
from typing import List, Dict, Optional


class DataGovAPI:
    """Interface for Data.gov CKAN API"""
    
    def __init__(self):
        self.base_url = "https://catalog.data.gov/api/3/action"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AutoAnalyzer/1.0 (Educational Tool)'
        })
    
    def search_datasets(self, query: str, limit: int = 20, offset: int = 0) -> Dict:
        """
        Search for datasets on Data.gov
        
        Args:
            query: Search term
            limit: Number of results to return (max 100)
            offset: Number of results to skip
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Focus on CSV datasets only
            search_query = f"{query} AND res_format:CSV"
            
            params = {
                'q': search_query,
                'rows': min(limit, 100),  # API limit
                'start': offset,
                'sort': 'score desc, metadata_modified desc'
            }
            
            response = self.session.get(
                f"{self.base_url}/package_search",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                return data.get('result', {})
            else:
                st.error(f"API Error: {data.get('error', 'Unknown error')}")
                return {}
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error searching Data.gov: {str(e)}")
            return {}
        except Exception as e:
            st.error(f"Error searching Data.gov: {str(e)}")
            return {}
    
    def get_dataset_details(self, dataset_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific dataset
        
        Args:
            dataset_id: The dataset identifier
            
        Returns:
            Dictionary containing dataset details or None if error
        """
        try:
            params = {'id': dataset_id}
            response = self.session.get(
                f"{self.base_url}/package_show",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                return data.get('result', {})
            else:
                st.error(f"API Error: {data.get('error', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error getting dataset details: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error getting dataset details: {str(e)}")
            return None
    
    def get_csv_resources(self, dataset: Dict) -> List[Dict]:
        """
        Extract CSV resources from a dataset
        
        Args:
            dataset: Dataset dictionary from API
            
        Returns:
            List of CSV resource dictionaries
        """
        csv_resources = []
        resources = dataset.get('resources', [])
        
        for resource in resources:
            format_type = resource.get('format', '').upper()
            mimetype = resource.get('mimetype', '').lower()
            
            # Check if it's a CSV file
            if (format_type == 'CSV' or 
                'csv' in mimetype or 
                resource.get('url', '').lower().endswith('.csv')):
                csv_resources.append(resource)
        
        return csv_resources
    
    def download_csv(self, url: str, max_size_mb: int = 50) -> Optional[pd.DataFrame]:
        """
        Download and parse a CSV file from Data.gov
        
        Args:
            url: URL of the CSV file
            max_size_mb: Maximum file size to download in MB
            
        Returns:
            pandas DataFrame or None if error
        """
        try:
            # First, check the file size
            head_response = self.session.head(url, timeout=10)
            if head_response.status_code == 200:
                content_length = head_response.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > max_size_mb:
                        st.error(f"File too large ({size_mb:.1f} MB). Maximum allowed: {max_size_mb} MB")
                        return None
            
            # Download the file
            with st.spinner("Downloading CSV file..."):
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Try to parse as CSV
                try:
                    # Use StringIO to handle the content
                    from io import StringIO
                    csv_content = StringIO(response.text)
                    df = pd.read_csv(csv_content)
                    
                    # Basic validation
                    if df.empty:
                        st.warning("Downloaded CSV file is empty")
                        return None
                    
                    # Limit rows for performance (can be adjusted)
                    if len(df) > 10000:
                        st.info(f"Large dataset detected ({len(df)} rows). Loading first 10,000 rows for analysis.")
                        df = df.head(10000)
                    
                    return df
                    
                except pd.errors.EmptyDataError:
                    st.error("The CSV file is empty or invalid")
                    return None
                except pd.errors.ParserError as e:
                    st.error(f"Error parsing CSV file: {str(e)}")
                    return None
                    
        except requests.exceptions.RequestException as e:
            st.error(f"Network error downloading file: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error downloading CSV: {str(e)}")
            return None
    
    def save_csv_to_desktop(self, url: str, filename: str) -> bool:
        """
        Download CSV file and save to user's desktop
        
        Args:
            url: URL of the CSV file
            filename: Name for the saved file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get user's desktop path
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            if not os.path.exists(desktop_path):
                # Fallback to home directory if Desktop doesn't exist
                desktop_path = os.path.expanduser("~")
            
            # Ensure filename has .csv extension
            if not filename.lower().endswith('.csv'):
                filename += '.csv'
            
            # Make filename safe
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            file_path = os.path.join(desktop_path, safe_filename)
            
            # Download the file
            with st.spinner(f"Downloading {safe_filename} to desktop..."):
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Save to desktop
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                st.success(f"File saved to: {file_path}")
                return True
                
        except Exception as e:
            st.error(f"Error saving file to desktop: {str(e)}")
            return False


def _load_page_results():
    """Load results for the current page"""
    if hasattr(st.session_state, 'datagov_last_query'):
        current_page = st.session_state.datagov_current_page
        results_per_page = 15
        offset = current_page * results_per_page
        
        with st.spinner("Loading page..."):
            results = st.session_state.datagov_api.search_datasets(
                st.session_state.datagov_last_query,
                limit=results_per_page,
                offset=offset
            )
            
            if results and results.get('results'):
                st.session_state.datagov_search_results = results.get('results', [])


def render_datagov_interface():
    """
    Render the Data.gov CSV file selection interface
    """
    st.markdown("### 🏛️ Data.gov CSV Files")
    st.info("Search and analyze CSV datasets from the U.S. government's open data portal")
    
    # Initialize API client
    if 'datagov_api' not in st.session_state:
        st.session_state.datagov_api = DataGovAPI()
    
    # Initialize session state for search results
    if 'datagov_search_results' not in st.session_state:
        st.session_state.datagov_search_results = []
    if 'datagov_selected_dataset' not in st.session_state:
        st.session_state.datagov_selected_dataset = None
    if 'datagov_current_page' not in st.session_state:
        st.session_state.datagov_current_page = 0
    if 'datagov_total_results' not in st.session_state:
        st.session_state.datagov_total_results = 0
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for datasets:",
            placeholder="e.g., health, education, climate, crime, demographics",
            help="Enter keywords to search for CSV datasets on Data.gov"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Perform search
    if search_button and search_query.strip():
        # Reset pagination when performing new search
        st.session_state.datagov_current_page = 0
        st.session_state.datagov_last_query = search_query.strip()
        
        with st.spinner("Searching Data.gov..."):
            results = st.session_state.datagov_api.search_datasets(
                search_query.strip(), 
                limit=15,  # Show 15 results per page
                offset=0
            )
            
            if results and results.get('count', 0) > 0:
                st.session_state.datagov_search_results = results.get('results', [])
                st.session_state.datagov_total_results = results.get('count', 0)
                st.success(f"Found {results.get('count', 0)} datasets")
            else:
                st.warning("No datasets found. Try different search terms.")
                st.session_state.datagov_search_results = []
                st.session_state.datagov_total_results = 0
    
    # Display search results
    if st.session_state.datagov_search_results:
        # Show pagination info
        total_results = st.session_state.datagov_total_results
        current_page = st.session_state.datagov_current_page
        results_per_page = 15
        total_pages = (total_results + results_per_page - 1) // results_per_page
        
        st.markdown(f"### Search Results (Page {current_page + 1} of {total_pages})")
        st.caption(f"Showing results {current_page * results_per_page + 1}-{min((current_page + 1) * results_per_page, total_results)} of {total_results}")
        
        # Show link to selected dataset if one is selected
        if st.session_state.datagov_selected_dataset:
            st.info("✅ Dataset selected! Scroll down for next step options.", icon="🎯")

        # Create a scrollable list of datasets
        for i, dataset in enumerate(st.session_state.datagov_search_results):
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Dataset title and description
                    title = dataset.get('title', 'Untitled Dataset')
                    notes = dataset.get('notes', 'No description available')
                    
                    # Truncate long descriptions
                    if len(notes) > 200:
                        notes = notes[:200] + "..."
                    
                    # Organization info
                    org_title = dataset.get('organization', {}).get('title', 'Unknown Organization')
                    
                    # Get CSV resources count
                    csv_resources = st.session_state.datagov_api.get_csv_resources(dataset)
                    csv_count = len(csv_resources)
                    
                    st.markdown(f"**{title}**")
                    st.markdown(f"*{org_title}* • {csv_count} CSV file(s)")
                    st.markdown(f"{notes}")
                
                with col2:
                    if st.button("Select", key=f"select_{i}", use_container_width=True):
                        st.session_state.datagov_selected_dataset = dataset
                        # Auto-scroll to download options after selection
                        st.markdown("""
                        <script>
                        setTimeout(function() {
                            const element = document.querySelector('h2[data-testid="stHeader"]:has-text("📋 Dataset Details & Download Options")');
                            if (element) {
                                element.scrollIntoView({ behavior: 'smooth' });
                            } else {
                                // Fallback: scroll to bottom
                                window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                            }
                        }, 100);
                        </script>
                        """, unsafe_allow_html=True)
                
                st.divider()
        
        # Pagination controls
        if total_pages > 1:
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⏮️ First", disabled=(current_page == 0)):
                    st.session_state.datagov_current_page = 0
                    _load_page_results()
                    st.rerun()
            
            with col2:
                if st.button("◀️ Previous", disabled=(current_page == 0)):
                    st.session_state.datagov_current_page = current_page - 1
                    _load_page_results()
                    st.rerun()
            
            with col3:
                st.markdown(f"<div style='text-align: center; padding: 8px;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            
            with col4:
                if st.button("▶️ Next", disabled=(current_page >= total_pages - 1)):
                    st.session_state.datagov_current_page = current_page + 1
                    _load_page_results()
                    st.rerun()
            
            with col5:
                if st.button("⏭️ Last", disabled=(current_page >= total_pages - 1)):
                    st.session_state.datagov_current_page = total_pages - 1
                    _load_page_results()
                    st.rerun()
    
    # Display selected dataset details
    if st.session_state.datagov_selected_dataset:
        dataset = st.session_state.datagov_selected_dataset
        
        # Create a very prominent visual separator
        st.markdown("---")
        st.header("Selected Dataset")
        
        # Show a prominent notice that the dataset was selected
        st.success("✅ Dataset selected! Download options are available below.")
        
        # Create anchor point for navigation
        st.header("📋 Dataset Details & Download Options")
        
        # Dataset information
        title = dataset.get('title', 'Untitled Dataset')
        notes = dataset.get('notes', 'No description available')
        org_title = dataset.get('organization', {}).get('title', 'Unknown Organization')
        
        st.markdown(f"**{title}**")
        st.markdown(f"*Organization:* {org_title}")
        
        with st.expander("Dataset Description", expanded=True):
            st.markdown(notes)
        
        # Get CSV resources
        csv_resources = st.session_state.datagov_api.get_csv_resources(dataset)
        
        if csv_resources:
            st.markdown("### Available CSV Files")
            
            for i, resource in enumerate(csv_resources):
                resource_name = resource.get('name', f'CSV File {i+1}')
                resource_description = resource.get('description', 'No description')
                resource_url = resource.get('url', '')
                
                # File size info if available
                size_info = ""
                if resource.get('size'):
                    try:
                        size_bytes = int(resource['size'])
                        size_mb = size_bytes / (1024 * 1024)
                        size_info = f" ({size_mb:.1f} MB)"
                    except Exception:
                        pass
                
                with st.container():
                    st.markdown(f"**{resource_name}**{size_info}")
                    if resource_description and resource_description != 'No description':
                        st.markdown(f"*{resource_description}*")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(
                            "📊 Load into Analyzer", 
                            key=f"load_{i}",
                            help="Download and load this CSV into the analyzer for analysis"
                        ):
                            df = st.session_state.datagov_api.download_csv(resource_url)
                            if df is not None:
                                st.session_state.df = df
                                st.session_state.gpt_working_df = df.copy()
                                st.success(f"✅ Dataset loaded! ({len(df)} rows, {len(df.columns)} columns)")
                                st.info("You can now use this dataset in the Data Exploration and Machine Learning tabs.")
                                
                                # Show preview
                                with st.expander("Preview of loaded data"):
                                    st.dataframe(df.head())
                    
                    with col2:
                        if st.button(
                            "💾 Download to Desktop", 
                            key=f"download_{i}",
                            help="Download this CSV file directly to your desktop"
                        ):
                            filename = f"{title}_{resource_name}".replace(' ', '_')
                            st.session_state.datagov_api.save_csv_to_desktop(resource_url, filename)
                    
                    with col3:
                        if st.button(
                            "🔗 View Source", 
                            key=f"view_{i}",
                            help="Open the original data source in a new tab"
                        ):
                            st.markdown(f"[Open in new tab]({resource_url})")
                    
                    st.divider()
        else:
            st.warning("No CSV files found in this dataset.")
        
        # Back button
        if st.button("← Back to Search Results"):
            st.session_state.datagov_selected_dataset = None
            st.rerun()
    
    # Help section
    with st.expander("ℹ️ About Data.gov Integration"):
        st.markdown("""
        **Data.gov** is the U.S. government's open data portal, providing access to thousands of datasets from federal agencies.
        
        **Features:**
        - Search for CSV datasets by keyword
        - Preview dataset descriptions and metadata
        - Load datasets directly into AutoAnalyzer for analysis
        - Download files to your desktop for offline use
        
        **Tips for better searches:**
        - Use specific keywords (e.g., "health outcomes", "education statistics")
        - Try different terms if you don't find what you're looking for
        - Government datasets often use formal terminology
        
        **Note:** Large datasets (>10,000 rows) are automatically truncated for performance.
        """)


# Utility functions for integration with main app
def is_datagov_available() -> bool:
    """Check if Data.gov API is accessible"""
    try:
        response = requests.get("https://catalog.data.gov/api/3/action/status_show", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_popular_datasets() -> List[Dict]:
    """Get a list of popular/featured datasets for quick access"""
    # This could be expanded to fetch actual popular datasets
    return [
        {
            'title': 'COVID-19 Data',
            'query': 'covid coronavirus pandemic',
            'description': 'Datasets related to COVID-19 pandemic'
        },
        {
            'title': 'Climate Data',
            'query': 'climate weather temperature',
            'description': 'Weather and climate-related datasets'
        },
        {
            'title': 'Health Statistics',
            'query': 'health medical disease',
            'description': 'Public health and medical datasets'
        },
        {
            'title': 'Education Data',
            'query': 'education school student',
            'description': 'Educational statistics and school data'
        },
        {
            'title': 'Economic Data',
            'query': 'economic employment income',
            'description': 'Economic indicators and employment data'
        }
    ]
