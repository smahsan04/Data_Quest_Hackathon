import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import traceback
import base64
from io import StringIO
try:
    from data_utils import (
    clean_data, 
    infer_schema,
    merge_dataframes,
    generate_data_insights,
    recommend_visualizations,
    generate_visualization,
    summarize_dataset,
    generate_advanced_insights,
    suggest_next_steps,
    query_together_ai,
    generate_pdf_report,
    json_serializable  # Make sure this is imported
)
except ImportError as e:
    st.error(f"Error importing data_utils: {str(e)}")
    st.error("Please make sure all required packages are installed and data_utils.py is in the same directory.")
    st.stop()

# Set up page configuration with enhanced styling
st.set_page_config(
    page_title="AI Data Explorer", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
    }
    .insight-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 10px;
    }
    .highlight {
        background-color: #e8f4f8;
        border-left: 5px solid #1E88E5;
        padding: 10px;
        margin: 10px 0;
    }

    /* Progress animation */
    @keyframes progress {
        from {width: 0%;}
        to {width: 100%;}
    }
    .progress-bar {
        height: 4px;
        background-color: #1E88E5;
        width: 100%;
        animation: progress 8s ease-in-out;
    }
    
    /* Enhanced Cards */
    .stat-card {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 15px;
        margin-bottom: 15px;
        border-top: 4px solid #1E88E5;
        transition: transform 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-5px);
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .card-label {
        font-size: 0.9rem;
        color: #555;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        border-bottom: 2px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "datasets" not in st.session_state:
    st.session_state.datasets = {}  # Store multiple datasets
if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = None
if "insights" not in st.session_state:
    st.session_state.insights = {}
if "cleaning_reports" not in st.session_state:
    st.session_state.cleaning_reports = {}
if "merge_report" not in st.session_state:
    st.session_state.merge_report = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = None
if "visualization_recs" not in st.session_state:
    st.session_state.visualization_recs = None
if "generated_visualizations" not in st.session_state:
    st.session_state.generated_visualizations = {}
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "upload"
if "processing_time" not in st.session_state:
    st.session_state.processing_time = None

# Helper function to convert non-serializable types to serializable types
def json_serializable(obj):
    """Convert a dict with non-serializable values to serializable ones."""
    if isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return json_serializable(obj.tolist())
    elif pd.isna(obj):
        return None
    else:
        return obj

# --- Helper Functions ---
def reset_analysis_state():
    """Reset analysis-related state when new data is loaded."""
    st.session_state.insights = {}
    st.session_state.cleaning_reports = {}
    st.session_state.merge_report = None
    st.session_state.dataset_summary = None
    st.session_state.visualization_recs = None
    st.session_state.generated_visualizations = {}
    st.session_state.chat_history = []
    st.session_state.processing_time = None

def analyze_dataset(df):
    """Run all analysis functions on the dataset."""
    try:
        with st.spinner('AI is analyzing your data... This may take a moment.'):
            # Start timing the process
            start_time = time.time()
            
            # Show progress animation
            st.markdown("<div class='progress-bar'></div>", unsafe_allow_html=True)
            
            # Clean data
            df_cleaned, cleaning_report = clean_data(df)
            st.session_state.datasets['cleaned'] = df_cleaned
            st.session_state.cleaning_reports['main'] = cleaning_report
            
            # Generate insights
            insights = generate_data_insights(df_cleaned)
            # Convert to JSON serializable format
            st.session_state.insights = json_serializable(insights)
            
            # Generate viz recommendations
            viz_recs = recommend_visualizations(df_cleaned)
            st.session_state.visualization_recs = viz_recs
            
            # Generate key visualizations in advance
            if viz_recs and len(viz_recs) > 0:
                for viz in viz_recs[:min(2, len(viz_recs))]:  # Generate first 2 recommended visualizations
                    try:
                        viz_result = generate_visualization(
                            df_cleaned, 
                            viz["type"],
                            viz["columns"],
                            viz["parameters"]
                        )
                        if viz_result["success"]:
                            st.session_state.generated_visualizations[viz["id"]] = viz_result
                    except Exception as e:
                        st.warning(f"Could not generate visualization: {str(e)}")
            
            # Generate summary using LLM
            try:
                summary = summarize_dataset(df_cleaned)
                st.session_state.dataset_summary = summary
            except Exception as e:
                st.warning(f"Could not generate dataset summary: {str(e)}")
                st.session_state.dataset_summary = "Could not generate automatic summary."
            
            # Set current dataset as active
            st.session_state.active_dataset = 'cleaned'
            
            # Record processing time
            st.session_state.processing_time = time.time() - start_time
            
            # Switch to insights tab
            st.session_state.current_tab = "insights"
            
            st.success("Analysis complete! Explore the insights tab to see results.")
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.error(traceback.format_exc())

# --- Main App Layout ---
def render_header():
    """Render the application header."""
    st.markdown("<h1 class='main-header'>📊 AI Data Explorer</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align: center; font-size: 1.2rem;'>
        Upload your data and let AI discover insights automatically - no coding required!
        </p>
        """, 
        unsafe_allow_html=True
    )

def render_sidebar():
    """Render the sidebar with navigation and dataset info."""
    with st.sidebar:
        st.title("Navigation")
        
        # Create navigation tabs
        tabs = {
            "upload": "📤 Upload Data",
            "insights": "🔍 Insights",
            "visualizations": "📊 Visualizations",
            "chat": "💬 Ask Questions",
            "report": "📄 Generate Report"
        }
        
        # Disable tabs if no data is loaded
        disabled_tabs = []
        if not st.session_state.datasets:
            disabled_tabs = ["insights", "visualizations", "chat", "report"]
        
        # Show navigation options
        for tab_id, tab_name in tabs.items():
            button_type = "primary" if st.session_state.current_tab == tab_id else "secondary"
            disabled = tab_id in disabled_tabs
            if st.button(tab_name, type=button_type, disabled=disabled, key=f"nav_{tab_id}"):
                st.session_state.current_tab = tab_id
                st.rerun()
        
        # Show dataset info if available
        if st.session_state.datasets:
            st.divider()
            st.subheader("Dataset Information")
            
            active_df = st.session_state.datasets.get(st.session_state.active_dataset)
            if active_df is not None:
                st.write(f"Rows: {len(active_df):,}")
                st.write(f"Columns: {len(active_df.columns):,}")
                
                # Show column types summary
                dtype_counts = active_df.dtypes.value_counts().to_dict()
                st.write("Column Types:")
                for dtype, count in dtype_counts.items():
                    st.write(f"- {dtype}: {count}")
                
                # Show processing time if available
                if st.session_state.processing_time:
                    st.write(f"Analysis Time: {st.session_state.processing_time:.2f} seconds")
            
            # Dataset selector if multiple datasets
            if len(st.session_state.datasets) > 1:
                st.divider()
                st.subheader("Switch Dataset")
                
                # Create a radio button to switch between datasets
                selected_dataset = st.radio(
                    "Select active dataset:",
                    options=list(st.session_state.datasets.keys()),
                    key="selected_dataset_radio"
                )
                
                if selected_dataset != st.session_state.active_dataset:
                    st.session_state.active_dataset = selected_dataset
                    st.rerun()

def render_upload_page():
    """Render the data upload page."""
    st.markdown("<h2 class='sub-header'>Upload Your Data</h2>", unsafe_allow_html=True)
    
    # Quick intro/benefits
    st.markdown("""
    <div class="highlight">
    <strong>🚀 Let AI do the heavy lifting!</strong>
    <ul>
        <li>Automatically clean and preprocess messy data</li>
        <li>Generate insights and visualizations in seconds</li>
        <li>Ask questions in plain English</li>
        <li>Get a comprehensive PDF report with one click</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    upload_method = st.radio(
        "Choose upload method:",
        options=["Upload CSV/Excel Files", "Use Sample Dataset"],
        horizontal=True,
        key="upload_method_selection_page1"
    )

    
    if upload_method == "Upload CSV/Excel Files":
        uploaded_files = st.file_uploader(
            "Upload one or more CSV or Excel files", 
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Drag and drop one or more files here. Multiple files can be merged if they share common columns."
        )
        
        if uploaded_files:
            # Process files and store individual datasets
            all_dfs = []
            with st.spinner("Processing uploaded files..."):
                for file in uploaded_files:
                    try:
                        # Try different parsers based on file type
                        if file.name.endswith('.csv'):
                            # First try with comma delimiter
                            try:
                                df = pd.read_csv(file)
                            except:
                                # Try with other common delimiters
                                try:
                                    df = pd.read_csv(file, sep=';')
                                except:
                                    df = pd.read_csv(file, sep='\t')
                        else:  # Excel files
                            df = pd.read_excel(file)
                            
                        file_key = os.path.splitext(file.name)[0]
                        st.session_state.datasets[file_key] = df
                        all_dfs.append(df)
                        
                        st.success(f"Successfully loaded '{file.name}' with {len(df):,} rows and {len(df.columns):,} columns.")
                    except Exception as e:
                        st.error(f"Error loading '{file.name}': {str(e)}")
            
            if len(all_dfs) > 0:
                # Option to proceed with analysis
                proceed = st.button("Process Data with AI", type="primary")
                
                # Show merge option if multiple datasets
                if len(all_dfs) > 1:
                    merge_option = st.checkbox("Merge datasets", value=True)
                    
                    if merge_option:
                        with st.expander("Merge Settings", expanded=True):
                            # Try to identify common columns
                            common_cols = set.intersection(*[set(df.columns) for df in all_dfs])
                            merge_keys = st.multiselect(
                                "Select columns to merge on:",
                                options=list(common_cols),
                                default=list(common_cols)[:1] if common_cols else None
                            )
                            
                            if proceed and merge_keys:
                                # Merge datasets
                                with st.spinner("Merging datasets..."):
                                    try:
                                        merged_df, merge_report = merge_dataframes(all_dfs, merge_keys)
                                        st.session_state.datasets['merged'] = merged_df
                                        st.session_state.merge_report = merge_report
                                        
                                        if merge_report["status"] == "success":
                                            st.success(f"Successfully merged {len(all_dfs)} datasets: {merge_report['message']}")
                                            reset_analysis_state()
                                            analyze_dataset(merged_df)
                                        else:
                                            st.error(f"Error merging datasets: {merge_report['message']}")
                                    except Exception as e:
                                        st.error(f"Error during merge: {str(e)}")
                            elif proceed:
                                # Use first dataset only
                                reset_analysis_state()
                                analyze_dataset(all_dfs[0])
                    elif proceed:
                        # Use first dataset only
                        reset_analysis_state()
                        analyze_dataset(all_dfs[0])
                elif proceed:
                    # Only one dataset
                    reset_analysis_state()
                    analyze_dataset(all_dfs[0])
    
    else:  # Sample dataset
        # Enhanced sample dataset selection with descriptions
        st.markdown("### Sample Datasets")
        
        sample_descriptions = {
            "Iris Flower Dataset": "A classic dataset containing measurements of different iris flower species.",
            "Titanic Passengers": "Demographic and survival data for passengers on the Titanic.",
            "Housing Prices": "California housing price dataset with various property features."
        }
        
        # Create sample dataset cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                <div class="stat-card">
                <h4>Iris Flower Dataset</h4>
                <p>150 rows × 5 columns</p>
                <small>Classification dataset with flower measurements</small>
                </div>
                """, 
                unsafe_allow_html=True
            )
            iris_selected = st.button("Load Iris", key="load_iris")
        
        with col2:
            st.markdown(
                """
                <div class="stat-card">
                <h4>Titanic Passengers</h4>
                <p>891 rows × 12 columns</p>
                <small>Survival data with passenger demographics</small>
                </div>
                """, 
                unsafe_allow_html=True
            )
            titanic_selected = st.button("Load Titanic", key="load_titanic")
            
        with col3:
            st.markdown(
                """
                <div class="stat-card">
                <h4>Housing Prices</h4>
                <p>20,640 rows × 10 columns</p>
                <small>California housing market data</small>
                </div>
                """, 
                unsafe_allow_html=True
            )
            housing_selected = st.button("Load Housing", key="load_housing")
        
        # Handle sample dataset loading
        if iris_selected or titanic_selected or housing_selected:
            with st.spinner("Loading sample dataset..."):
                try:
                    if iris_selected:
                        from sklearn.datasets import load_iris
                        data = load_iris()
                        df = pd.DataFrame(data.data, columns=data.feature_names)
                        df['species'] = [data.target_names[i] for i in data.target]
                        sample_dataset = "Iris Flower Dataset"
                        
                    elif titanic_selected:
                        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                        df = pd.read_csv(url)
                        sample_dataset = "Titanic Passengers"
                        
                    else:  # Housing
                        from sklearn.datasets import fetch_california_housing
                        data = fetch_california_housing()
                        df = pd.DataFrame(data.data, columns=data.feature_names)
                        df['price'] = data.target
                        sample_dataset = "Housing Prices"
                    
                    st.session_state.datasets['sample'] = df
                    st.success(f"Loaded {sample_dataset} with {len(df):,} rows and {len(df.columns):,} columns.")
                    
                    # Process the dataset
                    reset_analysis_state()
                    analyze_dataset(df)
                except Exception as e:
                    st.error(f"Error loading sample dataset: {str(e)}")

def render_insights_page():
    """Render the insights page with AI-generated insights."""
    st.markdown("<h2 class='sub-header'>AI-Generated Insights</h2>", unsafe_allow_html=True)
    
    if not st.session_state.insights:
        st.warning("No insights available. Please upload and process data first.")
        return
    
    try:
        # Display dataset summary
        if st.session_state.dataset_summary:
            st.markdown("<h3>Dataset Summary</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='highlight'>{st.session_state.dataset_summary}</div>",
                unsafe_allow_html=True
            )
        
        # Create three columns layout for key metrics
        col1, col2, col3 = st.columns(3)
        
        # Safely access values from insights dict
        basic_info = st.session_state.insights.get('basic_info', {})
        num_rows = basic_info.get('num_rows', 0)
        num_columns = basic_info.get('num_columns', 0)
        
        # Dataset size
        with col1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <p class="card-label">Rows</p>
                    <p class="card-value">{num_rows:,}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Column count
        with col2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <p class="card-label">Columns</p>
                    <p class="card-value">{num_columns:,}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Data completeness
        with col3:
            missing_vals = st.session_state.insights.get('missing_values', {})
            missing_pct = missing_vals.get('missing_percentage', '0%')
            try:
                completeness = 100 - float(missing_pct.rstrip('%'))
            except:
                completeness = 100.0  # Default if parsing fails
            
            st.markdown(
                f"""
                <div class="stat-card">
                    <p class="card-label">Data Completeness</p>
                    <p class="card-value">{completeness:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Display tabs for detailed insights
        insight_tabs = st.tabs([
            "📊 Overview", 
            "🧹 Data Quality", 
            "📈 Statistical Insights",
            "🔍 Advanced Analysis"
        ])
        
        # Overview tab
        with insight_tabs[0]:
            st.subheader("Dataset Overview")
            
            # Column type breakdown
            st.write("Column Type Distribution:")
            col_types = st.session_state.insights.get('column_types', {})
            
            if col_types:
                # Create a plotly pie chart for column types
                labels = list(col_types.keys())
                values = list(col_types.values())
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values,
                    textinfo='label+percent',
                    marker_colors=['#1E88E5', '#42A5F5', '#90CAF9', '#E3F2FD', '#5C6BC0', '#7986CB']
                )])
                fig.update_layout(
                    margin=dict(t=20, b=0, l=20, r=20),
                    height=300,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show cleaning report if available
            if st.session_state.cleaning_reports.get('main'):
                with st.expander("Data Preprocessing Steps", expanded=False):
                    cleaning_report = st.session_state.cleaning_reports['main']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Show operations performed
                        if cleaning_report.get('operations'):
                            st.write("Operations performed:")
                            for issue in cleaning_report['operations']:
                                st.write(f"- {issue}")
                    
                    with col2:
                        # Show issues found
                        if cleaning_report.get('issues_found'):
                            st.write("Issues found:")
                            for issue in cleaning_report['issues_found']:
                                st.write(f"- {issue}")
            
            # Show merge report if available
            if st.session_state.merge_report:
                with st.expander("Dataset Merge Information", expanded=False):
                    merge_report = st.session_state.merge_report
                    st.write(f"Status: {merge_report['status']}")
                    st.write(f"Message: {merge_report['message']}")
                    
                    if merge_report.get('details'):
                        st.write("Details:")
                        for detail in merge_report['details']:
                            st.write(f"- {detail}")
        
        # Data Quality tab
        with insight_tabs[1]:
            st.subheader("Data Quality Assessment")
            
            # Missing values
            missing_vals = st.session_state.insights.get('missing_values', {})
            
            # Create a bar chart for columns with missing values
            cols_with_missing = missing_vals.get('columns_with_missing', {})
            if cols_with_missing:
                missing_count = missing_vals.get('total_missing_cells', 0)
                total_cells = num_rows * num_columns
                missing_percent = missing_vals.get('missing_percentage', '0%')
                
                # Create a header with missing value summary
                st.markdown(
                    f"""
                    <div class="highlight">
                    <p><strong>Missing Data Summary:</strong> {missing_count:,} missing cells ({missing_percent} of all data)</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Sort columns by missing count
                sorted_missing = dict(sorted(cols_with_missing.items(), key=lambda x: x[1], reverse=True))
                
                # Create bar chart for columns with missing values
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(sorted_missing.keys()),
                    y=list(sorted_missing.values()),
                    marker_color='#E57373',
                    text=[f"{val:,} ({val/num_rows:.1%})" for val in sorted_missing.values()],
                    textposition="auto"
                ))
                fig.update_layout(
                    title="Missing Values by Column",
                    xaxis_title="Column",
                    yaxis_title="Missing Count",
                    margin=dict(t=30, b=0, l=0, r=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add missing data pattern analysis
                st.subheader("Missing Data Patterns")
                
                df = st.session_state.datasets.get(st.session_state.active_dataset)
                if df is not None:
                    # Get subset of dataframe with just the columns containing missing values
                    missing_cols = list(sorted_missing.keys())
                    if len(missing_cols) > 0:
                        # Create a boolean mask for missingness patterns (limit to first 10 cols if too many)
                        display_cols = missing_cols[:10]
                        missing_patterns = df[display_cols].isnull()
                        
                        # Count pattern frequencies
                        pattern_counts = missing_patterns.value_counts().reset_index()
                        pattern_counts.columns = list(pattern_counts.columns[:-1]) + ['count']
                        
                        # Display top patterns
                        st.write("Top missing data patterns:")
                        
                        # Format the patterns for display
                        if len(pattern_counts) > 0:
                            # Display as a styled table
                            def highlight_true(val):
                                color = '#ffcdd2' if val else '#dcedc8'
                                return f'background-color: {color}'
                            
                            styled_patterns = pattern_counts.head(5).style.applymap(
                                highlight_true, subset=pattern_counts.columns[:-1]
                            )
                            st.dataframe(styled_patterns)
                        else:
                            st.write("No clear patterns found in missing data.")
                    
            else:
                st.success("No missing values detected in the dataset - excellent data quality!")
                
            # Check for outliers
            st.subheader("Outlier Detection")
            
            numerical_cols = list(st.session_state.insights.get('numerical_stats', {}).keys())
            if numerical_cols:
                df = st.session_state.datasets.get(st.session_state.active_dataset)
                if df is not None:
                    col = st.selectbox("Select column for outlier analysis:", options=numerical_cols)
                    
                    if col:
                        # Calculate quartiles and IQR
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Find outliers
                        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                        outlier_count = outliers_mask.sum()
                        
                        # Display outlier info
                        outlier_percent = outlier_count / len(df) * 100
                        
                        st.markdown(
                            f"""
                            <div class="highlight">
                            <p><strong>Outlier Analysis for {col}:</strong></p>
                            <p>Detected {outlier_count:,} outliers ({outlier_percent:.2f}% of the data)</p>
                            <p>Normal range: {lower_bound:.2f} to {upper_bound:.2f}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Create boxplot
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=df[col],
                            name=col,
                            boxpoints='outliers',
                            jitter=0.3,
                            pointpos=-1.8,
                            marker=dict(
                                color='rgb(7,40,89)',
                                size=5
                            ),
                            line=dict(
                                color='rgb(7,40,89)'
                            )
                        ))
                        fig.update_layout(
                            title=f"Boxplot for {col} (showing outliers)",
                            yaxis_title=col,
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if outlier_count > 0:
                            with st.expander("View outlier values", expanded=False):
                                st.dataframe(df[outliers_mask][col].sort_values(ascending=False).head(20))
                
        # Statistical Insights tab
        with insight_tabs[2]:
            st.subheader("Statistical Analysis")
            
            # Numerical statistics
            numerical_stats = st.session_state.insights.get('numerical_stats', {})
            if numerical_stats:
                # Create selector for numerical columns
                num_cols = list(numerical_stats.keys())
                if num_cols:
                    num_col_select = st.selectbox(
                        "Select a numerical column:",
                        options=num_cols
                    )
                    
                    if num_col_select and num_col_select in numerical_stats:
                        col_stats = numerical_stats[num_col_select]
                        
                        # Create columns for key statistics with enhanced styling
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            if 'mean' in col_stats:
                                st.markdown(
                                    f"""
                                    <div class="stat-card">
                                        <p class="card-label">Mean</p>
                                        <p class="card-value">{col_stats['mean']:.2f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                        with col2:
                            if '50%' in col_stats:
                                st.markdown(
                                    f"""
                                    <div class="stat-card">
                                        <p class="card-label">Median</p>
                                        <p class="card-value">{col_stats['50%']:.2f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                        with col3:
                            if 'std' in col_stats:
                                st.markdown(
                                    f"""
                                    <div class="stat-card">
                                        <p class="card-label">Std Deviation</p>
                                        <p class="card-value">{col_stats['std']:.2f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                        with col4:
                            if "skewness" in col_stats:
                                skew = col_stats['skewness']
                                skew_desc = "Symmetric" if abs(skew) < 0.5 else ("Right Skewed" if skew > 0 else "Left Skewed")
                                st.markdown(
                                    f"""
                                    <div class="stat-card">
                                        <p class="card-label">Skewness ({skew_desc})</p>
                                        <p class="card-value">{skew:.2f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        
                        # Additional stats in expandable section
                        with st.expander("View Full Statistical Profile", expanded=False):
                            # Create a styled table for all statistics
                            stats_dict = {
                                "Minimum": col_stats.get('min', 'N/A'),
                                "Maximum": col_stats.get('max', 'N/A'),
                                "Range": col_stats.get('max', 0) - col_stats.get('min', 0),
                                "25th Percentile": col_stats.get('25%', 'N/A'),
                                "75th Percentile": col_stats.get('75%', 'N/A'),
                                "IQR": col_stats.get('75%', 0) - col_stats.get('25%', 0),
                                "Kurtosis": col_stats.get('kurtosis', 'N/A'),
                                "Count": col_stats.get('count', 'N/A')
                            }
                            
                            # Convert to DataFrame for display
                            stats_df = pd.DataFrame(stats_dict.items(), columns=["Statistic", "Value"])
                            st.table(stats_df)
                        
                        # Get the actual data to create histogram
                        df = st.session_state.datasets.get(st.session_state.active_dataset)
                        if df is not None and num_col_select in df.columns:
                            # Create a histogram for the distribution
                            st.subheader(f"Distribution of {num_col_select}")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=df[num_col_select].dropna(),
                                nbinsx=30,
                                marker_color='#42A5F5',
                                opacity=0.7,
                                name="Histogram"
                            ))
                            
                            # Add mean and median vertical lines
                            if 'mean' in col_stats:
                                fig.add_vline(
                                    x=col_stats['mean'], 
                                    line_dash="dash", 
                                    line_color="red",
                                    annotation_text="Mean",
                                    annotation_position="top right"
                                )
                            
                            if '50%' in col_stats:
                                fig.add_vline(
                                    x=col_stats['50%'], 
                                    line_dash="dash", 
                                    line_color="green",
                                    annotation_text="Median",
                                    annotation_position="top left"
                                )
                            
                            fig.update_layout(
                                xaxis_title=num_col_select,
                                yaxis_title="Frequency",
                                bargap=0.05,
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            correlation_info = st.session_state.insights.get('correlation_info', {})
            if correlation_info and correlation_info.get('correlation_matrix'):
                st.subheader("Correlation Analysis")
                
                # Get the correlation matrix
                corr_matrix = correlation_info.get('correlation_matrix', {})
                
                # Convert to DataFrame
                corr_df = pd.DataFrame(corr_matrix)
                
                # Create interactive heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=corr_df.values.round(2),
                    texttemplate="%{text}",
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title="Correlation Matrix Heatmap",
                    height=500,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    yaxis_autorange='reversed'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display high correlations
                high_corrs = correlation_info.get('high_correlations', [])
                if high_corrs:
                    st.subheader("Key Correlations")
                    
                    # Create key correlations table with icons
                    for i, corr in enumerate(high_corrs):
                        if isinstance(corr, dict) and 'column1' in corr and 'column2' in corr and 'correlation' in corr:
                            corr_val = corr['correlation']
                            corr_color = "blue" if corr_val > 0 else "red"
                            relationship = corr.get('relationship', 'strong correlation')
                            
                            corr_icon = "📈" if corr_val > 0 else "📉"
                            
                            st.markdown(
                                f"""
                                <div class="insight-card">
                                <h4>{corr_icon} {relationship.capitalize()}</h4>
                                <p><b>{corr['column1']}</b> and <b>{corr['column2']}</b> have a 
                                <span style='color:{corr_color}'><b>{relationship}</b></span> 
                                with correlation coefficient <b>{corr_val:.2f}</b></p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
        
        # Advanced Analysis tab
        with insight_tabs[3]:
            st.subheader("Advanced Insights")
            
            # Check if we have advanced insights
            if not st.session_state.insights.get('advanced_insights'):
                # Button to generate advanced insights
                if st.button("Generate Advanced Insights with AI", type="primary"):
                    with st.spinner("AI is analyzing patterns in your data..."):
                        try:
                            df = st.session_state.datasets[st.session_state.active_dataset]
                            adv_insights = generate_advanced_insights(df)
                            
                            if adv_insights.get("success", False):
                                st.session_state.insights['advanced_insights'] = json_serializable(adv_insights["insights"])
                                st.rerun()
                            else:
                                st.error(f"Could not generate advanced insights: {adv_insights.get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error generating advanced insights: {str(e)}")
            else:
                # Display advanced insights with improved styling
                advanced_insights = st.session_state.insights.get('advanced_insights', [])
                if isinstance(advanced_insights, list):
                    # Group insights by confidence level
                    high_conf = [i for i in advanced_insights if isinstance(i, dict) and i.get('confidence', '').lower() == 'high']
                    med_conf = [i for i in advanced_insights if isinstance(i, dict) and i.get('confidence', '').lower() == 'medium']
                    low_conf = [i for i in advanced_insights if isinstance(i, dict) and i.get('confidence', '').lower() == 'low']
                    
                    if high_conf:
                        st.markdown("### High Confidence Insights")
                        for insight in high_conf:
                            title = insight.get('title', 'Insight')
                            description = insight.get('description', 'No description available')
                            related_columns = insight.get('related_columns', ['N/A'])
                            
                            st.markdown(
                                f"""
                                <div class="insight-card" style="border-left: 5px solid #4CAF50;">
                                <h4>🔍 {title}</h4>
                                <p>{description}</p>
                                <p style='color:#4CAF50'><b>Confidence:</b> High</p>
                                <p><b>Related columns:</b> {', '.join(related_columns)}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    if med_conf:
                        st.markdown("### Medium Confidence Insights")
                        for insight in med_conf:
                            title = insight.get('title', 'Insight')
                            description = insight.get('description', 'No description available')
                            related_columns = insight.get('related_columns', ['N/A'])
                            
                            st.markdown(
                                f"""
                                <div class="insight-card" style="border-left: 5px solid #FFC107;">
                                <h4>🔍 {title}</h4>
                                <p>{description}</p>
                                <p style='color:#FFC107'><b>Confidence:</b> Medium</p>
                                <p><b>Related columns:</b> {', '.join(related_columns)}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    if low_conf:
                        st.markdown("### Potential Insights (Low Confidence)")
                        for insight in low_conf:
                            title = insight.get('title', 'Insight')
                            description = insight.get('description', 'No description available')
                            related_columns = insight.get('related_columns', ['N/A'])
                            
                            st.markdown(
                                f"""
                                <div class="insight-card" style="border-left: 5px solid #F44336;">
                                <h4>🔍 {title}</h4>
                                <p>{description}</p>
                                <p style='color:#F44336'><b>Confidence:</b> Low</p>
                                <p><b>Related columns:</b> {', '.join(related_columns)}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            
            # Next steps suggestions
            st.markdown("### What's Next?")
            if st.button("Suggest Next Analytical Steps", type="primary"):
                with st.spinner("AI is generating suggestions..."):
                    try:
                        df = st.session_state.datasets[st.session_state.active_dataset]
                        next_steps = suggest_next_steps(df, st.session_state.insights)
                        
                        for i, step in enumerate(next_steps):
                            if isinstance(step, dict):
                                title = step.get('title', 'Suggestion')
                                description = step.get('description', 'No description available')
                                
                                st.markdown(
                                    f"""
                                    <div class="insight-card">
                                    <h4>🚀 Step {i+1}: {title}</h4>
                                    <p>{description}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    except Exception as e:
                        st.error(f"Error generating next steps: {str(e)}")
    except Exception as e:
        st.error(f"Error rendering insights page: {str(e)}")
        st.error(traceback.format_exc())

def render_visualizations_page():
    """Render the visualizations page with AI-recommended charts."""
    st.markdown("<h2 class='sub-header'>Interactive Data Visualizations</h2>", unsafe_allow_html=True)
    
    if not st.session_state.visualization_recs:
        st.warning("No visualization recommendations available. Please process data first.")
        return
    
    try:
        # Get active dataset
        df = st.session_state.datasets.get(st.session_state.active_dataset)
        if df is None:
            st.error("No active dataset found.")
            return
        
        # Create tabs - recommended vs. custom
        viz_tabs = st.tabs(["AI-Recommended", "Custom Visualization", "Download Charts"])
        
        # AI-Recommended visualizations
        with viz_tabs[0]:
            st.markdown(
                """
                <div class="highlight">
                <p>The AI has analyzed your data and recommended these visualizations based on your dataset's characteristics.</p>
                <p>Click on any visualization card to generate and display the chart.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Create responsive grid layout for visualization cards
            viz_recs = st.session_state.visualization_recs
            
            # Create visualization cards in rows of 2
            for i in range(0, len(viz_recs), 2):
                col1, col2 = st.columns(2)
                
                # First visualization in the row
                with col1:
                    if i < len(viz_recs):
                        viz = viz_recs[i]
                        with st.expander(viz["title"], expanded=i < 2):  # Expand first 2 by default
                            st.markdown(f"**Description:** {viz['description']}")
                            st.markdown(f"**Columns:** {', '.join(viz['columns'])}")
                            
                            # Check if visualization already generated
                            viz_id = viz["id"]
                            if viz_id in st.session_state.generated_visualizations:
                                viz_result = st.session_state.generated_visualizations[viz_id]
                                
                                if viz_result["success"]:
                                    try:
                                        # Convert the JSON string back to a plotly figure
                                        fig_dict = json.loads(viz_result["plot_data"])
                                        fig = go.Figure(fig_dict)
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error displaying visualization: {str(e)}")
                                else:
                                    st.error(f"Error generating visualization: {viz_result.get('error')}")
                            else:
                                # Generate the visualization on demand
                                if st.button(f"Generate {viz['title']}", key=f"gen_{viz_id}"):
                                    with st.spinner("Generating visualization..."):
                                        try:
                                            viz_result = generate_visualization(
                                                df, 
                                                viz["type"],
                                                viz["columns"],
                                                viz["parameters"]
                                            )
                                            
                                            st.session_state.generated_visualizations[viz_id] = viz_result
                                            
                                            if viz_result["success"]:
                                                # Convert the JSON string back to a plotly figure
                                                fig_dict = json.loads(viz_result["plot_data"])
                                                fig = go.Figure(fig_dict)
                                                st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.error(f"Error generating visualization: {viz_result.get('error')}")
                                        except Exception as e:
                                            st.error(f"Error generating visualization: {str(e)}")
                
                # Second visualization in the row
                with col2:
                    if i + 1 < len(viz_recs):
                        viz = viz_recs[i + 1]
                        with st.expander(viz["title"], expanded=(i + 1) < 2):  # Expand first 2 by default
                            st.markdown(f"**Description:** {viz['description']}")
                            st.markdown(f"**Columns:** {', '.join(viz['columns'])}")
                            
                            # Check if visualization already generated
                            viz_id = viz["id"]
                            if viz_id in st.session_state.generated_visualizations:
                                viz_result = st.session_state.generated_visualizations[viz_id]
                                
                                if viz_result["success"]:
                                    try:
                                        # Convert the JSON string back to a plotly figure
                                        fig_dict = json.loads(viz_result["plot_data"])
                                        fig = go.Figure(fig_dict)
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error displaying visualization: {str(e)}")
                                else:
                                    st.error(f"Error generating visualization: {viz_result.get('error')}")
                            else:
                                # Generate the visualization on demand
                                if st.button(f"Generate {viz['title']}", key=f"gen_{viz_id}"):
                                    with st.spinner("Generating visualization..."):
                                        try:
                                            viz_result = generate_visualization(
                                                df, 
                                                viz["type"],
                                                viz["columns"],
                                                viz["parameters"]
                                            )
                                            
                                            st.session_state.generated_visualizations[viz_id] = viz_result
                                            
                                            if viz_result["success"]:
                                                # Convert the JSON string back to a plotly figure
                                                fig_dict = json.loads(viz_result["plot_data"])
                                                fig = go.Figure(fig_dict)
                                                st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.error(f"Error generating visualization: {viz_result.get('error')}")
                                        except Exception as e:
                                            st.error(f"Error generating visualization: {str(e)}")
        
        # Custom visualization builder
        with viz_tabs[1]:
            st.markdown(
                """
                <div class="highlight">
                <p>Build your own custom visualization by selecting chart type and columns.</p>
                <p>Experiment with different chart types to find the best visualization for your data.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Select chart type
            chart_type = st.selectbox(
                "Select chart type:",
                options=["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Heatmap", "Pie Chart"]
            )
            
            # Get columns based on data types
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Show appropriate column selectors based on chart type
            if chart_type == "Scatter Plot":
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns for a scatter plot.")
                else:
                    x_col = st.selectbox("X-axis (numeric):", options=numeric_cols)
                    y_col = st.selectbox("Y-axis (numeric):", options=numeric_cols, index=min(1, len(numeric_cols)-1))
                    color_col = st.selectbox("Color by (optional):", options=["None"] + categorical_cols)
                    
                    if st.button("Generate Custom Scatter Plot", type="primary"):
                        with st.spinner("Creating visualization..."):
                            try:
                                if color_col != "None":
                                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
                                else:
                                    fig = px.scatter(df, x=x_col, y=y_col)
                                
                                fig.update_layout(
                                    title=f"Scatter Plot: {y_col} vs {x_col}",
                                    xaxis_title=x_col,
                                    yaxis_title=y_col,
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add explanatory text
                                if color_col != "None":
                                    st.markdown(f"""
                                    **Interpretation:** This scatter plot shows the relationship between **{x_col}** and **{y_col}**, 
                                    colored by **{color_col}**. Each point represents a single data record.
                                    
                                    Look for:
                                    - Patterns or clusters in the data
                                    - Positive or negative correlations
                                    - Outliers or unusual points
                                    - Differences between categories (colors)
                                    """)
                                else:
                                    st.markdown(f"""
                                    **Interpretation:** This scatter plot shows the relationship between **{x_col}** and **{y_col}**. 
                                    Each point represents a single data record.
                                    
                                    Look for:
                                    - Positive correlation (points trending upward)
                                    - Negative correlation (points trending downward)
                                    - No correlation (random scatter)
                                    - Outliers or unusual points
                                    """)
                            except Exception as e:
                                st.error(f"Error creating scatter plot: {str(e)}")
                    
            elif chart_type == "Line Chart":
                if datetime_cols:
                    x_col = st.selectbox("X-axis (date/time):", options=datetime_cols)
                else:
                    x_col = st.selectbox("X-axis:", options=numeric_cols)
                    
                if not numeric_cols:
                    st.warning("Need at least 1 numeric column for a line chart.")
                else:
                    y_col = st.selectbox("Y-axis (numeric):", options=numeric_cols)
                    group_col = st.selectbox("Group by (optional):", options=["None"] + categorical_cols)
                    
                    if st.button("Generate Custom Line Chart", type="primary"):
                        with st.spinner("Creating visualization..."):
                            try:
                                # Sort by x-axis column
                                plot_df = df.sort_values(by=x_col)
                                
                                if group_col != "None":
                                    fig = px.line(plot_df, x=x_col, y=y_col, color=group_col)
                                else:
                                    fig = px.line(plot_df, x=x_col, y=y_col)
                                
                                fig.update_layout(
                                    title=f"Line Chart: {y_col} over {x_col}",
                                    xaxis_title=x_col,
                                    yaxis_title=y_col,
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add explanatory text
                                if group_col != "None":
                                    st.markdown(f"""
                                    **Interpretation:** This line chart shows how **{y_col}** changes over **{x_col}**, 
                                    grouped by **{group_col}**. Each line represents a different category.
                                    
                                    Look for:
                                    - Trends over time or sequence
                                    - Seasonal patterns or cycles
                                    - Differences between groups
                                    - Unusual spikes or drops
                                    """)
                                else:
                                    st.markdown(f"""
                                    **Interpretation:** This line chart shows how **{y_col}** changes over **{x_col}**.
                                    
                                    Look for:
                                    - Overall trends (increasing, decreasing, stable)
                                    - Seasonal patterns or cycles
                                    - Unusual spikes or drops
                                    - Turning points or change in trends
                                    """)
                            except Exception as e:
                                st.error(f"Error creating line chart: {str(e)}")
                    
            elif chart_type == "Bar Chart":
                if not categorical_cols:
                    st.warning("Need at least 1 categorical column for a bar chart.")
                elif not numeric_cols:
                    st.warning("Need at least 1 numeric column for a bar chart.")
                else:
                    x_col = st.selectbox("X-axis (categories):", options=categorical_cols)
                    y_col = st.selectbox("Y-axis (numeric):", options=numeric_cols)
                    
                    # Add orientation option
                    orientation = st.radio(
                        "Bar orientation:",
                        options=["Vertical", "Horizontal"],
                        horizontal=True,
                        key="bar_orientation_selection"
                    )
                    sort_bars = st.checkbox("Sort bars by value", value=True)
                    
                    if st.button("Generate Custom Bar Chart", type="primary"):
                        with st.spinner("Creating visualization..."):
                            try:
                                # For large categories, limit to top N
                                if df[x_col].nunique() > 20:
                                    # Get top categories by count
                                    top_cats = df[x_col].value_counts().head(20).index.tolist()
                                    plot_df = df[df[x_col].isin(top_cats)]
                                    st.info(f"Showing only top 20 categories (out of {df[x_col].nunique()}).")
                                else:
                                    plot_df = df
                                
                                # Group data
                                grouped = plot_df.groupby(x_col)[y_col].mean().reset_index()
                                
                                # Sort if requested
                                if sort_bars:
                                    grouped = grouped.sort_values(by=y_col, ascending=False)
                                
                                # Create the bar chart - horizontal or vertical based on selection
                                if orientation == "Horizontal":
                                    fig = px.bar(
                                        grouped, 
                                        y=x_col, 
                                        x=y_col,
                                        text_auto='.2s',
                                        title=f"Bar Chart: Average {y_col} by {x_col}"
                                    )
                                    fig.update_layout(
                                        xaxis_title=f"Average {y_col}",
                                        yaxis_title=x_col,
                                        height=max(500, len(grouped) * 25)  # Adjust height based on number of bars
                                    )
                                else:
                                    fig = px.bar(
                                        grouped, 
                                        x=x_col, 
                                        y=y_col,
                                        text_auto='.2s',
                                        title=f"Bar Chart: Average {y_col} by {x_col}"
                                    )
                                    fig.update_layout(
                                        xaxis_title=x_col,
                                        yaxis_title=f"Average {y_col}",
                                        height=500
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add explanatory text
                                st.markdown(f"""
                                **Interpretation:** This bar chart shows the average **{y_col}** for each **{x_col}** category.
                                
                                Look for:
                                - Categories with highest and lowest values
                                - Significant differences between categories
                                - Patterns or groupings of similar categories
                                - Outlier categories that differ dramatically from others
                                """)
                            except Exception as e:
                                st.error(f"Error creating bar chart: {str(e)}")
                    
            elif chart_type == "Box Plot":
                if not numeric_cols:
                    st.warning("Need at least 1 numeric column for a box plot.")
                else:
                    y_col = st.selectbox("Values (numeric):", options=numeric_cols)
                    x_col = st.selectbox("Group by (categorical):", options=["None"] + categorical_cols)
                    
                    if st.button("Generate Custom Box Plot", type="primary"):
                        with st.spinner("Creating visualization..."):
                            try:
                                if x_col != "None":
                                    # For large categories, limit to top N
                                    if df[x_col].nunique() > 15:
                                        # Get top categories by count
                                        top_cats = df[x_col].value_counts().head(15).index.tolist()
                                        plot_df = df[df[x_col].isin(top_cats)]
                                        st.info(f"Showing only top 15 categories (out of {df[x_col].nunique()}).")
                                    else:
                                        plot_df = df
                                        
                                    fig = px.box(
                                        plot_df, 
                                        x=x_col, 
                                        y=y_col,
                                        title=f"Box Plot: Distribution of {y_col} by {x_col}",
                                        points="outliers"
                                    )
                                else:
                                    fig = px.box(
                                        df, 
                                        y=y_col,
                                        title=f"Box Plot: Distribution of {y_col}",
                                        points="outliers"
                                    )
                                
                                fig.update_layout(
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add explanatory text
                                if x_col != "None":
                                    st.markdown(f"""
                                    **Interpretation:** This box plot shows the distribution of **{y_col}** for each **{x_col}** category.
                                    
                                    Each box shows:
                                    - The median (middle line)
                                    - Interquartile range (IQR) from 25th to 75th percentile (box)
                                    - Whiskers extending to 1.5 × IQR
                                    - Outliers (individual points beyond whiskers)
                                    
                                    Look for:
                                    - Differences in medians across categories
                                    - Differences in spread or variability
                                    - Skewness (asymmetry in the boxes)
                                    - Outliers in specific categories
                                    """)
                                else:
                                    st.markdown(f"""
                                    **Interpretation:** This box plot shows the distribution of **{y_col}**.
                                    
                                    The box shows:
                                    - The median (middle line)
                                    - Interquartile range (IQR) from 25th to 75th percentile (box)
                                    - Whiskers extending to 1.5 × IQR
                                    - Outliers (individual points beyond whiskers)
                                    
                                    Look for:
                                    - Central tendency (median)
                                    - Spread or variability (width of box/whiskers)
                                    - Skewness (asymmetry in the box)
                                    - Outliers (points beyond whiskers)
                                    """)
                            except Exception as e:
                                st.error(f"Error creating box plot: {str(e)}")
                    
            elif chart_type == "Heatmap":
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns for a heatmap.")
                else:
                    selected_cols = st.multiselect(
                        "Select columns (numeric):", 
                        options=numeric_cols,
                        default=numeric_cols[:min(len(numeric_cols), 6)]  # Select first 6 by default
                    )
                    
                    if st.button("Generate Custom Heatmap", type="primary"):
                        if len(selected_cols) < 2:
                            st.error("Please select at least 2 columns for the heatmap.")
                        else:
                            with st.spinner("Creating visualization..."):
                                try:
                                    # Calculate correlation matrix
                                    corr_matrix = df[selected_cols].corr().round(2)
                                    
                                    # Create heatmap
                                    fig = go.Figure(data=go.Heatmap(
                                        z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.index,
                                        colorscale='RdBu_r',
                                        zmid=0,
                                        text=corr_matrix.values.round(2),
                                        texttemplate="%{text}",
                                        hoverongaps=False
                                    ))
                                    
                                    fig.update_layout(
                                        title="Correlation Heatmap",
                                        height=600,
                                        width=700
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add explanatory text
                                    st.markdown("""
                                    **Interpretation:** This heatmap shows the correlation between each pair of selected numerical variables.
                                    
                                    Correlation ranges from -1 to 1:
                                    - **1**: Perfect positive correlation (as one variable increases, the other increases proportionally)
                                    - **0**: No correlation (no relationship between variables)
                                    - **-1**: Perfect negative correlation (as one variable increases, the other decreases proportionally)
                                    
                                    Look for:
                                    - Strong positive correlations (dark blue)
                                    - Strong negative correlations (dark red)
                                    - Clusters of correlated variables
                                    - Independent variables (near-zero correlations)
                                    """)
                                except Exception as e:
                                    st.error(f"Error creating heatmap: {str(e)}")
                    
            elif chart_type == "Pie Chart":
                if not categorical_cols:
                    st.warning("Need at least 1 categorical column for a pie chart.")
                else:
                    cat_col = st.selectbox("Category column:", options=categorical_cols)
                    
                    # Option for count or sum
                    aggregation = st.radio(
                        "Aggregation method:",
                        options=["Count (frequency)", "Sum of values"],
                        horizontal=True,
                        key="aggregation_method_selection"
                    )
                    
                    # If sum, need a numeric column
                    if aggregation == "Sum of values":
                        if not numeric_cols:
                            st.warning("Need at least 1 numeric column for sum aggregation.")
                        else:
                            value_col = st.selectbox("Value column to sum:", options=numeric_cols)
                    
                    if st.button("Generate Custom Pie Chart", type="primary"):
                        with st.spinner("Creating visualization..."):
                            try:
                                # Get value counts
                                if aggregation == "Count (frequency)":
                                    counts = df[cat_col].value_counts()
                                    title = f"Distribution of {cat_col} (Counts)"
                                else:  # Sum
                                    counts = df.groupby(cat_col)[value_col].sum()
                                    title = f"Distribution of {cat_col} (Sum of {value_col})"
                                
                                # If too many categories, limit to top 9 and group others
                                if len(counts) > 9:
                                    top_counts = counts.head(8)
                                    others = pd.Series({'Others': counts[8:].sum()})
                                    counts = pd.concat([top_counts, others])
                                    st.info(f"Grouped {len(counts) - 9} smaller categories as 'Others'.")
                                
                                # Create the pie chart
                                fig = px.pie(
                                    values=counts.values,
                                    names=counts.index,
                                    title=title,
                                    hole=0.4  # Make it a donut chart for better readability
                                )
                                
                                fig.update_layout(
                                    height=500
                                )
                                
                                fig.update_traces(
                                    textposition='inside',
                                    textinfo='percent+label',
                                    insidetextorientation='radial'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add explanatory text
                                if aggregation == "Count (frequency)":
                                    st.markdown(f"""
                                    **Interpretation:** This pie chart shows the distribution of **{cat_col}** categories by count (frequency).
                                    
                                    Look for:
                                    - Dominant categories (largest slices)
                                    - Even or uneven distribution
                                    - Rare categories (smallest slices)
                                    - Relative proportions between categories
                                    """)
                                else:
                                    st.markdown(f"""
                                    **Interpretation:** This pie chart shows the distribution of **{cat_col}** categories by sum of **{value_col}**.
                                    
                                    Look for:
                                    - Categories with highest contribution to total sum
                                    - Disparities between categories
                                    - Categories that contribute disproportionately to their frequency
                                    - Relative proportions between categories
                                    """)
                            except Exception as e:
                                st.error(f"Error creating pie chart: {str(e)}")
                    
        # Download Charts tab
        with viz_tabs[2]:
            st.markdown(
                """
                <div class="highlight">
                <p>Download generated visualizations for use in presentations or reports.</p>
                <p>All visualizations you've generated are available here for download as PNG files.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Check if any visualizations have been generated
            if st.session_state.generated_visualizations:
                st.subheader("Generated Visualizations")
                
                # List of generated visualizations with download buttons
                for viz_id, viz_result in st.session_state.generated_visualizations.items():
                    if viz_result["success"]:
                        try:
                            # Get title from the original recommendation
                            title = next((viz["title"] for viz in st.session_state.visualization_recs if viz["id"] == viz_id), viz_id)
                            
                            # Create a section for each visualization
                            st.markdown(f"### {title}")
                            
                            # Convert the JSON string back to a plotly figure
                            fig_dict = json.loads(viz_result["plot_data"])
                            fig = go.Figure(fig_dict)
                            
                            # Display the visualization
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add HTML download for PNG
                            # Note: This requires JavaScript to be enabled
                            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
                            b64_img = base64.b64encode(img_bytes).decode()
                            
                            st.markdown(
                                f"""
                                <a href="data:image/png;base64,{b64_img}" download="{title.replace(' ', '_')}.png">
                                    <button style="
                                        background-color: #4CAF50;
                                        border: none;
                                        color: white;
                                        padding: 10px 24px;
                                        text-align: center;
                                        text-decoration: none;
                                        display: inline-block;
                                        font-size: 16px;
                                        margin: 4px 2px;
                                        cursor: pointer;
                                        border-radius: 4px;
                                    ">
                                        Download as PNG
                                    </button>
                                </a>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            st.markdown("---")
                        except Exception as e:
                            st.error(f"Error preparing visualization for download: {str(e)}")
            else:
                st.info("No visualizations have been generated yet. Go to the AI-Recommended or Custom Visualization tabs to create charts.")
                
    except Exception as e:
        st.error(f"Error rendering visualizations page: {str(e)}")
        st.error(traceback.format_exc())

def render_chat_page():
    """Render the chat interface for asking questions about the data."""
    st.markdown("<h2 class='sub-header'>Chat with Your Data</h2>", unsafe_allow_html=True)
    
    if not st.session_state.datasets:
        st.warning("No data available. Please upload and process data first.")
        return
    
    try:
        # Get active dataset
        df = st.session_state.datasets.get(st.session_state.active_dataset)
        if df is None:
            st.error("No active dataset found.")
            return
        
        # Introduction to the chat feature
        st.markdown(
            """
            <div class="highlight">
            <p><strong>Ask questions about your data in plain English!</strong></p>
            <p>Examples:</p>
            <ul>
                <li>What is the average age?</li>
                <li>How many missing values are in each column?</li>
                <li>What's the relationship between column A and column B?</li>
                <li>What insights can you extract from this dataset?</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                    <p><strong>You:</strong> {message['content']}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #1E88E5;'>
                    <p><strong>AI:</strong> {message['content']}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Input area for user question
        with st.form(key="chat_form"):
            user_query = st.text_area("Ask a question about your data:", height=100)
            submit_button = st.form_submit_button("Ask AI", type="primary")
            
            if submit_button and user_query:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_query
                })
                
                # Prepare context for LLM
                with st.spinner("AI is analyzing your question..."):
                    try:
                        # Create a compact data summary for context
                        schema_data = infer_schema(df)
                        sample_data = df.head(5).to_dict(orient="records")
                        
                        context = {
                            "dataset_info": {
                                "rows": len(df),
                                "columns": len(df.columns),
                                "column_types": {str(col): str(dtype) for col, dtype in df.dtypes.items()}
                            },
                            "schema": schema_data,
                            "sample_data": sample_data,
                            "data_insights": st.session_state.insights
                        }
                        
                        # Convert to JSON serializable format BEFORE trying to dump to JSON
                        serializable_context = json_serializable(context)
                        context_str = json.dumps(serializable_context, indent=2)
                        
                        # Get response from LLM
                        response = query_together_ai(user_query, context_str)
                        
                        # Add AI response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        # Force UI refresh to show new messages
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        st.error(traceback.format_exc())
        
        # Suggested questions
        st.markdown("<h4>Suggested Questions</h4>", unsafe_allow_html=True)
        
        # Create two columns for suggestion buttons layout
        col1, col2 = st.columns(2)
        
        # Generate suggested questions based on dataset
        numerical_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        
        suggested_questions = []
        
        # Add questions about numerical columns
        if len(numerical_cols) > 0:
            col_name = numerical_cols[0]
            suggested_questions.append(f"What is the average {col_name}?")
            
            if len(numerical_cols) > 1:
                col2_name = numerical_cols[1]
                suggested_questions.append(f"Is there a correlation between {col_name} and {col2_name}?")
                suggested_questions.append(f"What are the top 5 outliers in {col_name}?")
        
        # Add questions about categorical columns
        if len(categorical_cols) > 0:
            col_name = categorical_cols[0]
            suggested_questions.append(f"What is the most common value in {col_name}?")
            suggested_questions.append(f"What is the distribution of {col_name}?")
            
            if len(numerical_cols) > 0 and len(categorical_cols) > 0:
                num_col = numerical_cols[0]
                cat_col = categorical_cols[0]
                suggested_questions.append(f"What is the average {num_col} for each {cat_col}?")
        
        # Add time-based question if datetime columns exist
        if len(datetime_cols) > 0 and len(numerical_cols) > 0:
            date_col = datetime_cols[0]
            num_col = numerical_cols[0]
            suggested_questions.append(f"How does {num_col} change over time?")
            suggested_questions.append(f"Is there any seasonality in the {num_col} values?")
        
        # Add general analytical questions
        suggested_questions.extend([
            "What are the key insights from this data?",
            "What patterns or trends can you identify?",
            "Are there any outliers in the data?",
            "What recommendations would you make based on this data?",
            "How complete is this dataset and where are the gaps?",
            "What further analysis would you recommend?"
        ])
        
        # Display suggested questions as styled buttons in two columns
        with col1:
            for i in range(0, len(suggested_questions), 2):
                if i < len(suggested_questions):
                    if st.button(
                        suggested_questions[i], 
                        key=f"sugg_{i}",
                        help="Click to ask this question"
                    ):
                        try:
                            # Add to chat history and generate response
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": suggested_questions[i]
                            })
                            
                            # Prepare context and get response
                            schema_data = infer_schema(df)
                            sample_data = df.head(5).to_dict(orient="records")
                            
                            context = {
                                "dataset_info": {
                                    "rows": len(df),
                                    "columns": len(df.columns),
                                    "column_types": {str(col): str(dtype) for col, dtype in df.dtypes.items()}
                                },
                                "schema": schema_data,
                                "sample_data": sample_data,
                                "data_insights": st.session_state.insights
                            }
                            
                            # Convert to JSON serializable format first
                            serializable_context = json_serializable(context)
                            context_str = json.dumps(serializable_context, indent=2)
                            
                            response = query_together_ai(suggested_questions[i], context_str)
                            
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing suggested question: {str(e)}")
                            st.error(traceback.format_exc())
                        
        with col2:
            for i in range(1, len(suggested_questions), 2):
                if i < len(suggested_questions):
                    if st.button(
                        suggested_questions[i], 
                        key=f"sugg_{i}",
                        help="Click to ask this question"
                    ):
                        try:
                            # Add to chat history and generate response
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": suggested_questions[i]
                            })
                            
                            # Prepare context and get response
                            schema_data = infer_schema(df)
                            sample_data = df.head(5).to_dict(orient="records")
                            
                            context = {
                                "dataset_info": {
                                    "rows": len(df),
                                    "columns": len(df.columns),
                                    "column_types": {str(col): str(dtype) for col, dtype in df.dtypes.items()}
                                },
                                "schema": schema_data,
                                "sample_data": sample_data,
                                "data_insights": st.session_state.insights
                            }
                            
                            # Convert to JSON serializable format first
                            serializable_context = json_serializable(context)
                            context_str = json.dumps(serializable_context, indent=2)
                            
                            response = query_together_ai(suggested_questions[i], context_str)
                            
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing suggested question: {str(e)}")
                            st.error(traceback.format_exc())
    except Exception as e:
        st.error(f"Error rendering chat page: {str(e)}")
        st.error(traceback.format_exc())

def render_report_page():
    """Render the report generation page."""
    st.markdown("<h2 class='sub-header'>Generate Analysis Report</h2>", unsafe_allow_html=True)
    
    if not st.session_state.datasets or not st.session_state.insights:
        st.warning("No data or insights available. Please upload and analyze data first.")
        return
    
    try:
        # Get active dataset
        df = st.session_state.datasets.get(st.session_state.active_dataset)
        if df is None:
            st.error("No active dataset found.")
            return
        
        # Introduction to report generation
        st.markdown(
            """
            <div class="highlight">
            <p><strong>Generate a comprehensive PDF report with all insights and visualizations.</strong></p>
            <p>This report includes data quality assessment, statistical analysis, key insights,
            and actionable recommendations that you can share with stakeholders.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Report preview
        st.markdown(
            """
            <div style="border:1px solid #ddd; padding:15px; border-radius:5px; margin-bottom:20px;">
            <h4 style="margin-top:0">Sample Report Content</h4>
            <ul>
                <li><strong>Executive Summary:</strong> AI-generated overview of key findings</li>
                <li><strong>Data Quality Assessment:</strong> Completeness, consistency, and issues</li>
                <li><strong>Key Insights:</strong> Patterns, correlations, and notable findings</li>
                <li><strong>Statistical Analysis:</strong> Detailed metrics and distributions</li>
                <li><strong>Advanced Insights:</strong> AI-powered deeper analysis</li>
                <li><strong>Recommendations:</strong> Data-driven suggestions for action</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Report configuration
        with st.form("report_config"):
            # Use columns to arrange form fields
            col1, col2 = st.columns(2)
            
            with col1:
                report_title = st.text_input("Report Title", value="Data Analysis Report")
                
                include_sections = st.multiselect(
                    "Sections to include:",
                    options=["Executive Summary", "Data Quality", "Key Insights", "Visualizations", "Statistical Analysis", "Recommendations", "Advanced Insights"],
                    default=["Executive Summary", "Data Quality", "Key Insights", "Visualizations", "Statistical Analysis", "Recommendations"]
                )
            
            with col2:
                # Executive Summary configuration
                if "Executive Summary" in include_sections:
                    summary_length = st.select_slider(
                        "Summary Detail Level",
                        options=["Brief", "Standard", "Detailed"],
                        value="Standard",
                        help="Controls the length and detail of the executive summary"
                    )
                
                # Visualizations configuration
                if "Visualizations" in include_sections:
                    max_viz = st.slider(
                        "Maximum number of visualizations",
                        min_value=1,
                        max_value=10,
                        value=5,
                        help="Limit the number of visualization charts in the report"
                    )
            
            # Report formatting options
            st.markdown("#### Report Style Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                report_color = st.color_picker("Primary color theme", "#1E88E5")
            
            with col2:
                font_size = st.select_slider(
                    "Font size",
                    options=["Small", "Medium", "Large"],
                    value="Medium"
                )
            
            with col3:
                include_timestamp = st.checkbox("Include timestamp", value=True)
            
            # Submit button
            submit_report = st.form_submit_button("Generate Report", type="primary", use_container_width=True)
        
        if submit_report:
            with st.spinner("Generating comprehensive report... This may take a moment."):
                try:
                    # Show progress animation
                    st.markdown("<div class='progress-bar'></div>", unsafe_allow_html=True)
                    
                    # Get any pre-generated visualizations
                    visualizations = list(st.session_state.generated_visualizations.values())
                    
                    # Make sure we have a summary
                    summary = st.session_state.dataset_summary
                    if not summary:
                        summary = summarize_dataset(df)
                    
                    # Generate the report
                    report_path = generate_pdf_report(df, st.session_state.insights, visualizations, summary)
                    
                    if report_path and isinstance(report_path, str) and report_path.startswith("Error"):
                        st.error(report_path)
                    else:
                        st.success("Report generated successfully!")
                        
                        # Show a preview image of the PDF
                        st.markdown("### Report Preview")
                        st.markdown(
                            """
                            <div style="border:1px solid #ddd; padding:10px; text-align:center; margin-bottom:20px; background-color:#f8f9fa; border-radius:5px;">
                            <img src="https://img.icons8.com/color/96/000000/pdf.png" style="width:100px;">
                            <p style="margin-top:10px;">Your PDF report is ready to download</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Create a download button
                        with open(report_path, "rb") as file:
                            btn = st.download_button(
                                label="Download Report",
                                data=file,
                                file_name="data_analysis_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        
                        # Add suggestions for next steps
                        st.markdown("### Next Steps")
                        st.markdown(
                            """
                            - Share this report with stakeholders
                            - Use the insights to guide decision making
                            - Continue exploring specific areas of interest in the Chat tab
                            - Generate custom visualizations to highlight key findings
                            """
                        )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.error(traceback.format_exc())
    except Exception as e:
        st.error(f"Error rendering report page: {str(e)}")
        st.error(traceback.format_exc())

# --- Main Application ---
def main():
    """Main application function."""
    render_header()
    render_sidebar()
    
    # Render the appropriate page based on current tab
    current_tab = st.session_state.current_tab
    
    if current_tab == "upload":
        render_upload_page()
    elif current_tab == "insights":
        render_insights_page()
    elif current_tab == "visualizations":
        render_visualizations_page()
    elif current_tab == "chat":
        render_chat_page()
    elif current_tab == "report":
        render_report_page()

if __name__ == "__main__":
    main()