# data_utils.py

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import traceback
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# === Config ===
OPENROUTER_API_KEY = "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"
LLM_MODEL = "meta-llama/llama-4-maverick:free"

# Enhanced json_serializable function to handle Timestamp objects

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
    elif isinstance(obj, pd.Timestamp):  # Add specific handling for Timestamp objects
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif hasattr(obj, 'to_dict'):  # For pandas objects with to_dict method
        return json_serializable(obj.to_dict())
    else:
        return obj

# === Data Cleaning and Preprocessing ===
def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Comprehensive data cleaning and preprocessing.
    Returns cleaned dataframe and cleaning report.
    """
    cleaning_report = {
        "operations": [],
        "issues_found": [],
        "columns_transformed": [],
        "rows_before": len(df),
        "columns_before": len(df.columns)
    }
    
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Clean column names
    old_cols = df_cleaned.columns.tolist()
    df_cleaned.columns = [clean_column_name(col) for col in df_cleaned.columns]
    if old_cols != df_cleaned.columns.tolist():
        cleaning_report["operations"].append("Standardized column names")
        cleaning_report["columns_transformed"].extend([
            f"{old} → {new}" for old, new in zip(old_cols, df_cleaned.columns) 
            if old != new
        ])
    
    # Remove duplicates
    dups = df_cleaned.duplicated().sum()
    if dups > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        cleaning_report["operations"].append(f"Removed {dups} duplicate rows")
        cleaning_report["issues_found"].append(f"Found {dups} duplicate rows")
    
    # Check for and handle missing values
    missing = df_cleaned.isnull().sum()
    cols_with_missing = missing[missing > 0].index.tolist()
    if cols_with_missing:
        cleaning_report["issues_found"].append(
            f"Found missing values in {len(cols_with_missing)} columns: {', '.join(cols_with_missing)}"
        )
        
        for col in df_cleaned.columns:
            missing_count = df_cleaned[col].isnull().sum()
            if missing_count == 0:
                continue
                
            # Decide how to handle based on data type and missing percentage
            missing_pct = missing_count / len(df_cleaned)
            
            # If too many missing values, consider dropping
            if missing_pct > 0.5:
                cleaning_report["operations"].append(f"Column '{col}' has {missing_pct:.1%} missing values - marked for review")
                continue
                
            # For numerical columns
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                # Impute missing values with median for numerical
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
                cleaning_report["operations"].append(f"Filled {missing_count} missing values in '{col}' with median ({median_val})")
            
            # For categorical/text columns
            else:
                # Replace with "Unknown" or most frequent for categorical
                most_common = df_cleaned[col].mode()[0]
                fill_value = "Unknown" if isinstance(most_common, str) else most_common
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                cleaning_report["operations"].append(f"Filled {missing_count} missing values in '{col}' with '{fill_value}'")
    
    # Handle outliers in numerical columns
    for col in df_cleaned.select_dtypes(include=['number']).columns:
        # Use IQR method to detect outliers
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
        if outliers > 0:
            cleaning_report["issues_found"].append(f"Found {outliers} outliers in column '{col}'")
            cleaning_report["operations"].append(f"Identified outliers in '{col}' (no automatic removal)")
    
    # Try to convert data types appropriately
    for col in df_cleaned.columns:
        # Skip columns that are already numerical
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            continue
            
        # Try to convert string columns to datetime if they look like dates
        if df_cleaned[col].dtype == object:
            # Check if column might contain dates
            sample = df_cleaned[col].dropna().iloc[:5].astype(str).tolist()
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}'   # DD-MM-YYYY
            ]
            
            could_be_date = any(any(re.search(pattern, str(val)) for val in sample) for pattern in date_patterns)
            
            if could_be_date:
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    non_converted = df_cleaned[col].isna().sum() - missing.get(col, 0)
                    if non_converted / len(df_cleaned) < 0.3:  # If less than 30% failed conversion
                        cleaning_report["operations"].append(f"Converted '{col}' to datetime format")
                        cleaning_report["columns_transformed"].append(f"{col}: string → datetime")
                    else:
                        # Revert if too many failed conversions
                        df_cleaned[col] = df[col]
                except:
                    pass
                    
            # Try to convert to numeric if appropriate
            elif all(is_convertible_to_numeric(val) for val in sample if val is not None and str(val).strip()):
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    non_converted = df_cleaned[col].isna().sum() - missing.get(col, 0)
                    if non_converted / len(df_cleaned) < 0.3:  # If less than 30% failed conversion
                        cleaning_report["operations"].append(f"Converted '{col}' to numeric format")
                        cleaning_report["columns_transformed"].append(f"{col}: string → numeric")
                    else:
                        # Revert if too many failed conversions
                        df_cleaned[col] = df[col]
                except:
                    pass
    
    # Record final stats
    cleaning_report["rows_after"] = len(df_cleaned)
    cleaning_report["columns_after"] = len(df_cleaned.columns)
    
    return df_cleaned, cleaning_report

def infer_schema(df: pd.DataFrame) -> Dict:
    """Infer and return the schema of the dataframe."""
    schema = {}
    for col in df.columns:
        col_info = {
            "data_type": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique())
        }
        
        # Add additional info based on data type
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
            })
        elif pd.api.types.is_datetime64_dtype(df[col]):
            col_info.update({
                "min_date": df[col].min().strftime('%Y-%m-%d') if not pd.isna(df[col].min()) else None,
                "max_date": df[col].max().strftime('%Y-%m-%d') if not pd.isna(df[col].max()) else None
            })
        else:
            # For categorical columns, get top values
            if df[col].nunique() < 10:  # Only if small number of unique values
                top_values = df[col].value_counts().head(5).to_dict()
                # Convert any numpy integers to Python integers
                col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                
        schema[col] = col_info
        
    return schema

def clean_column_name(col_name: str) -> str:
    """Clean and standardize column names."""
    # Convert to lowercase
    col_name = str(col_name).lower()
    
    # Replace spaces and special chars with underscore
    col_name = re.sub(r'[^a-z0-9]', '_', col_name)
    
    # Remove multiple consecutive underscores
    col_name = re.sub(r'_+', '_', col_name)
    
    # Remove leading/trailing underscores
    col_name = col_name.strip('_')
    
    return col_name

def is_convertible_to_numeric(val) -> bool:
    """Check if a value is convertible to numeric."""
    if val is None or pd.isna(val) or (isinstance(val, str) and not val.strip()):
        return True
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def merge_dataframes(dfs: List[pd.DataFrame], common_keys: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Attempt to merge multiple dataframes intelligently.
    Returns the merged dataframe and a report.
    """
    if len(dfs) == 1:
        return dfs[0], {"status": "success", "message": "Only one dataframe provided, no merging needed"}
    
    merge_report = {
        "status": "unknown",
        "message": "",
        "details": []
    }
    
    # If common keys not provided, try to find common columns
    if not common_keys:
        # Get all column names from each dataframe
        all_columns = [set(df.columns) for df in dfs]
        common_columns = set.intersection(*all_columns)
        
        if not common_columns:
            merge_report["status"] = "error"
            merge_report["message"] = "No common columns found between dataframes"
            # Try to concatenate if columns are similar
            similar_structure = len(set(len(df.columns) for df in dfs)) == 1
            if similar_structure:
                try:
                    result = pd.concat(dfs, ignore_index=True)
                    merge_report["status"] = "success" 
                    merge_report["message"] = "Dataframes concatenated vertically (no common keys found but similar structure)"
                    return result, merge_report
                except Exception as e:
                    merge_report["message"] += f". Failed to concatenate: {str(e)}"
                    return dfs[0], merge_report
            return dfs[0], merge_report
        
        # Try to find good candidate keys
        candidate_keys = []
        for col in common_columns:
            # Check uniqueness in each dataframe
            uniqueness_scores = [df[col].nunique() / len(df) for df in dfs]
            avg_uniqueness = sum(uniqueness_scores) / len(uniqueness_scores)
            
            # Good candidates have high uniqueness
            if avg_uniqueness > 0.8:
                candidate_keys.append(col)
        
        # If no good candidates, use all common columns
        common_keys = candidate_keys if candidate_keys else list(common_columns)
        
        merge_report["details"].append(f"Automatically selected merge keys: {', '.join(common_keys)}")
    
    # Perform the merge
    try:
        result = dfs[0].copy()
        for i, df in enumerate(dfs[1:], 2):
            result = result.merge(df, on=common_keys, how='outer', suffixes=(f'', f'_{i}'))
        
        merge_report["status"] = "success"
        merge_report["message"] = f"Successfully merged {len(dfs)} dataframes on keys: {', '.join(common_keys)}"
        
        # Analyze merged results
        rows_before = sum(len(df) for df in dfs)
        cols_before = sum(len(df.columns) for df in dfs)
        merge_report["details"].append(f"Rows before: {rows_before}, after: {len(result)}")
        merge_report["details"].append(f"Columns before: {cols_before}, after: {len(result.columns)}")
        
        return result, merge_report
        
    except Exception as e:
        merge_report["status"] = "error"
        merge_report["message"] = f"Error merging dataframes: {str(e)}"
        return dfs[0], merge_report

# === Insights Generation ===
def generate_data_insights(df: pd.DataFrame) -> Dict:
    """Generate comprehensive insights from the dataframe."""
    insights = {
        "basic_info": {
            "num_rows": int(len(df)),
            "num_columns": int(len(df.columns)),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
        },
        "column_types": {},
        "numerical_stats": {},
        "categorical_stats": {},
        "missing_values": {},
        "correlation_info": {},
        "distribution_info": {},
        "time_series_info": {},
        "advanced_insights": []
    }
    
    # Column types breakdown
    dtype_counts = df.dtypes.value_counts().to_dict()
    insights["column_types"] = {str(k): int(v) for k, v in dtype_counts.items()}
    
    # Missing values analysis
    missing = df.isnull().sum()
    insights["missing_values"] = {
        "total_missing_cells": int(missing.sum()),
        "missing_percentage": f"{missing.sum() / (len(df) * len(df.columns)) * 100:.2f}%",
        "columns_with_missing": {col: int(count) for col, count in missing[missing > 0].items()}
    }
    
    # Numerical column statistics
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        # Get basic statistics
        desc = df[numerical_cols].describe().to_dict()
        
        # Convert all numeric values (including numpy types) to Python native types
        for col, stats in desc.items():
            insights["numerical_stats"][col] = {
                stat: float(val) if isinstance(val, (float, int, np.int64, np.float64)) else val
                for stat, val in stats.items()
            }
        
        # Add skewness and kurtosis
        for col in numerical_cols:
            if col in insights["numerical_stats"]:
                insights["numerical_stats"][col]["skewness"] = float(df[col].skew())
                insights["numerical_stats"][col]["kurtosis"] = float(df[col].kurtosis())
    
    # Categorical column analysis
    categorical_cols = df.select_dtypes(exclude=['number', 'datetime']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(5).to_dict()
            insights["categorical_stats"][col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in value_counts.items()}
            }
    
    # Correlation analysis for numerical data
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr().round(2)
        
        # Convert corr_matrix to a serializable dictionary
        corr_dict = {}
        for col1 in corr_matrix.columns:
            corr_dict[col1] = {}
            for col2 in corr_matrix.columns:
                corr_dict[col1][col2] = float(corr_matrix.loc[col1, col2])
        
        # Find high correlations (absolute value > 0.7)
        high_corrs = []
        for col1 in numerical_cols:
            for col2 in numerical_cols:
                if col1 < col2:  # Only check each pair once
                    corr = float(df[col1].corr(df[col2]))
                    if abs(corr) > 0.7:
                        high_corrs.append({
                            "column1": str(col1),
                            "column2": str(col2),
                            "correlation": float(corr),
                            "relationship": "strong positive" if corr > 0 else "strong negative"
                        })
        
        insights["correlation_info"] = {
            "correlation_matrix": corr_dict,
            "high_correlations": high_corrs
        }
    
    # Check for datetime columns for time series analysis
    datetime_cols = df.select_dtypes(include=['datetime']).columns
    if len(datetime_cols) > 0:
        for col in datetime_cols:
            # Sort data by date first
            date_sorted = df.sort_values(by=col)
            date_col = date_sorted[col]
            
            # Time range analysis
            if not pd.isna(date_col.max()) and not pd.isna(date_col.min()):
                date_range = date_col.max() - date_col.min()
                days = date_range.days
                
                # Check if we have regular intervals
                if len(date_col) > 1:
                    # Calculate the differences between consecutive dates
                    date_diffs = date_col.diff().dropna()
                    
                    # Count frequency of each difference
                    freq_counts = date_diffs.value_counts().head(3).to_dict()
                    most_common_diff = max(freq_counts.items(), key=lambda x: x[1])[0]
                    
                    # Convert to readable format
                    if isinstance(most_common_diff, pd.Timedelta):
                        if most_common_diff.days >= 1:
                            freq_text = f"{most_common_diff.days} days"
                        elif most_common_diff.seconds // 3600 >= 1:
                            freq_text = f"{most_common_diff.seconds // 3600} hours"
                        else:
                            freq_text = f"{most_common_diff}"
                    else:
                        freq_text = str(most_common_diff)
                    
                    insights["time_series_info"][col] = {
                        "start_date": date_col.min().strftime('%Y-%m-%d'),
                        "end_date": date_col.max().strftime('%Y-%m-%d'),
                        "time_span": f"{days} days",
                        "most_common_interval": freq_text
                    }
    
    # Advanced insights - patterns and suggestions
    insights["advanced_insights"] = []
    
    # Check for highly imbalanced categorical columns
    for col in categorical_cols:
        if df[col].nunique() < 10:  # Only for columns with few categories
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.max() > 0.9:  # If one category > 90%
                top_cat = value_counts.idxmax()
                insights["advanced_insights"].append({
                    "type": "imbalance_warning",
                    "column": str(col),
                    "description": f"Column '{col}' is highly imbalanced - '{top_cat}' represents {value_counts.max()*100:.1f}% of values"
                })
    
    # Look for columns with suspiciously low variance
    for col in numerical_cols:
        if df[col].nunique() <= 2 and len(df) > 10:
            insights["advanced_insights"].append({
                "type": "low_variance",
                "column": str(col),
                "description": f"Column '{col}' has very low variance (only {df[col].nunique()} unique values)"
            })
    
    # Highlight columns with high cardinality that might be keys/IDs
    for col in df.columns:
        if df[col].nunique() == len(df) and len(df) > 10:
            insights["advanced_insights"].append({
                "type": "possible_key",
                "column": str(col),
                "description": f"Column '{col}' has unique values for every row - likely a primary key or ID"
            })
    
    # Check for potential duplicate columns (high correlation or same values)
    if len(numerical_cols) > 1:
        for col1 in numerical_cols:
            for col2 in numerical_cols:
                if col1 < col2:  # Avoid checking pairs twice
                    if df[col1].equals(df[col2]):
                        insights["advanced_insights"].append({
                            "type": "duplicate_columns",
                            "columns": [str(col1), str(col2)],
                            "description": f"Columns '{col1}' and '{col2}' have identical values"
                        })
    
    return insights

def recommend_visualizations(df: pd.DataFrame) -> List[Dict]:
    """
    Analyze the dataframe and recommend appropriate visualizations.
    """
    recommendations = []
    
    # Count data types in the dataframe
    num_cols = len(df.select_dtypes(include=['number']).columns)
    cat_cols = len(df.select_dtypes(include=['object', 'category']).columns)
    datetime_cols = len(df.select_dtypes(include=['datetime']).columns)
    
    # Get specific column names by type
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols_list = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Low-cardinality categorical columns (good for charts)
    low_card_cols = [col for col in categorical_cols 
                   if df[col].nunique() <= 10 and df[col].nunique() > 1]
    
    # For categorical columns, recommend bar charts
    if cat_cols > 0:
        for col in low_card_cols[:5]:  # Only reasonably sized categories
            recommendations.append({
                "id": f"bar_{col}",
                "title": f"Frequency of {col} categories",
                "type": "bar", 
                "description": f"Bar chart showing frequency of each category in {col}",
                "columns": [col],
                "parameters": {"sort": True}
            })
    
    # Relationships between variables
    if num_cols >= 2:
        # Correlation heatmap
        recommendations.append({
            "id": "correlation_heatmap",
            "title": "Correlation Heatmap",
            "type": "heatmap",
            "description": "Heatmap showing correlations between numerical variables",
            "columns": numerical_cols,
            "parameters": {"annot": True}
        })
        
        # Recommend scatter plots for potentially related variables
        if len(numerical_cols) >= 2:
            # Create scatter plot recommendations for first few pairs
            for i, col1 in enumerate(numerical_cols[:3]):
                for col2 in numerical_cols[i+1:i+3]:  # Limit pairs to avoid too many recommendations
                    recommendations.append({
                        "id": f"scatter_{col1}_{col2}",
                        "title": f"Relationship: {col1} vs {col2}",
                        "type": "scatter",
                        "description": f"Scatter plot exploring relationship between {col1} and {col2}",
                        "columns": [col1, col2],
                        "parameters": {}
                    })
    
    # If we have both numerical and categorical data
    if num_cols > 0 and cat_cols > 0:
        # Boxplots for numerical vs categorical
        for num_col in numerical_cols[:2]:
            for cat_col in low_card_cols[:2]:
                recommendations.append({
                    "id": f"box_{num_col}_by_{cat_col}",
                    "title": f"{num_col} grouped by {cat_col}",
                    "type": "boxplot",
                    "description": f"Boxplot showing distribution of {num_col} for each {cat_col} category",
                    "columns": [num_col, cat_col],
                    "parameters": {}
                })
    
    # Time series visualizations
    if datetime_cols > 0:
        date_col = datetime_cols_list[0]  # Use first datetime column
        if num_cols > 0:
            for num_col in numerical_cols[:3]:
                recommendations.append({
                    "id": f"time_{date_col}_{num_col}",
                    "title": f"{num_col} over time",
                    "type": "line",
                    "description": f"Line chart showing {num_col} trends over time",
                    "columns": [date_col, num_col],
                    "parameters": {"x": date_col, "y": num_col}
                })
    
    # If enough numeric columns for PCA or clustering
    if num_cols >= 3:
        recommendations.append({
            "id": "pca_plot",
            "title": "Principal Component Analysis",
            "type": "pca",
            "description": "PCA visualization to show data structure in lower dimensions",
            "columns": numerical_cols,
            "parameters": {"n_components": 2}
        })
    
    return recommendations

def generate_visualization(df: pd.DataFrame, viz_type: str, columns: List[str], 
                         parameters: Dict = None) -> Dict:
    """
    Generate a visualization based on the specified type and parameters.
    Returns a dictionary with the plot data.
    """
    if parameters is None:
        parameters = {}
    
    result = {
        "success": False,
        "error": None,
        "plot_type": viz_type,
        "plot_data": None
    }
    
    try:
        # Handle missing data for visualization
        viz_df = df[columns].copy()
        
        # Basic visualizations
        if viz_type == "bar":
            column = columns[0]
            counts = viz_df[column].value_counts()
            
            if parameters.get("sort", False):
                counts = counts.sort_values(ascending=False)
                
            fig = go.Figure([go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                text=counts.values,
                textposition='auto'
            )])
            
            fig.update_layout(
                title=parameters.get("title", f"Frequency of {column} Categories"),
                xaxis_title=column,
                yaxis_title="Count"
            )
            
            result["plot_data"] = fig.to_json()
            result["success"] = True
            
        elif viz_type == "heatmap":
            # Create correlation matrix for numerical columns
            corr_matrix = viz_df.corr().round(2)
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=parameters.get("title", "Correlation Heatmap"),
                height=600,
                width=700
            )
            
            result["plot_data"] = fig.to_json()
            result["success"] = True
            
        elif viz_type == "scatter":
            if len(columns) >= 2:
                x_col, y_col = columns[0], columns[1]
                color_col = columns[2] if len(columns) > 2 else None
                
                if color_col:
                    fig = px.scatter(
                        viz_df, x=x_col, y=y_col, color=color_col,
                        title=parameters.get("title", f"{y_col} vs {x_col} by {color_col}")
                    )
                else:
                    fig = px.scatter(
                        viz_df, x=x_col, y=y_col,
                        title=parameters.get("title", f"{y_col} vs {x_col}")
                    )
                
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col
                )
                
                result["plot_data"] = fig.to_json()
                result["success"] = True
                
        elif viz_type == "boxplot":
            if len(columns) >= 2:
                num_col, cat_col = columns[0], columns[1]
                
                fig = px.box(
                    viz_df, 
                    x=cat_col, 
                    y=num_col,
                    title=parameters.get("title", f"{num_col} by {cat_col}")
                )
                
                fig.update_layout(xaxis_title=cat_col,
                    yaxis_title=num_col
                )
                
                result["plot_data"] = fig.to_json()
                result["success"] = True
                
        elif viz_type == "line":
            if len(columns) >= 2:
                x_col, y_col = columns[0], columns[1]
                
                # Sort by x (typically time) column
                viz_df = viz_df.sort_values(by=x_col)
                
                fig = px.line(
                    viz_df, 
                    x=x_col, 
                    y=y_col,
                    title=parameters.get("title", f"{y_col} over {x_col}")
                )
                
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col
                )
                
                result["plot_data"] = fig.to_json()
                result["success"] = True
                
        elif viz_type == "pca":
            # Perform PCA on numerical columns
            numeric_df = viz_df.select_dtypes(include=['number'])
            
            if len(numeric_df.columns) < 2:
                result["error"] = "Not enough numerical columns for PCA"
                return result
                
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            imputed_data = imputer.fit_transform(numeric_df)
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(imputed_data)
            
            # Apply PCA
            n_components = min(parameters.get("n_components", 2), len(numeric_df.columns))
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            
            # Create a DataFrame with PCA results
            pca_df = pd.DataFrame(
                data=pca_result,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
            # Add color column if available
            color_col = parameters.get("color_column", None)
            if color_col and color_col in df.columns:
                pca_df[color_col] = df[color_col].values
                
                fig = px.scatter(
                    pca_df, x='PC1', y='PC2', color=color_col,
                    title="PCA Visualization"
                )
            else:
                fig = px.scatter(
                    pca_df, x='PC1', y='PC2',
                    title="PCA Visualization"
                )
                
            # Add variance explained info
            explained_var = pca.explained_variance_ratio_
            fig.update_layout(
                xaxis_title=f"PC1 ({explained_var[0]:.2%} variance)",
                yaxis_title=f"PC2 ({explained_var[1]:.2%} variance)"
            )
            
            result["plot_data"] = fig.to_json()
            result["success"] = True
            
        else:
            result["error"] = f"Visualization type '{viz_type}' not supported"
            
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        
    return result

# === LLM Integration with OpenRouter API ===
def query_together_ai(prompt: str, context: str = "", model: str = LLM_MODEL) -> str:
    """
    Call OpenRouter API with Meta's Llama 4 Maverick model.
    This function supports both text and image inputs (though we're only using text for now).
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://data-explorer-app.com",  # Replace with your actual site if available
        "X-Title": "AI Data Explorer"
    }
    
    # Create a context-enhanced prompt
    if context:
        full_content = f"""<data_context>
{context}
</data_context>

User question: {prompt}

Please analyze the data context provided and answer the question thoroughly. 
Include specific numbers, statistics and insights from the data when relevant.
Explain relationships between variables and provide actionable recommendations when possible.
"""
    else:
        full_content = prompt
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an advanced data analysis assistant with expertise in statistics, machine learning, and business intelligence. You help users understand their datasets by providing clear, concise answers based on the data provided. Include relevant statistics, trends, patterns, correlations, and data-driven recommendations. Always prioritize accuracy and actionable insights."
            },
            {
                "role": "user",
                "content": full_content
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise exception for HTTP errors
        
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response generated")
    except requests.exceptions.RequestException as e:
        return f"Error calling AI service: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def summarize_dataset(df: pd.DataFrame, include_sample: bool = False) -> str:
    """Generate a comprehensive dataset summary using LLM."""
    # Create a detailed context about the dataset
    dataset_info = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "column_info": {}
    }
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        col_info = {
            "data_type": col_type,
            "missing_values": int(df[col].isna().sum()),
            "unique_values": int(df[col].nunique())
        }
        
        # Add type-specific information
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "std_dev": float(df[col].std()) if not pd.isna(df[col].std()) else None
            })
        elif pd.api.types.is_datetime64_dtype(df[col]):
            if not pd.isna(df[col].min()) and not pd.isna(df[col].max()):
                col_info.update({
                    "min_date": str(df[col].min()),
                    "max_date": str(df[col].max())
                })
        else:
            # For categorical data, include top categories
            if df[col].nunique() < 20:  # Only if reasonably small number of categories
                top_cats = df[col].value_counts().head(5).to_dict()
                col_info["top_categories"] = {str(k): int(v) for k, v in top_cats.items()}
        
        dataset_info["column_info"][str(col)] = col_info
    
    # Include correlation information for numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr().round(2).to_dict()
        # Convert numpy values to Python native types
        serialized_corr = {}
        for col1, values in corr_matrix.items():
            serialized_corr[str(col1)] = {str(col2): float(val) for col2, val in values.items()}
        dataset_info["correlations"] = serialized_corr
    
    # Add sample data if requested (first 5 rows)
    if include_sample:
        sample_data = df.head(5).to_dict(orient='records')
        # Convert to serializable format
        serialized_sample = []
        for row in sample_data:
            serialized_row = {}
            for k, v in row.items():
                if isinstance(v, (np.int64, np.int32, np.int8)):
                    serialized_row[str(k)] = int(v)
                elif isinstance(v, (np.float64, np.float32)):
                    serialized_row[str(k)] = float(v)
                else:
                    serialized_row[str(k)] = v
            serialized_sample.append(serialized_row)
        dataset_info["sample_data"] = serialized_sample
    
    # Convert to string for LLM input
    dataset_context = json.dumps(dataset_info, indent=2)
    
    # Create the prompt
    prompt = """
    Please provide a comprehensive summary of this dataset. Include:
    1. A brief overview of what the dataset contains
    2. Key insights about the structure, completeness, and quality
    3. Notable patterns, relationships, or anomalies
    4. Potential use cases or analyses that could be performed
    5. Any recommendations for further exploration
    
    Be concise but thorough, highlighting the most important aspects of the data.
    """
    
    # Call LLM with the dataset context
    summary = query_together_ai(prompt, dataset_context)
    
    return summary

def generate_advanced_insights(df: pd.DataFrame) -> Dict:
    """Use LLM to generate more advanced insights beyond basic statistics."""
    # Create a comprehensive context with statistics and columns information
    stats = df.describe().to_dict()
    # Convert np types to Python native types
    serializable_stats = {}
    for col, col_stats in stats.items():
        serializable_stats[str(col)] = {
            str(stat): float(val) if isinstance(val, (np.float64, np.int64)) else val
            for stat, val in col_stats.items()
        }
    
    # Add column types information
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[str(col)] = "numeric"
        elif pd.api.types.is_datetime64_dtype(df[col]):
            column_types[str(col)] = "datetime"
        else:
            column_types[str(col)] = "categorical"
    
    # Numerical correlations
    numeric_cols = df.select_dtypes(include=['number']).columns
    correlations = None
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().round(2).to_dict()
        # Convert numpy values to Python native types
        serialized_corr = {}
        for col1, values in corr_matrix.items():
            serialized_corr[str(col1)] = {str(col2): float(val) for col2, val in values.items()}
        correlations = serialized_corr
    
    # Sample data (first few rows)
    sample_rows = df.head(3).to_dict(orient='records')
    # Convert to serializable format
    serialized_sample = []
    for row in sample_rows:
        serialized_row = {}
        for k, v in row.items():
            if isinstance(v, (np.int64, np.int32, np.int8)):
                serialized_row[str(k)] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                serialized_row[str(k)] = float(v)
            else:
                serialized_row[str(k)] = v
        serialized_sample.append(serialized_row)
    
    # Create context
    context = {
        "dataset_size": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "column_types": column_types,
        "statistics": serializable_stats,
        "correlations": correlations,
        "sample_data": serialized_sample
    }
    
    context_str = json.dumps(context, indent=2)
    
    # Create prompt for advanced insights
    prompt = """
    Based on the dataset information provided, please generate 5-7 advanced insights that go beyond basic statistics.
    Consider:
    
    1. Unusual patterns or anomalies in the data
    2. Interesting relationships between variables
    3. Potential business implications
    4. Areas that warrant further investigation
    5. Possible data quality issues or biases
    
    Format your response as a JSON list of insight objects, each with:
    - A short title
    - A detailed description
    - The confidence level (high/medium/low)
    - Related columns
    
    For example:
    [
        {
            "title": "Strong Sales-Weather Correlation",
            "description": "There appears to be a strong positive correlation (0.78) between temperature and sales volume, suggesting seasonal effects on purchasing behavior.",
            "confidence": "high",
            "related_columns": ["temperature", "sales_volume"]
        },
        ...
    ]
    """
    
    # Call LLM
    response = query_together_ai(prompt, context_str)
    
    # Parse response
    try:
        # Try to extract JSON from response if it's embedded in text
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].strip()
            
        insights = json.loads(json_str)
        return {"success": True, "insights": insights}
    except Exception as e:
        # Fall back to returning the raw text
        return {"success": False, "error": str(e), "raw_response": response}

def suggest_next_steps(df: pd.DataFrame, existing_insights: Dict) -> List[Dict]:
    """Suggest next analytical steps based on dataset and current insights."""
    # Create context with dataset properties and existing insights
    context = {
        "dataset_properties": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_types": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {str(col): int(val) for col, val in df.isnull().sum().to_dict().items()},
        },
        "existing_insights": existing_insights
    }
    
    context_str = json.dumps(json_serializable(context), indent=2)
    
    # Create prompt for next steps
    prompt = """
    Based on the dataset properties and existing insights, suggest 3-5 next analytical steps.
    Each suggestion should:
    
    1. Address a specific question or hypothesis
    2. Recommend specific techniques or approaches
    3. Explain the potential value or insight to be gained
    
    Format your response as a JSON list of suggestion objects, each with:
    - A title
    - A detailed description of the analytical approach
    - Required data columns
    - Expected outcome
    """
    
    # Call LLM
    response = query_together_ai(prompt, context_str)
    
    # Parse response
    try:
        # Try to extract JSON from response if it's embedded in text
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].strip()
            
        suggestions = json.loads(json_str)
        return suggestions
    except:
        # Create a structured response manually if JSON parsing fails
        fallback_suggestions = [
            {
                "title": "Further Analysis Needed",
                "description": "The AI provided suggestions but in an unstructured format. Please see the raw response.",
                "raw_response": response
            }
        ]
        return fallback_suggestions

# === PDF Report Generation ===
def generate_pdf_report(df: pd.DataFrame, insights: Dict, visualizations: List[Dict], summary: str) -> str:
    """Generate a comprehensive PDF report with data insights and visualizations."""
    try:
        from fpdf import FPDF
        import matplotlib.pyplot as plt
        
        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add title page
        pdf.add_page()
        pdf.set_font("Arial", 'B', 24)
        pdf.cell(200, 10, "Data Analysis Report", 0, 1, 'C')
        pdf.ln(10)

        # Current date
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 10, f"Generated on {current_date}", 0, 1, 'C')
        pdf.ln(5)
        
        # Add report description
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 5, "This report was automatically generated by AI Data Explorer, providing a comprehensive analysis of your dataset. It includes data quality assessment, key insights, statistical analysis, and actionable recommendations.")
        
        # Table of contents
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Table of Contents", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        sections = ["1. Dataset Overview", "2. Executive Summary", "3. Data Quality Assessment", 
                    "4. Key Insights", "5. Statistical Analysis", "6. Advanced Insights", "7. Recommendations"]
        for section in sections:
            pdf.cell(0, 8, section, 0, 1, 'L')
        
        # Dataset overview
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "1. Dataset Overview", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        
        # Safely access values from insights dict
        basic_info = insights.get('basic_info', {})
        num_rows = basic_info.get('num_rows', 0)
        num_columns = basic_info.get('num_columns', 0)
        
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "General Information", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 5, f"Rows: {num_rows:,}")
        pdf.multi_cell(0, 5, f"Columns: {num_columns:,}")
        memory_usage = basic_info.get('memory_usage', 'Unknown')
        pdf.multi_cell(0, 5, f"Memory Usage: {memory_usage}")
        
        # Add column types breakdown
        col_types = insights.get('column_types', {})
        if col_types:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Column Types", 0, 1, 'L')
            pdf.set_font("Arial", '', 11)
            for col_type, count in col_types.items():
                pdf.multi_cell(0, 5, f"- {col_type}: {count} columns")

        
        # Sample data preview
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Data Preview", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        
        # Create a sample table with first 5 rows and 5 columns
        sample_cols = list(df.columns)[:min(5, len(df.columns))]
        sample_df = df[sample_cols].head(5)
        
        # Column headers
        pdf.set_font("Arial", 'B', 9)
        col_width = 180 / len(sample_cols)
        for col in sample_cols:
            pdf.cell(col_width, 7, str(col)[:15], 1, 0, 'C')
        pdf.ln()
        
        # Row data
        pdf.set_font("Arial", '', 9)
        for i, row in sample_df.iterrows():
            for col in sample_cols:
                cell_value = str(row[col])[:15]  # Truncate long values
                pdf.cell(col_width, 7, cell_value, 1, 0, 'L')
            pdf.ln()
            
        # Add note about truncated data
        pdf.set_font("Arial", 'I', 8)
        pdf.ln(5)
        pdf.multi_cell(0, 4, "Note: This is just a preview. Some values may be truncated for display purposes.")
        
        # Add summary
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "2. Executive Summary", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 5, summary)
        
        # Data quality assessment
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "3. Data Quality Assessment", 0, 1, 'L')
        
        # Missing values
        missing_vals = insights.get('missing_values', {})
        missing_pct = missing_vals.get('missing_percentage', '0%')
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Data Completeness: {missing_pct} missing", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        
        cols_with_missing = missing_vals.get('columns_with_missing', {})
        if cols_with_missing:
            pdf.multi_cell(0, 5, f"Total missing cells: {missing_vals.get('total_missing_cells', 0):,}")
            pdf.ln(3)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "Columns with Missing Values:", 0, 1, 'L')
            pdf.set_font("Arial", '', 11)
            
            # Display missing values as a table
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(90, 7, "Column Name", 1, 0, 'C')
            pdf.cell(40, 7, "Missing Count", 1, 0, 'C')
            pdf.cell(40, 7, "Missing %", 1, 1, 'C')
            
            pdf.set_font("Arial", '', 10)
            for col, count in list(cols_with_missing.items())[:10]:  # Show top 10 columns with missing values
                pct = round(count / num_rows * 100, 2)
                pdf.cell(90, 7, str(col)[:30], 1, 0, 'L')
                pdf.cell(40, 7, f"{count:,}", 1, 0, 'R')
                pdf.cell(40, 7, f"{pct}%", 1, 1, 'R')
                
            if len(cols_with_missing) > 10:
                pdf.set_font("Arial", 'I', 9)
                pdf.multi_cell(0, 5, f"... and {len(cols_with_missing) - 10} more columns with missing values")
        else:
            pdf.multi_cell(0, 5, "No missing values detected in the dataset.")
        
        # Key insights
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "4. Key Insights", 0, 1, 'L')
        
        # Add correlation insights
        correlation_info = insights.get('correlation_info', {})
        high_corrs = correlation_info.get('high_correlations', [])
        if high_corrs:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Key Correlations", 0, 1, 'L')
            pdf.set_font("Arial", '', 11)
            
            for i, corr in enumerate(high_corrs[:7]):  # Show top 7 correlations
                if isinstance(corr, dict) and 'column1' in corr and 'column2' in corr and 'correlation' in corr:
                    corr_val = corr['correlation']
                    relationship = corr.get('relationship', 'strong correlation')
                    pdf.multi_cell(0, 5, f"{i+1}. {corr['column1']} and {corr['column2']} have a {relationship} ({corr_val:.2f})")
                    pdf.ln(2)
        
        # Statistical analysis
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "5. Statistical Analysis", 0, 1, 'L')
        
        # Numerical statistics
        numerical_stats = insights.get('numerical_stats', {})
        if numerical_stats:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Numerical Column Statistics", 0, 1, 'L')
            pdf.set_font("Arial", '', 11)
            
            # Create a table for each numerical column (limited to 5)
            for i, (col, stats) in enumerate(list(numerical_stats.items())[:5]):
                if i > 0:
                    pdf.ln(5)
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(0, 8, f"Statistics for '{col}'", 0, 1, 'L')
                pdf.set_font("Arial", '', 10)
                
                # Create a table for statistics
                pdf.cell(60, 7, "Measure", 1, 0, 'C')
                pdf.cell(40, 7, "Value", 1, 1, 'C')
                
                # Add key statistics
                stat_names = {"mean": "Mean", "std": "Std Deviation", "min": "Minimum", 
                              "25%": "25th Percentile", "50%": "Median", 
                              "75%": "75th Percentile", "max": "Maximum"}
                
                for stat, label in stat_names.items():
                    if stat in stats:
                        pdf.cell(60, 7, label, 1, 0, 'L')
                        pdf.cell(40, 7, f"{stats[stat]:.2f}", 1, 1, 'R')
                
                # Add skewness if available
                if "skewness" in stats:
                    pdf.cell(60, 7, "Skewness", 1, 0, 'L')
                    skew = stats["skewness"]
                    skew_desc = ""
                    if abs(skew) < 0.5:
                        skew_desc = " (Approximately Symmetric)"
                    elif skew > 0:
                        skew_desc = " (Right Skewed)"
                    else:
                        skew_desc = " (Left Skewed)"
                    pdf.cell(40, 7, f"{skew:.2f}{skew_desc}", 1, 1, 'R')
            
            if len(numerical_stats) > 5:
                pdf.set_font("Arial", 'I', 9)
                pdf.ln(5)
                pdf.multi_cell(0, 5, f"Note: Statistics for {len(numerical_stats) - 5} more numerical columns are available in the interactive dashboard.")
        
        # Categorical data
        categorical_stats = insights.get('categorical_stats', {})
        if categorical_stats:
            pdf.ln(8)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Categorical Column Analysis", 0, 1, 'L')
            pdf.set_font("Arial", '', 11)
            
            # For each categorical column (limited to 3)
            for i, (col, stats) in enumerate(list(categorical_stats.items())[:3]):
                if i > 0:
                    pdf.ln(5)
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(0, 8, f"Distribution of '{col}'", 0, 1, 'L')
                pdf.set_font("Arial", '', 10)
                
                # Show unique value count
                pdf.multi_cell(0, 5, f"Total unique values: {stats.get('unique_values', 'N/A')}")
                
                # Show top values in a table
                top_values = stats.get('top_values', {})
                if top_values:
                    pdf.ln(3)
                    pdf.set_font("Arial", 'B', 10)
                    pdf.cell(90, 7, "Category", 1, 0, 'C')
                    pdf.cell(40, 7, "Count", 1, 1, 'C')
                    
                    pdf.set_font("Arial", '', 10)
                    for category, count in top_values.items():
                        pdf.cell(90, 7, str(category)[:30], 1, 0, 'L')
                        pdf.cell(40, 7, f"{count:,}", 1, 1, 'R')
        
        # Advanced insights
        advanced_insights = insights.get('advanced_insights', [])
        if advanced_insights:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "6. Advanced Insights", 0, 1, 'L')
            pdf.set_font("Arial", '', 11)
            
            # Display each insight
            for i, insight in enumerate(advanced_insights):
                if isinstance(insight, dict):
                    title = insight.get('title', f'Insight {i+1}')
                    description = insight.get('description', 'No description available')
                    confidence = insight.get('confidence', 'medium').lower()
                    related_columns = insight.get('related_columns', ['N/A'])
                    
                    pdf.ln(5)
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, title, 0, 1, 'L')
                    pdf.set_font("Arial", '', 11)
                    pdf.multi_cell(0, 5, description)
                    
                    pdf.ln(2)
                    confidence_txt = f"Confidence: {confidence}"
                    columns_txt = f"Related columns: {', '.join(related_columns)}"
                    pdf.set_font("Arial", 'I', 10)
                    pdf.multi_cell(0, 5, f"{confidence_txt} | {columns_txt}")
                    pdf.ln(2)
        
        # Recommendations
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "7. Recommendations", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        
        # Generate dynamic recommendations based on dataset analysis
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Data Quality Recommendations", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        
        # Missing data recommendations
        missing_count = missing_vals.get('total_missing_cells', 0)
        if missing_count > 0:
            missing_pct_val = float(missing_pct.strip('%')) if isinstance(missing_pct, str) else 0
            if missing_pct_val > 15:
                pdf.multi_cell(0, 5, "1. Consider data imputation techniques for missing values, as they constitute a significant portion of your dataset.")
            else:
                pdf.multi_cell(0, 5, "1. Address missing values in key columns to improve analysis accuracy.")
        else:
            pdf.multi_cell(0, 5, "1. Your dataset is complete with no missing values - excellent data quality!")
        
        # Outlier recommendations
        if 'advanced_insights' in insights:
            has_outliers = any('outlier' in str(insight.get('description', '')).lower() 
                              for insight in insights['advanced_insights'] if isinstance(insight, dict))
            if has_outliers:
                pdf.multi_cell(0, 5, "2. Investigate outliers identified in numerical columns, which may significantly affect statistical calculations.")
        
        # General analysis recommendations
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Analysis Recommendations", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        
        # Correlation recommendations
        if high_corrs:
            pdf.multi_cell(0, 5, f"1. Explore the strong relationship between {high_corrs[0]['column1']} and {high_corrs[0]['column2']} further, as this may yield valuable business insights.")
        
        # Column-specific recommendations
        numerical_count = len(insights.get('numerical_stats', {}))
        categorical_count = len(insights.get('categorical_stats', {}))
        
        if numerical_count > 3:
            pdf.multi_cell(0, 5, "2. Consider dimensionality reduction techniques like PCA to identify underlying patterns in your numerical variables.")
        
        if categorical_count > 0 and numerical_count > 0:
            pdf.multi_cell(0, 5, "3. Perform group-based analysis to understand how key metrics vary across different categories.")
        
        # Time series recommendation if datetime columns exist
        if insights.get('time_series_info', {}):
            pdf.multi_cell(0, 5, "4. Conduct time series analysis to identify trends, seasonality, and forecasting opportunities.")
        
        # Additional recommendations
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Next Steps", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 5, "1. Use the interactive dashboard to explore visualizations that highlight key patterns in your data.")
        pdf.multi_cell(0, 5, "2. Leverage the chat interface to ask specific questions about your dataset.")
        pdf.multi_cell(0, 5, "3. Consider performing advanced statistical tests to validate key hypotheses.")
        pdf.multi_cell(0, 5, "4. Export insights and visualizations for presentation to stakeholders.")
        
        # Save PDF
        output_path = "data_analysis_report.pdf"
        pdf.output(output_path)
        
        return output_path
        
    except Exception as e:
        return f"Error generating PDF: {str(e)}"