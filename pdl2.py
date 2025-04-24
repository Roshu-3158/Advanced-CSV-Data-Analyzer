import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import io

# Apply custom CSS using markdown
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        h1 {
            color: #2e86de;
        }
        .st-bq {
            border-left: 4px solid #2e86de;
            padding-left: 1rem;
        }
        .stButton>button {
            background-color: #2e86de;
            color: white;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Advanced CSV Data Analyzer")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Cleaning", "Exploratory Analysis", "Advanced Visualizations", "Machine Learning"])

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Initialize session state for storing data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'numeric_cols' not in st.session_state:  # Add numeric_cols to session state
    st.session_state.numeric_cols = None

# If a file is uploaded
if uploaded_file is not None:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.cleaned_df = st.session_state.df.copy()
        # Calculate numeric columns once when file is first uploaded
        st.session_state.numeric_cols = st.session_state.cleaned_df.select_dtypes(include=np.number).columns.tolist()
    
    df = st.session_state.df
    cleaned_df = st.session_state.cleaned_df
    numeric_cols = st.session_state.numeric_cols  # Get from session state

    # Page 1: Data Overview
    if page == "Data Overview":
        st.header("📄 Data Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Preview")
            st.dataframe(cleaned_df.head(10))
        
        with col2:
            st.subheader("Basic Information")
            st.write(f"Shape: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
            
            buffer = io.StringIO()
            cleaned_df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.download_button(
                label="Download Data",
                data=cleaned_df.to_csv(index=False),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        
        st.subheader("Column Statistics")
        selected_cols = st.multiselect("Select columns to analyze", cleaned_df.columns)
        if selected_cols:
            st.write(cleaned_df[selected_cols].describe(include='all'))
        
        st.subheader("Missing Values")
        missing_values = cleaned_df.isnull().sum()
        st.write(missing_values[missing_values > 0])
        
        if st.checkbox("Show Data Types"):
            st.write(cleaned_df.dtypes.astype(str))

    # Page 2: Data Cleaning
    elif page == "Data Cleaning":
        st.header("🧹 Data Cleaning")
        
        st.subheader("Handle Missing Values")
        missing_cols = cleaned_df.columns[cleaned_df.isnull().any()].tolist()
        
        if missing_cols:
            col1, col2 = st.columns(2)
            with col1:
                selected_missing_col = st.selectbox("Select column with missing values", missing_cols)
                missing_count = cleaned_df[selected_missing_col].isnull().sum()
                st.write(f"Missing values: {missing_count}")
                
            with col2:
                action = st.radio(f"Action for {selected_missing_col}",
                                 ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value"])
                
                if action == "Fill with custom value":
                    custom_value = st.text_input("Enter custom value")
                else:
                    custom_value = None
                
                if st.button("Apply"):
                    if action == "Drop rows":
                        cleaned_df.dropna(subset=[selected_missing_col], inplace=True)
                    elif action == "Fill with mean":
                        if is_numeric_dtype(cleaned_df[selected_missing_col]):
                            cleaned_df[selected_missing_col].fillna(cleaned_df[selected_missing_col].mean(), inplace=True)
                        else:
                            st.warning("Cannot fill non-numeric column with mean")
                    elif action == "Fill with median":
                        if is_numeric_dtype(cleaned_df[selected_missing_col]):
                            cleaned_df[selected_missing_col].fillna(cleaned_df[selected_missing_col].median(), inplace=True)
                        else:
                            st.warning("Cannot fill non-numeric column with median")
                    elif action == "Fill with mode":
                        cleaned_df[selected_missing_col].fillna(cleaned_df[selected_missing_col].mode()[0], inplace=True)
                    elif action == "Fill with custom value" and custom_value:
                        try:
                            # Try to convert to float first
                            custom_value = float(custom_value)
                        except ValueError:
                            pass
                        cleaned_df[selected_missing_col].fillna(custom_value, inplace=True)
                    
                    st.success("Operation completed!")
                    st.session_state.cleaned_df = cleaned_df
                    # Update numeric_cols after cleaning
                    st.session_state.numeric_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
        else:
            st.success("No missing values found in the dataset!")
        
        st.subheader("Remove Columns")
        cols_to_drop = st.multiselect("Select columns to remove", cleaned_df.columns)
        if st.button("Remove Selected Columns") and cols_to_drop:
            cleaned_df.drop(cols_to_drop, axis=1, inplace=True)
            st.session_state.cleaned_df = cleaned_df
            # Update numeric_cols after column removal
            st.session_state.numeric_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
            st.success(f"Removed columns: {', '.join(cols_to_drop)}")
        
        st.subheader("Filter Rows")
        if not cleaned_df.empty:
            filter_col = st.selectbox("Select column to filter", cleaned_df.columns)
            if is_numeric_dtype(cleaned_df[filter_col]):
                min_val, max_val = float(cleaned_df[filter_col].min()), float(cleaned_df[filter_col].max())
                selected_range = st.slider("Select range", min_val, max_val, (min_val, max_val))
                if st.button("Apply Filter"):
                    filtered_df = cleaned_df[(cleaned_df[filter_col] >= selected_range[0]) & 
                                           (cleaned_df[filter_col] <= selected_range[1])]
                    st.session_state.cleaned_df = filtered_df
                    st.success(f"Filter applied! Rows remaining: {len(filtered_df)}")
            else:
                unique_values = cleaned_df[filter_col].unique()
                selected_values = st.multiselect("Select values to keep", unique_values, default=unique_values)
                if st.button("Apply Filter"):
                    filtered_df = cleaned_df[cleaned_df[filter_col].isin(selected_values)]
                    st.session_state.cleaned_df = filtered_df
                    st.success(f"Filter applied! Rows remaining: {len(filtered_df)}")
        
        st.subheader("Reset to Original Data")
        if st.button("Reset All Changes"):
            st.session_state.cleaned_df = st.session_state.df.copy()
            # Reset numeric_cols to original
            st.session_state.numeric_cols = st.session_state.cleaned_df.select_dtypes(include=np.number).columns.tolist()
            st.success("Data reset to original state!")

    # Page 3: Exploratory Analysis
    elif page == "Exploratory Analysis":
        st.header("🔍 Exploratory Analysis")
        
        st.subheader("Descriptive Statistics")
        if st.checkbox("Show Full Descriptive Statistics"):
            st.write(cleaned_df.describe(include='all'))
        
        st.subheader("Correlation Analysis")
        if len(numeric_cols) > 1:
            corr_method = st.selectbox("Correlation method", ["pearson", "kendall", "spearman"])
            corr_matrix = cleaned_df[numeric_cols].corr(method=corr_method)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)
            
            if st.checkbox("Show Correlation Pairs"):
                corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
                corr_pairs = corr_pairs[corr_pairs != 1]  # Remove self-correlations
                st.write(corr_pairs.head(10))
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis")
        
        st.subheader("Column-wise Analysis")
        col = st.selectbox("Select a column", cleaned_df.columns)
        
        if is_numeric_dtype(cleaned_df[col]):
            tab1, tab2, tab3 = st.tabs(["Histogram", "Box Plot", "Violin Plot"])
            
            with tab1:
                fig, ax = plt.subplots()
                sns.histplot(cleaned_df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
            
            with tab2:
                fig, ax = plt.subplots()
                sns.boxplot(x=cleaned_df[col], ax=ax)
                ax.set_title(f"Box Plot of {col}")
                st.pyplot(fig)
            
            with tab3:
                fig, ax = plt.subplots()
                sns.violinplot(x=cleaned_df[col], ax=ax)
                ax.set_title(f"Violin Plot of {col}")
                st.pyplot(fig)
        else:
            tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 6))
                cleaned_df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"Value Counts of {col}")
                st.pyplot(fig)
            
            with tab2:
                fig, ax = plt.subplots(figsize=(8, 8))
                cleaned_df[col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)

    # Page 4: Advanced Visualizations
    elif page == "Advanced Visualizations":
        st.header("📊 Advanced Visualizations")
        
        st.subheader("Scatter Plot")
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", numeric_cols, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
            
            hue_col = st.selectbox("Hue (optional)", [None] + list(cleaned_df.columns))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=cleaned_df, x=x_axis, y=y_axis, hue=hue_col, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Need at least 2 numeric columns for scatter plot")
        
        st.subheader("Pair Plot")
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select columns for pair plot", numeric_cols, default=numeric_cols[:3])
            hue_col = st.selectbox("Hue for pair plot", [None] + list(cleaned_df.columns))
            
            if len(selected_cols) >= 2:
                if st.checkbox("Generate Pair Plot (may take time for large datasets)"):
                    fig = sns.pairplot(cleaned_df, vars=selected_cols, hue=hue_col)
                    st.pyplot(fig)
            else:
                st.warning("Select at least 2 columns for pair plot")
        else:
            st.warning("Need at least 2 numeric columns for pair plot")
        
        st.subheader("Interactive 3D Plot (Plotly)")
        if len(numeric_cols) >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_axis = st.selectbox("X-axis 3D", numeric_cols, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis 3D", numeric_cols, index=1)
            with col3:
                z_axis = st.selectbox("Z-axis 3D", numeric_cols, index=2)
            
            color_col = st.selectbox("Color by", [None] + list(cleaned_df.columns))
            size_col = st.selectbox("Size by (numeric only)", [None] + numeric_cols)
            
            fig = px.scatter_3d(cleaned_df, x=x_axis, y=y_axis, z=z_axis, 
                               color=color_col, size=size_col, hover_name=cleaned_df.index)
            st.plotly_chart(fig)
        else:
            st.warning("Need at least 3 numeric columns for 3D plot")

    # Page 5: Machine Learning
    elif page == "Machine Learning":
        st.header("🤖 Basic Machine Learning")
        
        st.subheader("Principal Component Analysis (PCA)")
        if len(numeric_cols) >= 2:
            n_components = st.slider("Number of components", 2, min(10, len(numeric_cols)), 2)
            
            if st.button("Run PCA"):
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cleaned_df[numeric_cols].dropna())
                
                # Perform PCA
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(scaled_data)
                
                # Create a DataFrame with the PCA results
                pca_df = pd.DataFrame(data=pca_result, 
                                     columns=[f"PC{i+1}" for i in range(n_components)])
                
                # Explained variance
                st.subheader("Explained Variance")
                explained_var = pca.explained_variance_ratio_
                fig, ax = plt.subplots()
                ax.bar(range(1, n_components+1), explained_var, alpha=0.5, align='center',
                       label='Individual explained variance')
                ax.step(range(1, n_components+1), np.cumsum(explained_var), where='mid',
                        label='Cumulative explained variance')
                ax.set_ylabel('Explained variance ratio')
                ax.set_xlabel('Principal component index')
                ax.legend(loc='best')
                st.pyplot(fig)
                
                # PCA Scatter plot
                st.subheader("PCA Results")
                if n_components >= 2:
                    x_pc = st.selectbox("X-axis PC", pca_df.columns, index=0)
                    y_pc = st.selectbox("Y-axis PC", pca_df.columns, index=1)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=pca_df, x=x_pc, y=y_pc, ax=ax)
                    ax.set_title("PCA Components")
                    st.pyplot(fig)
                
                # Component loadings
                st.subheader("Component Loadings")
                loadings = pd.DataFrame(pca.components_.T, 
                                      columns=[f"PC{i+1}" for i in range(n_components)],
                                      index=numeric_cols)
                st.write(loadings)
        else:
            st.warning("Need at least 2 numeric columns for PCA")

else:
    st.info("📁 Please upload a CSV file to get started.")

# Footer
st.markdown("---")
st.markdown("### 🚀 Advanced CSV Data Analyzer")
st.markdown("Created with Streamlit, Pandas, and Matplotlib")