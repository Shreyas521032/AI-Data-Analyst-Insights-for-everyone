import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import json
import io

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .status-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    .status-info {
        background-color: #d1ecf1;
        border-color: #17a2b8;
        color: #0c5460;
    }
    .status-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'

def initialize_gemini(api_key):
    """Initialize Gemini AI client"""
    genai.configure(api_key=api_key)
    try:
        # Use Gemini 1.0 Flash (free tier model)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        # Fall back to Gemini 1.0 (free tier model)
        return genai.GenerativeModel('gemini-1.0-pro')

def clean_and_validate_data(df):
    """Data cleaning and preprocessing"""
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df, issues

def perform_eda(df):
    """Perform exploratory data analysis"""
    eda_results = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'summary_stats': df.describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_counts': {col: df[col].nunique() for col in df.columns}
    }
    return eda_results

def generate_visualizations(df):
    """Generate initial visualizations"""
    figs = []
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        # Correlation heatmap
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, 
                           title="Correlation Heatmap",
                           labels=dict(color="Correlation"),
                           color_continuous_scale="RdBu",
                           aspect="auto")
            figs.append(("Correlation Heatmap", fig))
        
        # Distribution plots for first few numeric columns
        for col in numeric_cols[:3]:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}",
                             marginal="box")
            figs.append((f"Distribution: {col}", fig))
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:2]:
        if df[col].nunique() < 20:
            value_counts = df[col].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f"Count of {col}",
                        labels={'x': col, 'y': 'Count'})
            figs.append((f"Count: {col}", fig))
    
    return figs

def analyze_with_gemini(model, df, eda_results, user_goal):
    """Use Gemini to analyze data and provide insights"""
    
    # Prepare data summary for Gemini
    data_summary = f"""
    Dataset Shape: {eda_results['shape']}
    Columns: {', '.join(eda_results['columns'])}
    
    Summary Statistics:
    {json.dumps(eda_results['summary_stats'], indent=2, default=str)}
    
    Data Types:
    {json.dumps(eda_results['dtypes'], indent=2, default=str)}
    
    User Goal: {user_goal}
    """
    
    # Get sample data
    sample_data = df.head(10).to_string()
    
    prompt = f"""You are an expert data analyst. Analyze the following dataset and provide insights based on the user's goal.

{data_summary}

Sample Data:
{sample_data}

Please provide:
1. Key insights and patterns in the data
2. Recommendations for analysis based on the user's goal
3. Suggested visualizations (be specific about chart types and variables)
4. Any potential issues or considerations

Format your response in clear sections."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

def create_custom_visualization(df, viz_type, x_col, y_col=None, color_col=None):
    """Create custom visualizations based on AI recommendations"""
    try:
        if viz_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=f"{viz_type.title()}: {x_col} vs {y_col}")
        elif viz_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                        title=f"{viz_type.title()}: {x_col}")
        elif viz_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col,
                         title=f"{viz_type.title()}: {x_col} vs {y_col}")
        elif viz_type == "box":
            fig = px.box(df, x=x_col, y=y_col, color=color_col,
                        title=f"{viz_type.title()}: {x_col}")
        elif viz_type == "histogram":
            fig = px.histogram(df, x=x_col, color=color_col,
                             title=f"{viz_type.title()}: {x_col}")
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

# Main App
st.markdown('<h1 class="main-header">🤖 AI Data Analysis Agent</h1>', unsafe_allow_html=True)
st.markdown("### Automated Data Analysis Pipeline with Gemini AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
    
    st.markdown("---")
    st.header("📊 Pipeline Status")
    
    stages = {
        'upload': '📁 Upload Data',
        'cleaning': '🧹 Data Cleaning',
        'eda': '📈 Exploratory Analysis',
        'goal': '🎯 Goal Setting',
        'analysis': '🔍 AI Analysis',
        'visualization': '📊 Visualization',
        'report': '📄 Report Generation'
    }
    
    for stage_key, stage_name in stages.items():
        if st.session_state.stage == stage_key:
            st.markdown(f"**✅ {stage_name}** (Current)")
        else:
            st.markdown(f"⏳ {stage_name}")

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🤖 AI Analysis", "📊 Results & Visualizations"])

with tab1:
    st.header("Step 1: Upload Your Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Check file size
            file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
            
            if file_size > 50:
                st.markdown('<div class="status-box status-error">❌ File too large! Please upload a file smaller than 50MB.</div>', unsafe_allow_html=True)
            else:
                # Data Ingestion
                st.session_state.stage = 'cleaning'
                df = pd.read_csv(uploaded_file)
                
                st.markdown('<div class="status-box status-success">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Data Cleaning & Preprocessing
                with st.spinner("🧹 Cleaning and preprocessing data..."):
                    cleaned_df, issues = clean_and_validate_data(df)
                    st.session_state.processed_data = cleaned_df
                
                if issues:
                    st.markdown('<div class="status-box status-warning">⚠️ Data Issues Found and Resolved:</div>', unsafe_allow_html=True)
                    for issue in issues:
                        st.write(f"• {issue}")
                else:
                    st.markdown('<div class="status-box status-success">✅ Data is clean and ready!</div>', unsafe_allow_html=True)
                
                # Show cleaned data preview
                st.subheader("📋 Cleaned Data Preview")
                st.dataframe(cleaned_df.head(10), use_container_width=True)
                
                # EDA
                st.session_state.stage = 'eda'
                with st.spinner("📈 Performing exploratory data analysis..."):
                    eda_results = perform_eda(cleaned_df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", eda_results['shape'][0])
                with col2:
                    st.metric("Total Columns", eda_results['shape'][1])
                with col3:
                    numeric_cols = sum(1 for dtype in eda_results['dtypes'].values() if dtype in ['float64', 'int64'])
                    st.metric("Numeric Columns", numeric_cols)
                
                # Initial Visualizations
                st.subheader("📊 Initial Insights & Visualizations")
                figs = generate_visualizations(cleaned_df)
                
                if figs:
                    for title, fig in figs[:3]:  # Show first 3 visualizations
                        st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.analysis_results = {
                    'eda': eda_results,
                    'visualizations': figs
                }
                
        except Exception as e:
            st.markdown(f'<div class="status-box status-error">❌ Error processing file: {str(e)}</div>', unsafe_allow_html=True)

with tab2:
    st.header("Step 2: AI-Powered Analysis")
    
    if st.session_state.processed_data is not None:
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar to continue.")
        else:
            st.session_state.stage = 'goal'
            
            # User Goal Input
            st.subheader("🎯 What would you like to discover?")
            user_goal = st.text_area(
                "Describe your analysis goal or question:",
                placeholder="e.g., I want to understand the relationship between sales and marketing spend, identify key trends, predict customer churn, etc.",
                height=100
            )
            
            if st.button("🚀 Start AI Analysis", type="primary"):
                if not user_goal:
                    st.warning("Please provide a goal or question for the analysis.")
                else:
                    st.session_state.stage = 'analysis'
                    
                    try:
                        model = initialize_gemini(api_key)
                        
                        with st.spinner("🤖 AI Agent is analyzing your data..."):
                            analysis = analyze_with_gemini(
                                model,
                                st.session_state.processed_data,
                                st.session_state.analysis_results['eda'],
                                user_goal
                            )
                        
                        st.session_state.analysis_results['ai_insights'] = analysis
                        st.session_state.stage = 'visualization'
                        
                        st.markdown('<div class="status-box status-success">✅ Analysis Complete!</div>', unsafe_allow_html=True)
                        
                        # Display AI Insights
                        st.subheader("🧠 AI Insights")
                        st.markdown(analysis)
                        
                    except Exception as e:
                        st.markdown(f'<div class="status-box status-error">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            
            # Show previous analysis if exists
            if 'ai_insights' in st.session_state.analysis_results:
                st.subheader("🧠 Previous AI Insights")
                st.markdown(st.session_state.analysis_results['ai_insights'])
    else:
        st.info("👆 Please upload a CSV file in the 'Upload & Process' tab first.")

with tab3:
    st.header("Step 3: Results & Custom Visualizations")
    
    if st.session_state.processed_data is not None:
        st.session_state.stage = 'report'
        
        df = st.session_state.processed_data
        
        # Custom Visualization Builder
        st.subheader("🎨 Create Custom Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            viz_type = st.selectbox(
                "Visualization Type",
                ["scatter", "bar", "line", "box", "histogram"]
            )
            
            x_col = st.selectbox("X-axis", df.columns)
        
        with col2:
            if viz_type in ["scatter", "bar", "line", "box"]:
                y_col = st.selectbox("Y-axis", [None] + df.columns.tolist())
            else:
                y_col = None
            
            color_col = st.selectbox("Color by (optional)", [None] + df.columns.tolist())
        
        if st.button("Generate Visualization"):
            fig = create_custom_visualization(df, viz_type, x_col, y_col, color_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Download Section
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download processed data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Processed Data (CSV)",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download analysis report
            if 'ai_insights' in st.session_state.analysis_results:
                report = f"""
# Data Analysis Report

## Dataset Overview
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}

## AI Insights
{st.session_state.analysis_results['ai_insights']}

## Column Information
{df.dtypes.to_string()}

## Summary Statistics
{df.describe().to_string()}
"""
                st.download_button(
                    label="📄 Download Analysis Report",
                    data=report,
                    file_name="analysis_report.md",
                    mime="text/markdown"
                )
    else:
        st.info("👆 Please upload and analyze data first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Powered by Gemini AI | Built with Streamlit</p>
    <p>Your AI-powered data analysis assistant</p>
</div>
""", unsafe_allow_html=True)
