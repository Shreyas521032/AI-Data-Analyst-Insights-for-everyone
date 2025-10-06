import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import json
import io
import base64
from PIL import Image

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
    st.session_state.analysis_results = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'
if 'generated_visualizations' not in st.session_state:
    st.session_state.generated_visualizations = []

def initialize_gemini(api_key):
    """Initialize Gemini AI client"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

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
    """Use Gemini to analyze data and provide insights with visualization recommendations"""
    
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
3. Specific visualization recommendations in this JSON format:
   {{"visualizations": [
     {{"type": "scatter/bar/line/box/histogram", "x_col": "column_name", "y_col": "column_name" (or null), "color_col": "column_name" (or null), "title": "Description", "insight": "What this shows"}},
     ...
   ]}}
4. Any potential issues or considerations

Format your response with clear sections. Put the JSON visualization recommendations in a code block."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

def parse_visualization_recommendations(analysis_text, df):
    """Parse visualization recommendations from AI response and create them"""
    visualizations = []
    
    try:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', analysis_text, re.DOTALL)
        
        if json_match:
            viz_data = json.loads(json_match.group(1))
            
            if 'visualizations' in viz_data:
                for viz in viz_data['visualizations']:
                    try:
                        viz_type = viz.get('type', 'scatter')
                        x_col = viz.get('x_col')
                        y_col = viz.get('y_col')
                        color_col = viz.get('color_col')
                        title = viz.get('title', f'{viz_type.title()} Plot')
                        insight = viz.get('insight', '')
                        
                        # Validate columns exist
                        if x_col and x_col in df.columns:
                            fig = create_custom_visualization(df, viz_type, x_col, y_col, color_col, title)
                            if fig:
                                visualizations.append({
                                    'figure': fig,
                                    'title': title,
                                    'insight': insight
                                })
                    except Exception as e:
                        continue
    except Exception as e:
        pass
    
    return visualizations

def create_custom_visualization(df, viz_type, x_col, y_col=None, color_col=None, title=None):
    """Create custom visualizations based on AI recommendations"""
    try:
        if not title:
            title = f"{viz_type.title()}: {x_col}"
            if y_col:
                title += f" vs {y_col}"
        
        if viz_type == "scatter" and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
        elif viz_type == "bar":
            if y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
            else:
                value_counts = df[x_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values, title=title)
        elif viz_type == "line" and y_col:
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
        elif viz_type == "box":
            if y_col:
                fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title)
            else:
                fig = px.box(df, y=x_col, title=title)
        elif viz_type == "histogram":
            fig = px.histogram(df, x=x_col, color=color_col, title=title)
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def fig_to_base64(fig):
    """Convert plotly figure to base64 for embedding in HTML"""
    img_bytes = fig.to_image(format="png", width=1200, height=600)
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

def generate_comprehensive_report(df, eda_results, ai_insights, visualizations):
    """Generate comprehensive HTML report with all visualizations"""
    
    report_html = f"""
    <html>
    <head>
        <title>Data Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .section {{
                background: white;
                padding: 25px;
                margin-bottom: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            h2 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            h3 {{
                color: #764ba2;
            }}
            .metric {{
                display: inline-block;
                background: #f0f0f0;
                padding: 15px 25px;
                margin: 10px;
                border-radius: 8px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 28px;
                font-weight: bold;
                color: #667eea;
            }}
            .metric-label {{
                color: #666;
                font-size: 14px;
            }}
            .visualization {{
                margin: 20px 0;
                text-align: center;
            }}
            .visualization img {{
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .insight {{
                background: #e8f4f8;
                padding: 15px;
                border-left: 4px solid #17a2b8;
                margin: 10px 0;
                border-radius: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #667eea;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Data Analysis Report</h1>
            <p>AI-Powered Analysis Results</p>
        </div>
        
        <div class="section">
            <h2>📈 Dataset Overview</h2>
            <div class="metric">
                <div class="metric-value">{eda_results['shape'][0]:,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            <div class="metric">
                <div class="metric-value">{eda_results['shape'][1]}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            <div class="metric">
                <div class="metric-value">{sum(1 for dtype in eda_results['dtypes'].values() if dtype in ['float64', 'int64'])}</div>
                <div class="metric-label">Numeric Columns</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🧠 AI Insights</h2>
            <div style="white-space: pre-wrap;">{ai_insights}</div>
        </div>
        
        <div class="section">
            <h2>📊 Visualizations & Insights</h2>
    """
    
    # Add visualizations
    for i, viz_data in enumerate(visualizations):
        try:
            img_base64 = fig_to_base64(viz_data['figure'])
            report_html += f"""
            <div class="visualization">
                <h3>{viz_data['title']}</h3>
                <img src="data:image/png;base64,{img_base64}" alt="{viz_data['title']}">
                {f'<div class="insight"><strong>Insight:</strong> {viz_data["insight"]}</div>' if viz_data.get('insight') else ''}
            </div>
            """
        except Exception as e:
            continue
    
    report_html += f"""
        </div>
        
        <div class="section">
            <h2>📋 Column Information</h2>
            <table>
                <tr>
                    <th>Column Name</th>
                    <th>Data Type</th>
                    <th>Unique Values</th>
                </tr>
    """
    
    for col in df.columns:
        report_html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{eda_results['dtypes'][col]}</td>
                    <td>{eda_results['unique_counts'][col]}</td>
                </tr>
        """
    
    report_html += """
            </table>
        </div>
        
        <div class="section">
            <h2>📊 Summary Statistics</h2>
    """
    
    summary_df = df.describe()
    report_html += summary_df.to_html(classes='dataframe')
    
    report_html += """
        </div>
        
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>Generated by AI Data Analysis Agent | Powered by Gemini AI</p>
        </div>
    </body>
    </html>
    """
    
    return report_html

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
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "🤖 AI Analysis", "📊 Results & Visualizations", "🔄 Project Flow"])

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
                        
                        # Parse and generate visualizations
                        with st.spinner("📊 Generating recommended visualizations..."):
                            visualizations = parse_visualization_recommendations(
                                analysis,
                                st.session_state.processed_data
                            )
                            st.session_state.generated_visualizations = visualizations
                        
                        st.session_state.stage = 'visualization'
                        
                        st.markdown('<div class="status-box status-success">✅ Analysis Complete!</div>', unsafe_allow_html=True)
                        
                        # Display AI Insights
                        st.subheader("🧠 AI Insights")
                        st.markdown(analysis)
                        
                        # Display Generated Visualizations
                        if visualizations:
                            st.subheader("📊 AI-Recommended Visualizations")
                            for viz_data in visualizations:
                                st.plotly_chart(viz_data['figure'], use_container_width=True)
                                if viz_data.get('insight'):
                                    st.info(f"💡 **Insight:** {viz_data['insight']}")
                        
                    except Exception as e:
                        st.markdown(f'<div class="status-box status-error">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
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
        
        col1, col2, col3 = st.columns(3)
        
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
            # Download simple analysis report
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
                    label="📄 Download Text Report",
                    data=report,
                    file_name="analysis_report.md",
                    mime="text/markdown"
                )
        
        with col3:
            # Download comprehensive HTML report with visualizations
            if 'ai_insights' in st.session_state.analysis_results:
                try:
                    html_report = generate_comprehensive_report(
                        df,
                        st.session_state.analysis_results['eda'],
                        st.session_state.analysis_results['ai_insights'],
                        st.session_state.generated_visualizations
                    )
                    st.download_button(
                        label="📊 Download Full Report (HTML)",
                        data=html_report,
                        file_name="comprehensive_report.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Error generating HTML report: {str(e)}")
    else:
        st.info("👆 Please upload and analyze data first.")

with tab4:
    st.header("🔄 Project Flow Diagram")
    
    st.markdown("""
    This section visualizes the complete workflow of the AI Data Analysis Agent.
    Upload an image of your project flow diagram below.
    """)
    
    # Option to upload flow diagram
    flow_image = st.file_uploader("Upload Project Flow Diagram", type=['png', 'jpg', 'jpeg', 'svg'])
    
    if flow_image is not None:
        try:
            image = Image.open(flow_image)
            st.image(image, caption="Project Flow Diagram", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    else:
        # Display default flow description
        st.markdown("""
        ### Default Project Workflow
        
        ```
        1. 📁 DATA UPLOAD
           ↓
           • CSV File Input
           • File Validation (Size < 50MB)
           ↓
        
        2. 🧹 DATA CLEANING
           ↓
           • Handle Missing Values
           • Remove Duplicates
           • Standardize Column Names
           ↓
        
        3. 📈 EXPLORATORY DATA ANALYSIS
           ↓
           • Statistical Summary
           • Data Type Analysis
           • Initial Visualizations
           ↓
        
        4. 🎯 GOAL DEFINITION
           ↓
           • User Input
           • Analysis Objective
           ↓
        
        5. 🤖 AI ANALYSIS (Gemini)
           ↓
           • Pattern Recognition
           • Insight Generation
           • Visualization Recommendations
           ↓
        
        6. 📊 VISUALIZATION GENERATION
           ↓
           • AI-Recommended Charts
           • Custom Visualizations
           • Interactive Plots
           ↓
        
        7. 📄 REPORT GENERATION
           ↓
           • Text Report (MD)
           • HTML Report with Visualizations
           • Processed Data Export
           ↓
        
        8. ✅ RESULTS & DOWNLOAD
        ```
        """)
        
        st.info("💡 **Tip:** Upload your own project flow diagram image to replace this default view!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Powered by Gemini AI | Built with Streamlit</p>
    <p>Your AI-powered data analysis assistant</p>
</div>
""", unsafe_allow_html=True)
