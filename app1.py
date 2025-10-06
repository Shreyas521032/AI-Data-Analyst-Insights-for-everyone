import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import json
import io
import base64
from PIL import Image
import requests

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

# AI Agent State Management
class DataAnalysisAgent:
    def __init__(self):
        self.raw_data = None
        self.cleaned_data = None
        self.eda_insights = {}
        self.ai_analysis = None
        self.generated_charts = []
        self.agent_memory = []
        self.current_phase = 'idle'
        self.flow_diagram_url = ""  # Add your GitHub raw image URL here
        
    def reset_agent(self):
        """Reset agent to initial state"""
        self.__init__()
    
    def update_phase(self, phase):
        """Update current processing phase"""
        self.current_phase = phase
        self.agent_memory.append(f"Phase: {phase}")
    
    def log_action(self, action):
        """Log agent actions"""
        self.agent_memory.append(action)

# Initialize AI Agent in session state
if 'agent' not in st.session_state:
    st.session_state.agent = DataAnalysisAgent()

def initialize_ai_model(api_key):
    """Initialize Gemini AI model for the agent"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def agent_clean_data(df, agent):
    """Agent performs data cleaning operations"""
    agent.update_phase('data_cleaning')
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values detected: {missing[missing > 0].to_dict()}")
        agent.log_action(f"Handled {missing.sum()} missing values")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Detected {duplicates} duplicate rows")
        agent.log_action(f"Removed {duplicates} duplicate entries")
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    agent.log_action("Standardized column names")
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    agent.cleaned_data = df
    return df, issues

def agent_perform_eda(df, agent):
    """Agent performs exploratory data analysis"""
    agent.update_phase('exploratory_analysis')
    
    insights = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'summary_stats': df.describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_counts': {col: df[col].nunique() for col in df.columns}
    }
    
    agent.eda_insights = insights
    agent.log_action(f"Analyzed {insights['shape'][1]} features across {insights['shape'][0]} samples")
    
    return insights

def agent_generate_initial_viz(df, agent):
    """Agent generates initial visualizations"""
    agent.update_phase('initial_visualization')
    charts = []
    
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
            charts.append(("Correlation Heatmap", fig))
            agent.log_action("Generated correlation analysis")
        
        # Distribution plots for first few numeric columns
        for col in numeric_cols[:3]:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}",
                             marginal="box")
            charts.append((f"Distribution: {col}", fig))
            agent.log_action(f"Created distribution chart for {col}")
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:2]:
        if df[col].nunique() < 20:
            value_counts = df[col].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f"Count of {col}",
                        labels={'x': col, 'y': 'Count'})
            charts.append((f"Count: {col}", fig))
            agent.log_action(f"Created frequency chart for {col}")
    
    return charts

def agent_ai_analysis(model, df, eda_insights, user_objective, agent):
    """Agent performs AI-powered deep analysis"""
    agent.update_phase('ai_powered_analysis')
    
    # Prepare data summary for AI
    data_context = f"""
    Dataset Shape: {eda_insights['shape']}
    Columns: {', '.join(eda_insights['columns'])}
    
    Summary Statistics:
    {json.dumps(eda_insights['summary_stats'], indent=2, default=str)}
    
    Data Types:
    {json.dumps(eda_insights['dtypes'], indent=2, default=str)}
    
    User Objective: {user_objective}
    """
    
    # Get sample data
    sample_records = df.head(10).to_string()
    
    prompt = f"""You are an advanced data analysis AI agent. Analyze the following dataset and provide insights based on the user's objective.

{data_context}

Sample Records:
{sample_records}

Please provide:
1. Key patterns and insights discovered in the data
2. Actionable recommendations based on the user's objective
3. Specific visualization recommendations in this JSON format:
   {{"visualizations": [
     {{"type": "scatter/bar/line/box/histogram", "x_col": "column_name", "y_col": "column_name" (or null), "color_col": "column_name" (or null), "title": "Description", "insight": "What this reveals"}},
     ...
   ]}}
4. Potential data quality considerations or anomalies

Format your response with clear sections. Put the JSON visualization recommendations in a code block."""

    try:
        response = model.generate_content(prompt)
        agent.ai_analysis = response.text
        agent.log_action("Completed AI-powered deep analysis")
        return response.text
    except Exception as e:
        agent.log_action(f"AI analysis error: {str(e)}")
        return f"Error during AI analysis: {str(e)}"

def agent_parse_viz_recommendations(analysis_text, df, agent):
    """Agent parses AI recommendations and generates visualizations"""
    agent.update_phase('intelligent_visualization')
    charts = []
    
    try:
        # Extract JSON from AI response
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', analysis_text, re.DOTALL)
        
        if json_match:
            viz_specs = json.loads(json_match.group(1))
            
            if 'visualizations' in viz_specs:
                for spec in viz_specs['visualizations']:
                    try:
                        viz_type = spec.get('type', 'scatter')
                        x_col = spec.get('x_col')
                        y_col = spec.get('y_col')
                        color_col = spec.get('color_col')
                        title = spec.get('title', f'{viz_type.title()} Plot')
                        insight = spec.get('insight', '')
                        
                        # Validate columns exist
                        if x_col and x_col in df.columns:
                            fig = agent_create_visualization(df, viz_type, x_col, y_col, color_col, title)
                            if fig:
                                charts.append({
                                    'figure': fig,
                                    'title': title,
                                    'insight': insight
                                })
                                agent.log_action(f"Generated {viz_type} visualization: {title}")
                    except Exception as e:
                        continue
    except Exception as e:
        agent.log_action(f"Visualization parsing error: {str(e)}")
    
    agent.generated_charts = charts
    return charts

def agent_create_visualization(df, viz_type, x_col, y_col=None, color_col=None, title=None):
    """Agent creates custom visualizations"""
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
    """Convert plotly figure to base64"""
    img_bytes = fig.to_image(format="png", width=1200, height=600)
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

def agent_generate_report(df, eda_insights, ai_analysis, charts, agent):
    """Agent generates comprehensive analysis report"""
    agent.update_phase('report_generation')
    
    report_html = f"""
    <html>
    <head>
        <title>AI Agent Analysis Report</title>
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
            <h1>🤖 AI Agent Analysis Report</h1>
            <p>Autonomous Data Analysis Results</p>
        </div>
        
        <div class="section">
            <h2>📊 Dataset Overview</h2>
            <div class="metric">
                <div class="metric-value">{eda_insights['shape'][0]:,}</div>
                <div class="metric-label">Total Records</div>
            </div>
            <div class="metric">
                <div class="metric-value">{eda_insights['shape'][1]}</div>
                <div class="metric-label">Features</div>
            </div>
            <div class="metric">
                <div class="metric-value">{sum(1 for dtype in eda_insights['dtypes'].values() if dtype in ['float64', 'int64'])}</div>
                <div class="metric-label">Numeric Features</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🧠 AI Agent Insights</h2>
            <div style="white-space: pre-wrap;">{ai_analysis}</div>
        </div>
        
        <div class="section">
            <h2>📊 Agent-Generated Visualizations</h2>
    """
    
    # Add visualizations
    for i, viz_data in enumerate(charts):
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
            <h2>📋 Feature Information</h2>
            <table>
                <tr>
                    <th>Feature Name</th>
                    <th>Data Type</th>
                    <th>Unique Values</th>
                </tr>
    """
    
    for col in df.columns:
        report_html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{eda_insights['dtypes'][col]}</td>
                    <td>{eda_insights['unique_counts'][col]}</td>
                </tr>
        """
    
    report_html += """
            </table>
        </div>
        
        <div class="section">
            <h2>📊 Statistical Summary</h2>
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
    
    agent.log_action("Generated comprehensive analysis report")
    return report_html

# Main App
st.markdown('<h1 class="main-header">🤖 AI Data Analysis Agent</h1>', unsafe_allow_html=True)
st.markdown("### Autonomous Data Analysis Pipeline with AI Intelligence")

# Sidebar
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
    
    st.markdown("---")
    st.header("🤖 Agent Status")
    
    agent_phases = {
        'idle': '⏸️ Idle',
        'data_cleaning': '🧹 Cleaning Data',
        'exploratory_analysis': '📈 Exploring Data',
        'goal_setting': '🎯 Setting Objective',
        'ai_powered_analysis': '🤖 AI Analysis',
        'intelligent_visualization': '📊 Creating Visualizations',
        'report_generation': '📄 Generating Report'
    }
    
    current_phase = st.session_state.agent.current_phase
    status_container = st.container()
    
    with status_container:
        for phase_key, (emoji, phase_name) in agent_phases.items():
            if current_phase == phase_key:
                st.markdown(f"**✅ {emoji} {phase_name}** 🔴 *Active*")
            elif phase_key in ['idle']:
                st.markdown(f"{emoji} {phase_name}")
            else:
                # Check if this phase has been completed
                phase_completed = False
                phase_order = list(agent_phases.keys())
                if current_phase != 'idle':
                    try:
                        current_idx = phase_order.index(current_phase)
                        phase_idx = phase_order.index(phase_key)
                        phase_completed = phase_idx < current_idx
                    except ValueError:
                        pass
                
                if phase_completed:
                    st.markdown(f"✅ {emoji} {phase_name}")
                else:
                    st.markdown(f"⏳ {emoji} {phase_name}")
    
    st.markdown("---")
    st.header("📝 Agent Memory")
    if st.session_state.agent.agent_memory:
        memory_expander = st.expander("View Agent Actions", expanded=False)
        with memory_expander:
            # Show last 15 actions in reverse chronological order
            recent_actions = st.session_state.agent.agent_memory[-15:]
            for i, action in enumerate(reversed(recent_actions)):
                st.caption(f"{len(recent_actions) - i}. {action}")
            
            if len(st.session_state.agent.agent_memory) > 15:
                st.caption(f"... and {len(st.session_state.agent.agent_memory) - 15} more actions")
    else:
        st.info("No actions recorded yet")
    
    st.markdown("---")
    if st.button("🔄 Reset Agent", type="secondary"):
        st.session_state.agent.reset_agent()
        st.rerun()

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📤 Data Ingestion", "🤖 AI Analysis", "📊 Results & Insights", "🔄 Agent Workflow"])

with tab1:
    st.header("Phase 1: Data Ingestion & Preprocessing")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Check file size
            file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
            
            if file_size > 50:
                st.markdown('<div class="status-box status-error">❌ Dataset too large! Maximum size: 50MB</div>', unsafe_allow_html=True)
            else:
                # Data Ingestion
                st.session_state.agent.update_phase('data_cleaning')
                df = pd.read_csv(uploaded_file)
                st.session_state.agent.raw_data = df
                
                st.markdown('<div class="status-box status-success">✅ Dataset ingested successfully!</div>', unsafe_allow_html=True)
                st.write(f"**Shape:** {df.shape[0]} records × {df.shape[1]} features")
                
                # Data Cleaning
                with st.spinner("🧹 Agent is cleaning and preprocessing data..."):
                    cleaned_df, issues = agent_clean_data(df, st.session_state.agent)
                
                if issues:
                    st.markdown('<div class="status-box status-warning">⚠️ Data Quality Issues Resolved by Agent:</div>', unsafe_allow_html=True)
                    for issue in issues:
                        st.write(f"• {issue}")
                else:
                    st.markdown('<div class="status-box status-success">✅ Data is clean and ready for analysis!</div>', unsafe_allow_html=True)
                
                # Show cleaned data preview
                st.subheader("📋 Cleaned Dataset Preview")
                st.dataframe(cleaned_df.head(10), use_container_width=True)
                
                # EDA
                st.session_state.agent.update_phase('exploratory_analysis')
                with st.spinner("📈 Agent performing exploratory analysis..."):
                    eda_insights = agent_perform_eda(cleaned_df, st.session_state.agent)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", eda_insights['shape'][0])
                with col2:
                    st.metric("Total Features", eda_insights['shape'][1])
                with col3:
                    numeric_cols = sum(1 for dtype in eda_insights['dtypes'].values() if dtype in ['float64', 'int64'])
                    st.metric("Numeric Features", numeric_cols)
                
                # Initial Visualizations
                st.subheader("📊 Initial Agent-Generated Insights")
                charts = agent_generate_initial_viz(cleaned_df, st.session_state.agent)
                
                if charts:
                    for title, fig in charts[:3]:
                        st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.markdown(f'<div class="status-box status-error">❌ Error processing dataset: {str(e)}</div>', unsafe_allow_html=True)

with tab2:
    st.header("Phase 2: AI-Powered Deep Analysis")
    
    if st.session_state.agent.cleaned_data is not None:
        if not api_key:
            st.warning("⚠️ Please configure Gemini API key in the sidebar.")
        else:
            st.session_state.agent.update_phase('goal_setting')
            
            # User Objective Input
            st.subheader("🎯 Define Analysis Objective")
            user_objective = st.text_area(
                "What insights are you seeking from this data?",
                placeholder="e.g., Discover patterns in customer behavior, identify factors affecting sales performance, predict trends, etc.",
                height=100
            )
            
            if st.button("🚀 Initiate AI Analysis", type="primary"):
                if not user_objective:
                    st.warning("Please define an analysis objective.")
                else:
                    st.session_state.agent.update_phase('ai_powered_analysis')
                    
                    try:
                        ai_model = initialize_ai_model(api_key)
                        
                        with st.spinner("🤖 AI Agent analyzing your dataset..."):
                            analysis = agent_ai_analysis(
                                ai_model,
                                st.session_state.agent.cleaned_data,
                                st.session_state.agent.eda_insights,
                                user_objective,
                                st.session_state.agent
                            )
                        
                        # Parse and generate visualizations
                        with st.spinner("📊 Agent generating intelligent visualizations..."):
                            charts = agent_parse_viz_recommendations(
                                analysis,
                                st.session_state.agent.cleaned_data,
                                st.session_state.agent
                            )
                        
                        st.session_state.agent.update_phase('intelligent_visualization')
                        
                        st.markdown('<div class="status-box status-success">✅ AI Analysis Complete!</div>', unsafe_allow_html=True)
                        
                        # Display AI Insights
                        st.subheader("🧠 AI Agent Discoveries")
                        st.markdown(analysis)
                        
                        # Display Generated Visualizations
                        if charts:
                            st.subheader("📊 Agent-Recommended Visualizations")
                            for viz_data in charts:
                                st.plotly_chart(viz_data['figure'], use_container_width=True)
                                if viz_data.get('insight'):
                                    st.info(f"💡 **Agent Insight:** {viz_data['insight']}")
                        
                    except Exception as e:
                        st.markdown(f'<div class="status-box status-error">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Please ingest data in the 'Data Ingestion' tab first.")

with tab3:
    st.header("Phase 3: Results & Custom Analytics")
    
    if st.session_state.agent.cleaned_data is not None:
        st.session_state.agent.update_phase('report_generation')
        
        df = st.session_state.agent.cleaned_data
        
        # Custom Visualization Builder
        st.subheader("🎨 Custom Visualization Builder")
        
        col1, col2 = st.columns(2)
        
        with col1:
            viz_type = st.selectbox(
                "Chart Type",
                ["scatter", "bar", "line", "box", "histogram"]
            )
            
            x_col = st.selectbox("X-axis Feature", df.columns)
        
        with col2:
            if viz_type in ["scatter", "bar", "line", "box"]:
                y_col = st.selectbox("Y-axis Feature", [None] + df.columns.tolist())
            else:
                y_col = None
            
            color_col = st.selectbox("Color by (optional)", [None] + df.columns.tolist())
        
        if st.button("Generate Custom Visualization"):
            fig = agent_create_visualization(df, viz_type, x_col, y_col, color_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Download Section
        st.subheader("📥 Export Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download processed data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Export Cleaned Data (CSV)",
                data=csv,
                file_name="agent_processed_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download analysis report
            if st.session_state.agent.ai_analysis:
                report = f"""
# AI Agent Analysis Report

## Dataset Overview
- Records: {df.shape[0]}
- Features: {df.shape[1]}

## AI Agent Discoveries
{st.session_state.agent.ai_analysis}

## Feature Information
{df.dtypes.to_string()}

## Statistical Summary
{df.describe().to_string()}
"""
                st.download_button(
                    label="📄 Export Text Report (MD)",
                    data=report,
                    file_name="agent_analysis_report.md",
                    mime="text/markdown"
                )
        
        with col3:
            # Download comprehensive HTML report
            if st.session_state.agent.ai_analysis:
                try:
                    html_report = agent_generate_report(
                        df,
                        st.session_state.agent.eda_insights,
                        st.session_state.agent.ai_analysis,
                        st.session_state.agent.generated_charts,
                        st.session_state.agent
                    )
                    st.download_button(
                        label="📊 Export Full Report (HTML)",
                        data=html_report,
                        file_name="agent_comprehensive_report.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    else:
        st.info("👆 Please ingest and analyze data first.")

with tab4:
    st.header("🔄 AI Agent Workflow Architecture")
    
    st.markdown("""
    This section displays the autonomous workflow architecture of the AI Data Analysis Agent.
    """)
    
    # Configuration for flow diagram URL
    st.subheader("⚙️ Flow Diagram")

    DEFAULT_FLOW_URL = "https://raw.githubusercontent.com/Shreyas521032/AI-Data-Analyst-Insights-for-everyone/main/flow/AI%20AGENT.png"
    
    st.markdown("---")

    def fetch_image_bytes(url: str, timeout: int = 10):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            return None

    # Load & display the flow image
    image_bytes = fetch_image_bytes(DEFAULT_FLOW_URL)
    if image_bytes:
        try:
            # Open via PIL to ensure format correctness
            img = Image.open(io.BytesIO(image_bytes))
            # Display full width in the Streamlit page container
            st.image(img, caption="AI Agent Workflow Architecture", use_container_width=True)
        except Exception:
            # Log full details on the server and show a friendly message to the user
            st.info("💡 Ensure the URL points to a raw PNG/JPG file in the GitHub repository.")
    else:
        st.error("❌ Failed to load flow diagram from repository.")
        st.info("💡 Check the DEFAULT_FLOW_URL value in the code and ensure it points to the raw file (e.g., raw.githubusercontent.com/...)")

    st.markdown("---")

    st.subheader("🏗️ Agent Architecture Components")
        
    col1, col2 = st.columns(2)
        
    with col1:
            st.markdown("""
            #### 🧠 Core Agent Components
            - **Data Manager**: Handles ingestion & storage
            - **Preprocessing Engine**: Cleans & transforms data
            - **EDA Module**: Statistical analysis & profiling
            - **AI Interface**: Gemini API integration
            - **Visualization Engine**: Chart generation system
            - **Report Builder**: Multi-format output generator
            """)
        
    with col2:
            st.markdown("""
            #### 💾 Agent State Management
            - **raw_data**: Original dataset buffer
            - **cleaned_data**: Processed dataset cache
            - **eda_insights**: Analysis results store
            - **ai_analysis**: AI insights repository
            - **generated_charts**: Visualization library
            - **agent_memory**: Action history log
            - **current_phase**: Workflow state tracker
            """)
        
    st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Made with ❤️ by Shreyas Kasture</p>
    <p>Autonomous AI Agent for Intelligent Data Analysis</p>
</div>
""", unsafe_allow_html=True)
