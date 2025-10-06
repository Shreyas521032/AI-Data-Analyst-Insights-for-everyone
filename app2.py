import streamlit as st
import pandas as pd
import time
import io
import random

# --- Configuration and Mock Functions ---

# Set up the page
st.set_page_config(
    page_title="AI Data Analyst Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'clean_data_ready' not in st.session_state:
    st.session_state.clean_data_ready = False
if 'user_goal' not in st.session_state:
    st.session_state.user_goal = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'step' not in st.session_state:
    st.session_state.step = "upload" # upload, eda, goal, planning, visualization, report

# --- Mock AI Agent Functions (Placeholders for complex logic) ---

# Mock function to simulate Data Ingestion and Edge Cases
def ai_agent_data_ingestion(uploaded_file):
    MAX_FILE_SIZE_MB = 10 
    
    # Edge Case: File Too Large
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.session_state.step = "upload"
        return {"status": "error", "message": f"File Too Large! The size limit is {MAX_FILE_SIZE_MB} MB."}
    
    try:
        # Check for CSV format (simple check)
        if not uploaded_file.name.lower().endswith('.csv'):
             # Edge Case: Invalid File Format
            st.session_state.step = "upload"
            return {"status": "error", "message": "Invalid File Format! Please upload a CSV file."}
            
        data = pd.read_csv(uploaded_file)
        
        # Mock Data Cleaning & Preprocessing (e.g., handling NaNs)
        data = data.dropna(axis=0, how='any') # Drop rows with any NaN for simplicity
        
        return {"status": "success", "data": data}
    except Exception as e:
        # Catch other read/processing errors
        st.session_state.step = "upload"
        return {"status": "error", "message": f"An error occurred during ingestion: {e}"}

# Mock function for Exploratory Data Analysis (EDA)
def ai_agent_eda(data):
    st.info("📊 AI Agent performing **Exploratory Data Analysis**...")
    time.sleep(1.5) # Simulate processing time

    # Generate Initial Insights & Visuals
    # In a real app, this would involve feature engineering, stats, and chart generation
    
    # Mock Insights
    num_rows = len(data)
    num_cols = len(data.columns)
    
    # Simple Mock Visuals (e.g., first 5 rows and a histogram of a random numeric column)
    st.subheader("Initial Insights")
    st.write(f"Dataset has **{num_rows} rows** and **{num_cols} columns**.")
    st.write("First 5 rows of the cleaned data:")
    st.dataframe(data.head())
    
    # Mock visual: Histogram of a numeric column
    numeric_cols = data.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        col = random.choice(numeric_cols)
        st.subheader(f"Distribution of '{col}'")
        st.bar_chart(data[col].value_counts().head(20)) # Simple bar chart for distribution
    else:
         st.subheader("No numeric columns found for initial visualization.")
         
    return "EDA Complete"

# Mock function for Analysis Planning & Execution and Insight Generation
def ai_agent_analysis(data, goal):
    st.info(f"🧠 AI Agent analyzing goal: **'{goal}'** and executing plan...")
    time.sleep(2)

    # Edge Case: Ambiguous Goal (Simulated by checking for a keyword)
    if any(word in goal.lower() for word in ["general", "something", "tell me about"]):
        return {"status": "ambiguous"}

    # Mock Insight Generation
    # In a real scenario, this would use an LLM or specific data analysis scripts
    if "sales" in goal.lower() or "revenue" in goal.lower():
        if 'sales' in data.columns:
            total = data['sales'].sum()
            mean = data['sales'].mean()
            # Edge Case: No Significant Trend (Simulated)
            if mean < 100:
                 return {"status": "no_trend"} # Reports "No Major Trend Found"
            
            insight = f"The analysis of **{goal}** reveals that the **Total Sales** across the dataset is **${total:,.2f}** with an average of **${mean:,.2f}** per entry. The top 5 customers account for 30% of total sales."
        else:
            insight = f"The analysis for **{goal}** could not be performed as a 'sales' column was not found. Here is a dummy insight: Data correlation analysis suggests column A and column B have a weak positive relationship (r=0.2)."
    else:
        insight = f"Specific analysis for **{goal}** complete. A deep dive into the data distribution shows column '{data.columns[0]}' has a skewed distribution."
        
    return {"status": "success", "insight": insight}

# Mock function for Visualization Creation
def ai_agent_visualization(data, insight):
    st.info("🎨 AI Agent creating the best visualization...")
    time.sleep(1.5)
    
    # Edge Case: Sampled Data Needs Advanced Viz (Simulated by checking insight length)
    if len(insight) > 200 and len(data) > 100:
        # Recommends Specific Chart Type
        chart_recommendation = "Due to the complexity and volume of data, a **Multi-series Scatter Plot** with trendlines is recommended to visualize the findings."
        return {"status": "recommend_viz", "recommendation": chart_recommendation}
        
    # Simple Mock Visualization (Bar Chart)
    st.subheader("Key Visualization")
    # Use the first two columns for a simple visualization
    if len(data.columns) >= 2:
        x_col = data.columns[0]
        y_col = data.columns[1]
        
        # Simple count for the first column
        st.bar_chart(data[x_col].value_counts().head(10))
        st.caption(f"Visualizing the count distribution of the column: **{x_col}**")
        
    return {"status": "success"}

# --- Streamlit UI Logic ---

st.title("🤖 Single-Click AI Data Analyst Agent")
st.markdown("""
This agent follows the complete workflow: **Ingestion $\\rightarrow$ Cleaning $\\rightarrow$ EDA $\\rightarrow$ Goal $\\rightarrow$ Analysis $\\rightarrow$ Visualization $\\rightarrow$ Report.**
""")
st.divider()

# --- Step 1: User Uploads CSV ---
def render_upload_step():
    st.header("Step 1: Upload Data")
    uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"], key="csv_uploader")
    
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            result = ai_agent_data_ingestion(uploaded_file)
            
            if result["status"] == "error":
                st.error(f"❌ Error: {result['message']}")
            else:
                st.session_state.data = result["data"]
                st.session_state.clean_data_ready = True
                st.session_state.step = "eda"
                st.success("✅ File Ingestion & Cleaning Complete!")
                st.rerun()

# --- Step 2: EDA & Initial Insights ---
def render_eda_step():
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    
    if st.session_state.data is not None:
        ai_agent_eda(st.session_state.data)
        
        st.divider()
        st.session_state.step = "goal"
        st.rerun() # Move to the next step

# --- Step 3: User Provides Goal/Question ---
def render_goal_step():
    st.header("Step 3: Define Goal/Question")
    st.info("The AI Agent is now ready. What question do you want to answer with this data?")
    
    user_goal = st.text_input(
        "Your Analysis Goal:", 
        placeholder="e.g., What are the key drivers of sales revenue?",
        key="goal_input"
    )
    
    if st.button("Start Analysis $\\rightarrow$"):
        if user_goal:
            st.session_state.user_goal = user_goal
            st.session_state.step = "planning"
            st.rerun()
        else:
            st.warning("Please enter a goal for the analysis.")

# --- Step 4: Analysis Planning & Execution ---
def render_planning_step():
    st.header("Step 4: Analysis Planning & Insight Generation")
    
    if st.session_state.user_goal and st.session_state.data is not None:
        result = ai_agent_analysis(st.session_state.data, st.session_state.user_goal)
        
        if result["status"] == "ambiguous":
            # Edge Case: Ambiguous Goal - Ask for Clarification
            st.error("❓ Ambiguous Goal detected. Please rephrase your question to be more specific.")
            st.session_state.step = "goal"
            st.session_state.user_goal = None # Reset goal
            st.rerun()
            
        elif result["status"] == "no_trend":
            # Edge Case: No Significant Trend - Reports "No Major Trend Found"
            st.session_state.report = "No Major Trend Found: The data analysis did not reveal any statistically significant patterns or trends related to your goal."
            st.session_state.step = "report_presentation"
            st.error("⚠️ No Major Trend Found!")
            st.rerun()
            
        elif result["status"] == "success":
            st.session_state.insights = result["insight"]
            st.success("✨ Insight Generation Complete!")
            st.markdown(f"**Generated Insight:** {st.session_state.insights}")
            st.divider()
            st.session_state.step = "visualization"
            st.rerun()

# --- Step 5: Visualization Creation ---
def render_visualization_step():
    st.header("Step 5: Visualization Creation")
    
    if st.session_state.insights and st.session_state.data is not None:
        result = ai_agent_visualization(st.session_state.data, st.session_state.insights)
        
        if result["status"] == "recommend_viz":
            # Edge Case: Complex/Sampled Data Needs Advanced Viz - Recommends Chart Type
            st.warning(f"💡 Advanced Visualization Recommended: {result['recommendation']}")
            st.session_state.report = f"Analysis Goal: {st.session_state.user_goal}\n\nKey Insight: {st.session_state.insights}\n\nVisualization Note: {result['recommendation']}"
            st.session_state.step = "report_presentation"
            st.rerun()
        
        elif result["status"] == "success":
            # Final Report compilation
            st.session_state.report = f"## AI Agent Analysis Report\n\n### 1. Analysis Goal\n{st.session_state.user_goal}\n\n### 2. Key Insight\n{st.session_state.insights}\n\n### 3. Conclusion\nBased on the data and insights, the primary takeaway is: [Your final summary goes here]."
            st.session_state.step = "report_presentation"
            st.rerun()

# --- Step 6: Report & Findings Presentation ---
def render_report_step():
    st.header("Step 6: Final Report Presentation")
    st.success("🎉 User Receives Report/Visualizations")
    
    if st.session_state.report:
        st.markdown(st.session_state.report, unsafe_allow_html=True)
    
    st.divider()
    # Option to restart
    if st.button("Start a New Analysis"):
        st.session_state.clear()
        st.experimental_rerun()

# --- Main App Flow Control ---

if st.session_state.step == "upload":
    render_upload_step()
elif st.session_state.step == "eda":
    render_eda_step()
elif st.session_state.step == "goal":
    render_goal_step()
elif st.session_state.step == "planning":
    render_planning_step()
elif st.session_state.step == "visualization":
    render_visualization_step()
elif st.session_state.step == "report_presentation":
    render_report_step()

st.sidebar.markdown("## Agent State")
st.sidebar.markdown(f"**Current Step:** `{st.session_state.step.upper()}`")
st.sidebar.markdown(f"**Data Loaded:** {'Yes' if st.session_state.data is not None else 'No'}")
st.sidebar.markdown(f"**Goal Set:** {'Yes' if st.session_state.user_goal is not None else 'No'}")
