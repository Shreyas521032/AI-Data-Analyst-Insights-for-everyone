# AI Data Analyst — Insights for Everyone

This repository contains a production-ready, extensible Streamlit application that implements an autonomous **AI Data Analysis Agent**. It takes user CSVs through a full pipeline (ingest → clean → EDA → user goal → AI analysis → visualizations → report), and includes safety/UX features such as silent visualization error handling, optional Gemini/OpenAI summarization, and a configurable workflow diagram.

---

## Table of contents

* [Features](#features)
* [Quick demo](#quick-demo)
* [Requirements](#requirements)
* [Install & run](#install--run)
* [Configuration & common options](#configuration--common-options)
* [App layout & tabs](#app-layout--tabs)
* [How the AI Agent works (overview)](#how-the-ai-agent-works-overview)
* [Troubleshooting & tips](#troubleshooting--tips)
* [Extending the app](#extending-the-app)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Features

* Upload CSV files (with file-type & size validation).
* Automated cleaning and preprocessing (missing values, type coercion, datetime detection).
* Exploratory Data Analysis: summary stats, missingness heatmap, correlation matrix.
* AI-assisted analysis & insights (rule-based heuristics + optional Gemini/OpenAI summarization).
* Visualization recommendations and interactive Plotly charts.
* “Run Full Analysis” — single-click pipeline execution.
* Exportable report (HTML / TXT / MD download), with charts embedded as PNGs.
* Configurable flow diagram (load from GitHub raw image URL) shown full-width.
* Silent/safe visualization mode: catches Plotly/plotting errors and hides raw tracebacks from UI.
* Caching for remote images and fast UX.
* Extensible — add ML modules (classification, forecasting), PDF reports, or cloud connectors.

---

## Quick demo

```bash
# Clone repo
git clone https://github.com/Shreyas521032/AI-Data-Analyst-Insights-for-everyone.git
cd AI-Data-Analyst-Insights-for-everyone

# Create venv (recommended)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv/Scripts/activate    # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py        # or streamlit run app1.py depending on the main file
```

Open the link printed by Streamlit (usually `http://localhost:8501`) and use the UI to upload a CSV and run the AI Agent.

---

## Requirements

The project uses the following Python packages (see `requirements.txt`):

```
streamlit
pandas
plotly
plotly-express
numpy
scikit-learn
requests
Pillow
google-generativeai    # Gemini integration
anthropic               # optional 
```

Install everything with:

```bash
pip install -r requirements.txt
```

> Note: `google-generativeai` (Gemini) requires credentials and quota from your Google Cloud/Generative AI project. If you don't have access, the app will still run using the rule-based analysis pipeline and Plotly visualizations.

---

## Install & run (detailed)

1. Clone this repo:

   ```bash
   git clone https://github.com/Shreyas521032/AI-Data-Analyst-Insights-for-everyone.git
   cd AI-Data-Analyst-Insights-for-everyone
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Configure the Gemini API key (if you want AI summaries):

   * Open the app sidebar and paste your Gemini API key in the `Gemini API Key` field.

---

## App layout & tabs

* **📤 Data Ingestion** — Upload CSV, preprocessing, quick EDA and previews.
* **⚙️ AI Analysis** — Set analysis objective, run Gemini-powered deep analysis and get recommendations.
* **📊 Results & Insights** — Custom visualization builder, export cleaned data and reports.
* **🔄 Agent Workflow** — Visual workflow diagram and architecture overview.

---

## How the AI Agent works (overview)

1. **Ingest**: Streamlit file uploader → `pd.read_csv()` with fallbacks.
2. **Preprocess**: drop extremely sparse columns, convert numeric strings, detect datetime columns, fill missing values with median/mode.
3. **EDA**: `df.describe()`, numeric distributions, correlation matrix, missingness visualization.
4. **Goal handling**: detect ambiguous goals via heuristics; ask for clarifications or auto-plan.
5. **AI Analysis**: call Gemini (if configured) with dataset context and user objective to obtain JSON visualization recommendations and insights.
6. **Visualize**: generate recommended charts safely with Plotly, embed PNG snapshots into HTML report.
7. **Report**: assemble HTML (safe string building) and serve as downloadable HTML/MD/CSV files.

---

## Troubleshooting & tips

* **Plotly tracebacks in UI**: If you see raw Python tracebacks in Streamlit, update to the safe plotting wrapper included in the code. This logs exceptions and prevents red error boxes from appearing.
* **Gemini API errors**: If generative calls fail, check your API key, network, and quota. The app falls back to rule-based insights if AI calls error out.
* **Large datasets**: For large CSVs, sample the data for EDA or increase memory limits. Consider adding chunked processing or offloading data to cloud storage.
* **Flow image not loading**: Use the raw GitHub URL (`raw.githubusercontent.com/...`) and ensure the file path is correct and publicly accessible.

---

## Contributing

Contributions are welcome!

1. Fork the repo
2. Create a branch for your feature/fix: `git checkout -b feat/my-feature`
3. Implement and test changes
4. Open a pull request with a clear description and screenshots (if applicable)

Please follow the project style and add tests where appropriate.

---

## License

This repository is provided under the **MIT License**. Add a `LICENSE` file with MIT license text when publishing.

---

## Contact

Made with ❤️ by **Shreyas Kasture**
Repository: [https://github.com/Shreyas521032/AI-Data-Analyst-Insights-for-everyone](https://github.com/Shreyas521032/AI-Data-Analyst-Insights-for-everyone)

---

