# AI Data Analyst — Insights for Everyone

> Streamlit AI Agent that ingests a CSV, performs cleaning & EDA, runs AI-assisted analysis & visualization, and produces downloadable reports — with a one-click “Run Full Analysis” mode.

This repository contains a production-ready, extensible Streamlit application that implements an autonomous **AI Data Analysis Agent**. It focuses on taking user CSVs through a full pipeline (ingest → clean → EDA → user goal → AI analysis → visualizations → report), with quality-of-life options such as silent visualization error handling, optional OpenAI summarization, and a configurable workflow diagram.

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

* Upload CSV files (with basic file-type & size validation).
* Automated cleaning and preprocessing (missing values, type coercion, datetime detection).
* Exploratory Data Analysis: summary stats, missingness heatmap, correlation matrix.
* AI-assisted analysis & insights (rule-based heuristics + optional OpenAI summarization).
* Visualization recommendations and interactive Plotly charts.
* “Run Full Analysis” — single-click pipeline execution.
* Exportable report (HTML / TXT download).
* Configurable flow diagram (load from repository raw image URL) shown full-width.
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
# .venv\Scripts\activate   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py         # or app1.py depending on which file you're using
```

Open the link printed by Streamlit (usually `http://localhost:8501`) and use the UI to upload a CSV and run the AI Agent.

---

## Requirements

Minimum tested versions (see `requirements.txt` file):

```
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.2.0
Pillow
requests
openai    # optional — only if you enable OpenAI summarization
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Install & run (detailed)

1. Clone this repo:

   ```bash
   git clone https://github.com/Shreyas521032/AI-Data-Analyst-Insights-for-everyone.git
   cd AI-Data-Analyst-Insights-for-everyone
   ```

2. Create a Python virtual environment and install:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

   or if your main file is named `app1.py`:

   ```bash
   streamlit run app1.py
   ```

---

## Configuration & common options

* **OpenAI summarization**

  * Optional: the app supports OpenAI-based natural-language summaries. Provide your API key in the sidebar (`OpenAI API Key`) and enable the checkbox to use it.
  * If you don’t want to use OpenAI, leave it blank — the agent will use rule-based summaries.

* **Flow diagram**

  * The flow diagram (tab 4) loads a raw image from a URL in the code (`DEFAULT_FLOW_URL`) or can be configured via the UI (if enabled).
  * Use a raw GitHub content URL, e.g.:

    ```
    https://raw.githubusercontent.com/<username>/<repo>/<branch>/path/to/flow_diagram.png
    ```

* **Upload size limit**

  * Default set to ~30 MB. Change the `MAX_UPLOAD_MB` constant in the app if needed.

* **Silent visualization errors**

  * Toggle in the sidebar: if enabled, plotting errors are suppressed and user won’t see Streamlit’s red tracebacks. All exceptions are logged to server logs for debugging.

---

## App layout & tabs

Typical app structure:

* **Upload CSV** — validate file format & size; preview.
* **Cleaning & Preprocessing** — remove sparse columns, detect datetimes, provide cleaning log.
* **EDA** — numeric summary, correlation matrix, missingness heatmap, chart recommendations.
* **AI Analysis** — user enters a goal; agent checks for ambiguity; runs anomaly detection & quick ML checks; suggests visualizations.
* **Visualization** — sample Plotly charts (line, scatter, bar, histogram, box).
* **Report** — generates HTML/TXT report and download link.
* **Workflow Architecture** (Tab 4) — full-width flow diagram loaded from raw GitHub image (cached).

---

## How the AI Agent works (overview)

1. **Ingest**: streamlit file uploader → `pd.read_csv()` with robust fallback separators.
2. **Preprocess**: drop extremely sparse columns, convert numeric strings, detect datetime columns.
3. **EDA**: `df.describe()`, numeric distributions, correlation matrix, missingness visualization.
4. **Goal handling**: detect ambiguous goals via heuristics; ask for clarifications or autoswitch to default plan.
5. **AI Analysis**: rule-based insights + optional `openai.ChatCompletion` summarization (if key provided).
6. **Visualize**: recommend charts based on data & goal; use `plotly.express` and safe-plot wrapper to handle exceptions silently.
7. **Report**: assemble HTML using a safe builder (avoid raw f-strings with `{}` in CSS), write or serve as download.

---

## Troubleshooting & tips

* **Raw Plotly errors appear in UI**: Update to the safe visualization functions — the repository includes `safe_plot` logic to catch plotting exceptions. Also enable the sidebar toggle “Suppress visualization error messages (silent mode)”.
* **Emoji / non-ASCII SyntaxError**: If you see `SyntaxError: invalid character` (e.g. heart emoji), replace emojis in Python source with HTML entities (e.g. `&#10084;`) or save the file encoded as UTF-8 and add `# -*- coding: utf-8 -*-` at the top.
* **Flow diagram not loading**: Ensure you use the **raw** GitHub URL (`raw.githubusercontent.com/...`). If the file is private, host the image in a public repo or supply a public URL.
* **Large CSVs**: For very large datasets, sample the data for EDA or implement chunked processing. Consider storing raw data externally (S3, GCS) and reading relevant slices.
* **OpenAI errors**: If OpenAI summarization fails, check your key and usage quotas. The app catches failures and falls back to non-OpenAI summaries.

---

## Extending the app

* Add classification/regression workflows with cross-validation and hyperparameter tuning (e.g., `scikit-learn`, `xgboost`).
* Add time-series forecasting (Prophet, pmdarima, or deep learning).
* Generate PDF reports (weasyprint, reportlab, or headless Chromium via `pyppeteer`).
* Add authentication & cloud deployment: deploy to Streamlit Cloud, Heroku, or as a container on AWS/GCP/Azure.
* Add a task queue or worker for heavy analyses (Celery/RQ) to avoid blocking the Streamlit server.

---

## Contributing

Contributions welcome! Steps:

1. Fork the repo
2. Create a branch for your feature/fix: `git checkout -b feat/my-feature`
3. Implement and test changes
4. Open a pull request with a clear description and screenshots (if applicable)

Please follow the style of the project and add tests where appropriate.

---

## License

This repository is provided under the **MIT License**. Add a `LICENSE` file with MIT license text when you publish.

---

## Contact

Made with ❤️ by **Shreyas Kasture**
Repository: [https://github.com/Shreyas521032/AI-Data-Analyst-Insights-for-everyone](https://github.com/Shreyas521032/AI-Data-Analyst-Insights-for-everyone)

If you'd like, I can:

* Create `README.md` as a file and a PR-ready patch.
* Add badges (build, license, python versions).
* Draft `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, or `LICENSE` files.

Which one should I add next?
