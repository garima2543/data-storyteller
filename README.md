# 📊 Data Storyteller

An interactive ecommerce data analysis dashboard built with Streamlit.
Upload any CSV and instantly get summaries, correlations, missing-value analysis, and an AI-written narrative.

## Features

- 📂 Drag-and-drop CSV upload (or load demo dataset)
- 📖 AI-generated data narrative powered by Claude
- 💡 Auto insights with severity tagging
- 📊 Interactive charts — numeric summary, missing values, correlations, categorical breakdown
- 🔬 Scatter explorer and raw data preview

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/data-storyteller.git
cd data-storyteller
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 4. Run
```bash
python -m streamlit run app.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).
Set `ANTHROPIC_API_KEY` as a secret in the Streamlit Cloud dashboard (no `.env` file needed there).

## Project Structure

```
├── app.py                 # Streamlit dashboard
├── data_storyteller.py    # Core analysis functions
├── requirements.txt       # Python dependencies
├── .env                   # Your API key (not committed)
└── .gitignore
```
