# AI Founder Ops Copilot

An AI-powered finance intelligence web app designed for early-stage startups. Upload raw transaction CSVs or PDF invoices and get instant financial clarity — automated data cleaning, anomaly detection, live KPI dashboards, and AI-generated insights.

!\[Dashboard Screenshot](screenshots/dashboard.png)

## What It Does

* **Data Ingestion** — Upload raw transaction CSVs and PDF invoices
* **Automated Cleaning \& Categorisation** — Cleans, deduplicates, and categorises financial data automatically
* **Anomaly Detection** — Flags duplicate invoices, spend spikes, and unusual patterns
* **Live KPI Dashboard** — Tracks revenue, expenses, burn rate, and cash runway in real time
* **AI Insights Engine** — Generates CFO-style variance commentary on your financial data
* **Natural Language Q\&A** — Ask operational questions directly against your own data

## Tech Stack

* **Backend:** Python, FastAPI, Pandas
* **Frontend:** HTML/CSS/JavaScript (single-page app)
* **AI:** Claude API for insights generation and natural language Q\&A
* **Database:** SQLite for processed financial data

## Getting Started

### Prerequisites

* Python 3.9+
* An Anthropic API key (for AI features)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Dhruvika001/ai-founder-ops-copilot.git
cd ai-founder-ops-copilot
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\\Scripts\\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your API key
```

5. Run the app:

```bash
uvicorn main:app --reload
```

6. Open your browser at `http://localhost:8000`

## Project Structure

```
ai-founder-ops-copilot/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── data/                   # Sample data files
│   └── sample\_transactions.csv
├── modules/
│   ├── ingestion.py        # CSV/PDF data ingestion
│   ├── cleaning.py         # Data cleaning \& categorisation
│   ├── anomaly.py          # Anomaly detection logic
│   ├── dashboard.py        # KPI calculations
│   └── insights.py         # AI insights \& Q\&A engine
├── static/                 # Frontend assets
│   ├── index.html
│   ├── style.css
│   └── script.js
└── screenshots/
    └── dashboard.png       # App screenshot for README
```

## Sample Data

A sample transaction CSV is included in the `data/` folder for testing. Upload it through the app to see the full pipeline in action.

## Demo

Built as a prototype for [INDX](https://www.indx.sg), a Singapore-based finance operations firm, to demonstrate how AI and automation can replace manual monthly reporting workflows.

## Author

**Dhruvikaa Agarwal**

* NTU Data Science \& AI, Year 2
* [LinkedIn](www.linkedin.com/in/dhruvikaa-agarwal)
* dhruvika001@e.ntu.edu.sg

