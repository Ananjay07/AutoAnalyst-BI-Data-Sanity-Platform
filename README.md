# AutoAnalyst 📈
> **Automated Business Intelligence, Data Sanity Auditor & Predictive Driver Engine**

**🚀 Live Demo: [https://auto-analyst-bi-data-sanity-platfor.vercel.app/](https://auto-analyst-bi-data-sanity-platfor.vercel.app/)**

AutoAnalyst is a self-serve analytics platform that automates data profiling, validates dataset quality, discovers predictive key drivers, and compiles executive-ready briefings — built with Python (FastAPI, Scikit-Learn, Pandas) and a responsive Glassmorphic vanilla dashboard.

---

## 🚀 Features

| Module | What It Does |
|---|---|
| **Data Sanity Auditor** | Scans uploaded datasets for missing value ratios, statistical outliers (IQR), and data type profiling to ensure day-to-day data integrity |
| **Adaptive ML Engine** | Auto-detects classification vs regression, trains a Random Forest pipeline (imputation → scaling → one-hot encoding → modeling) |
| **Key Driver Extraction** | Extracts feature importances to rank the top operational drivers causing variance in the target business metric |
| **5-Fold Cross Validation** | Validates model stability across data subsets to confirm reliable predictions |
| **Executive Report Compiler** | Translates statistical outputs into a structured 3-paragraph leadership-ready brief |
| **Custom File Upload** | Supports drag-and-drop CSV / Excel uploads alongside built-in preset datasets |

---

## 🏛️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Analysis & ML:** Pandas, NumPy, Scikit-Learn (Random Forest, Pipeline, ColumnTransformer)
- **Visualizations:** Matplotlib, Seaborn (base64-encoded, served dynamically)
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (no frameworks — zero dependency install)

---

## ⚡ Quick Start

### 1. Start the Backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API runs at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

### 2. Open the Dashboard

Simply open `frontend/index.html` in your browser.

---

## 📂 Project Structure

```
AutoAnalyst/
├── README.md
├── backend/
│   ├── main.py          # FastAPI server, routes, dataset loaders
│   ├── ml.py            # Sanity audit, EDA charts, ML pipeline, report compiler
│   └── requirements.txt
└── frontend/
    └── index.html       # Full BI dashboard (HTML, CSS, JS)
```

---

## 📊 Built-in Datasets

| Dataset | Type | Target | Features |
|---|---|---|---|
| **Customer Churn Performance** | Classification | `Churned` | Tenure, Monthly Charges, Contract Type, Support Calls. Includes injected missing values and outliers |
| **Regional Sales Channels** | Regression | `Sales_Revenue` | Marketing Spend, Sales Reps, Leads Generated, Discount. Includes missing discount entries |

Both datasets are **synthetically generated in-memory** on server startup — no external downloads needed.
