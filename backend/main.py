"""
main.py — FastAPI Application for AutoAnalyst
"""

import time
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml import (
    run_data_sanity_audit,
    generate_eda_charts,
    train_and_evaluate_drivers,
    compile_executive_report
)

app = FastAPI(title="AutoAnalyst API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS = {}

# ── Built-in Business Datasets ────────────────────────────────────────────────

def generate_churn_dataset() -> pd.DataFrame:
    """Synthetic Customer Churn dataset (Classification). Includes missing values & outliers."""
    np.random.seed(42)
    n = 1000
    tenure = np.random.randint(1, 72, size=n)
    charges = np.round(np.clip(np.random.normal(65, 30, size=n), 19.99, 120.00), 2)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"],
                                size=n, p=[0.55, 0.20, 0.25])
    calls = np.random.poisson(lam=1.5, size=n).astype(float)
    calls[np.random.rand(n) < 0.08] = np.nan          # inject missing values
    charges[np.random.choice(n, 20, replace=False)] = 250.00  # inject outliers
    p = np.clip(0.15 + 0.35*(contract == "Month-to-month")
                + 0.10*(calls >= 3) - 0.004*tenure, 0.02, 0.98)
    churn = np.where(np.random.binomial(1, np.nan_to_num(p, nan=0.3)), "Yes", "No")
    return pd.DataFrame({
        "Tenure_Months": tenure,
        "Monthly_Charges": charges,
        "Contract_Type": contract,
        "Support_Calls": calls,
        "Churned": churn
    })

def generate_sales_dataset() -> pd.DataFrame:
    """Synthetic Regional Sales dataset (Regression). Includes missing discount values."""
    np.random.seed(101)
    n = 800
    spend = np.round(np.random.exponential(scale=15000, size=n) + 2000, 2)
    reps  = np.random.randint(1, 15, size=n)
    leads = np.clip(np.round(spend * 0.045 + np.random.normal(50, 20, size=n)), 5, None).astype(int)
    discount = np.where(np.random.rand(n) < 0.05, np.nan,
                        np.random.uniform(0.0, 0.30, size=n))
    revenue = np.round(np.clip(
        (leads * 125) + (reps * 1200) - (np.nan_to_num(discount, nan=0.1) * 8000)
        + np.random.normal(1000, 500, size=n), 500, None), 2)
    return pd.DataFrame({
        "Marketing_Spend": spend,
        "Sales_Reps_Count": reps,
        "Leads_Generated": leads,
        "Average_Discount": discount,
        "Sales_Revenue": revenue
    })

SESSIONS["churn"] = generate_churn_dataset()
SESSIONS["sales"] = generate_sales_dataset()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/datasets")
def list_datasets():
    return [
        {
            "id": "churn",
            "name": "Customer Churn Performance",
            "type": "classification",
            "description": "Identify which operational variables drive customer attrition. Includes missing values and outliers.",
            "target": "Churned"
        },
        {
            "id": "sales",
            "name": "Regional Sales Channels",
            "type": "regression",
            "description": "Analyse marketing spend, lead generation, and discount ratios to identify revenue drivers.",
            "target": "Sales_Revenue"
        }
    ]

@app.get("/datasets/{session_id}/columns")
def get_columns(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Dataset not found.")
    df = SESSIONS[session_id]
    return {
        "columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist()
    }

@app.post("/analyze/upload")
async def upload_file(file: UploadFile = File(...)):
    name = file.filename
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {e}")
    if df.empty:
        raise HTTPException(400, "Uploaded file is empty.")
    sid = f"upload_{int(time.time())}"
    SESSIONS[sid] = df
    return {"session_id": sid, "columns": df.columns.tolist(), "rows": len(df), "name": name}

class AnalyzeRequest(BaseModel):
    session_id: str
    target_column: str

@app.post("/analyze/run")
def run_analysis(req: AnalyzeRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, "Session not found. Please re-upload your dataset.")
    df = SESSIONS[req.session_id]
    if req.target_column not in df.columns:
        raise HTTPException(400, f"Column '{req.target_column}' not found in dataset.")

    t0 = time.time()
    sanity              = run_data_sanity_audit(df, target_col=req.target_column)
    corr_chart, dist_chart = generate_eda_charts(df, target_col=req.target_column)
    analysis            = train_and_evaluate_drivers(df, target_col=req.target_column)
    report              = compile_executive_report(sanity, analysis, req.target_column)

    return {
        "status": "success",
        "target": req.target_column,
        "runtime_sec": round(time.time() - t0, 3),
        "sanity": sanity,
        "charts": {
            "correlation": corr_chart,
            "distribution": dist_chart,
            "cross_val": analysis["charts"]["cv"],
            "drivers": analysis["charts"]["drivers"],
            "model_eval": (analysis["charts"]["cm"]
                           if analysis["metrics"]["problem_type"] == "classification"
                           else analysis["charts"]["res"])
        },
        "metrics": analysis["metrics"],
        "cross_val": {
            "mean": analysis["cross_val"]["mean"],
            "std":  analysis["cross_val"]["std"],
            "metric": analysis["cross_val"]["metric"]
        },
        "drivers": analysis["drivers"],
        "report": report
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
