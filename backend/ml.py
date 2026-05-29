"""
ml.py — Core Analytical Services for AutoAnalyst
Handles Data Sanity Audits, Dynamic EDA, ML Modeling, Metric Evaluations, and Executive Summaries.
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

# ── Matplotlib Helpers ────────────────────────────────────────────────────────

def fig2b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{data}"

def safe_serialize(obj):
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_serialize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            cat_cols
        ))
    return ColumnTransformer(transformers)

# ── Data Sanity Auditor ───────────────────────────────────────────────────────

def run_data_sanity_audit(df, target_col=None):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    missing = {}
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        if cnt > 0:
            missing[col] = {"count": cnt, "pct": round(cnt / len(df) * 100, 2)}

    outliers = {}
    for col in num_cols:
        if col == target_col:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        cnt = int(mask.sum())
        if cnt > 0:
            outliers[col] = {"count": cnt, "pct": round(cnt / len(df) * 100, 2)}

    stats = df[num_cols].describe().round(3).to_dict()

    return safe_serialize({
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "missing": missing,
        "outliers": outliers,
        "statistics": stats
    })

# ── EDA Visualizations ────────────────────────────────────────────────────────

def generate_eda_charts(df, target_col=None):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_img = ""
    if len(num_cols) >= 2:
        corr_cols = num_cols[:10]
        fig, ax = plt.subplots(figsize=(max(5, len(corr_cols)), max(4, len(corr_cols) - 1)))
        sns.heatmap(df[corr_cols].corr().round(2), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
        ax.set_title("Operational Metrics Correlation Matrix", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        corr_img = fig2b64(fig)

    dist_img = ""
    plot_cols = [c for c in num_cols if c != target_col][:6]
    if plot_cols:
        ncols = min(3, len(plot_cols))
        nrows = (len(plot_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten() if len(plot_cols) > 1 else [axes]
        for i, col in enumerate(plot_cols):
            d = df[col].dropna()
            axes[i].hist(d, bins=25, color="#3b82f6", edgecolor="white", alpha=0.85)
            axes[i].axvline(d.mean(), color="red", linestyle="--", linewidth=1.2,
                            label=f"Mean={d.mean():.1f}")
            axes[i].set_title(f"Distribution: {col}", fontsize=10, fontweight="bold")
            axes[i].legend(fontsize=7)
            axes[i].grid(True, linestyle=":", alpha=0.6)
        for j in range(len(plot_cols), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Distribution of Key Numerical Metrics", fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        dist_img = fig2b64(fig)

    return corr_img, dist_img

# ── ML Training & Driver Extraction ──────────────────────────────────────────

def train_and_evaluate_drivers(df, target_col):
    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    is_classification = False
    if y.dtype == "object" or str(y.dtype) in ["category", "bool"] or y.nunique() <= 10:
        is_classification = True

    if is_classification:
        y = y.astype(str)
        classes = np.unique(y).tolist()
    else:
        classes = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if is_classification else None
    )

    preprocessor = build_preprocessor(X_train)
    model = (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
             if is_classification else
             RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {}
    cm_img = ""
    res_img = ""

    if is_classification:
        metrics["problem_type"] = "classification"
        metrics["accuracy"]  = round(accuracy_score(y_test, y_pred), 4)
        metrics["precision"] = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        metrics["recall"]    = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        metrics["f1"]        = round(f1_score(y_test, y_pred, average="weighted"), 4)
        metrics["report"]    = classification_report(y_test, y_pred, target_names=classes,
                                                     output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(max(5, len(classes)), max(4, len(classes) - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("Actual Label", fontweight="bold")
        ax.set_title("Model Confusion Matrix Evaluation", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        cm_img = fig2b64(fig)
    else:
        metrics["problem_type"] = "regression"
        metrics["r2"]   = round(r2_score(y_test, y_pred), 4)
        metrics["mae"]  = round(mean_absolute_error(y_test, y_pred), 4)
        metrics["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(y_test, y_pred, alpha=0.5, color="#2563eb", s=20)
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal Fit")
        axes[0].set_xlabel("Actual Value", fontweight="bold")
        axes[0].set_ylabel("Predicted Value", fontweight="bold")
        axes[0].set_title(f"Actual vs Predicted (R²={metrics['r2']})", fontsize=10, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.5)
        residuals = y_test - y_pred
        axes[1].hist(residuals, bins=30, color="#2563eb", edgecolor="white", alpha=0.85)
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Residual Error", fontweight="bold")
        axes[1].set_title("Residual Distribution", fontsize=10, fontweight="bold")
        axes[1].grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        res_img = fig2b64(fig)

    cv_metric = "accuracy" if is_classification else "r2"
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring=cv_metric, n_jobs=-1)
    cv_mean = round(float(cv_scores.mean()), 4)
    cv_std  = round(float(cv_scores.std()), 4)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar([f"Fold {i+1}" for i in range(5)], cv_scores, color="#3b82f6", edgecolor="white", width=0.6)
    ax.axhline(cv_mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean={cv_mean:.3f}")
    ax.set_title(f"5-Fold Cross Validation ({cv_metric.upper()})", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel(cv_metric.upper())
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002, f"{score:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    cv_img = fig2b64(fig)

    feat_img = ""
    drivers = []
    m = pipeline.named_steps["model"]
    if hasattr(m, "feature_importances_"):
        try:
            pre = pipeline.named_steps["preprocessor"]
            feat_names = []
            if "num" in pre.named_transformers_:
                feat_names.extend(pre.named_transformers_["num"].feature_names_in_.tolist())
            if "cat" in pre.named_transformers_:
                ohe = pre.named_transformers_["cat"].named_steps["encoder"]
                cat_raw = pre.named_transformers_["cat"].feature_names_in_
                feat_names.extend(ohe.get_feature_names_out(cat_raw).tolist())
        except Exception:
            feat_names = [f"Feature_{i}" for i in range(len(m.feature_importances_))]

        if len(feat_names) == len(m.feature_importances_):
            driver_df = pd.DataFrame({"feature": feat_names, "importance": m.feature_importances_})
            driver_df["feature"] = driver_df["feature"].str.replace(r"^(num__|cat__)", "", regex=True)
            driver_df = driver_df.sort_values("importance", ascending=False)
            drivers = driver_df.head(10).to_dict(orient="records")

            fig, ax = plt.subplots(figsize=(7, 4.5))
            plot_df = driver_df.head(10).sort_values("importance", ascending=True)
            ax.barh(plot_df["feature"], plot_df["importance"], color="#2563eb", edgecolor="white", height=0.6)
            ax.set_xlabel("Relative Importance Weight", fontweight="bold", fontsize=10)
            ax.set_title("Root Cause Drivers (Feature Importance Ranking)", fontsize=11, fontweight="bold", pad=12)
            ax.grid(True, axis="x", linestyle=":", alpha=0.5)
            plt.tight_layout()
            feat_img = fig2b64(fig)

    return safe_serialize({
        "metrics": metrics,
        "cross_val": {"scores": cv_scores.tolist(), "mean": cv_mean, "std": cv_std, "metric": cv_metric},
        "drivers": drivers,
        "charts": {"cv": cv_img, "cm": cm_img, "res": res_img, "drivers": feat_img}
    })

# ── Executive Report Compiler ─────────────────────────────────────────────────

def compile_executive_report(sanity, analysis, target_col):
    rows = sanity["shape"]["rows"]
    cols = sanity["shape"]["cols"]
    ptype = analysis["metrics"]["problem_type"]

    sanity_txt = (f"**Data Sanity Audit:** The dataset contains **{rows:,} records** across **{cols} parameters**. ")
    missing_keys = list(sanity["missing"].keys())
    outlier_keys = list(sanity["outliers"].keys())
    if not missing_keys and not outlier_keys:
        sanity_txt += "Quality scan indicates high data integrity — zero missing values or statistical outliers detected."
    else:
        issues = []
        if missing_keys:
            top = sorted(sanity["missing"].items(), key=lambda x: x[1]["count"], reverse=True)[0]
            issues.append(f"missing values detected (notably **{top[0]}** at {top[1]['pct']}% null)")
        if outlier_keys:
            top = sorted(sanity["outliers"].items(), key=lambda x: x[1]["count"], reverse=True)[0]
            issues.append(f"statistical outliers in **{top[0]}** ({top[1]['pct']}% of records)")
        sanity_txt += "Concerns identified: " + " and ".join(issues) + ". Imputation and scaling are recommended before reporting."

    drivers = analysis["drivers"]
    if drivers:
        driver_txt = (f"**Root-Cause Driver Analysis:** Targeting **{target_col}**, the predictive engine identifies "
                      f"**{drivers[0]['feature']}** as the primary operational driver "
                      f"({drivers[0]['importance']*100:.1f}% weight)")
        if len(drivers) > 1:
            driver_txt += f", followed by **{drivers[1]['feature']}** ({drivers[1]['importance']*100:.1f}%)"
        driver_txt += f". Optimizing these variables offers the highest leverage to shift **{target_col}**."
    else:
        driver_txt = f"**Root-Cause Analysis:** Driver extraction was completed for **{target_col}**. Ensure numeric feature columns are present for importance rankings."

    cv_mean  = analysis["cross_val"]["mean"]
    cv_metric = analysis["cross_val"]["metric"].upper()
    score_val = (f"Weighted F1 of {analysis['metrics']['f1']*100:.1f}%"
                 if ptype == "classification"
                 else f"R² of {analysis['metrics']['r2']:.3f}")

    action_txt = (f"**Decision Support:** The model achieved a **{score_val}** with a 5-Fold CV "
                  f"score of **{cv_mean:.3f}** ({cv_metric}), confirming reliable predictive stability. ")
    if drivers:
        action_txt += (f"**Recommendation:** Prioritise operational interventions on "
                       f"**{drivers[0]['feature']}** and set automated drift alerts to maintain metric integrity.")
    else:
        action_txt += "Conduct regular planning reviews against key performance metrics."

    return {
        "sanity_briefing": sanity_txt,
        "root_cause_analysis": driver_txt,
        "operational_strategy": action_txt
    }
