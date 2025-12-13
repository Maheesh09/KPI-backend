# backend_main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # allows cross-origin requests
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import re
import io

app = FastAPI(title="Sample Analytics Backend (Demo)")

# allow your local dev frontends (adjust as needed)
# Support production and development URLs
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000"
    
      # Allow all in production (or specify your frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sample_sales.csv")  # default CSV in same folder
UPLOADED_PATH = os.path.join(BASE_DIR, "uploaded_data.csv")



# Utility: robust CSV loader
def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to detect date-like columns and convert them to datetime dtype.
    """
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(200).tolist()
            date_like = sum(1 for s in sample if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", s))
            if len(sample) > 0 and date_like / len(sample) > 0.5:
                # attempt parse
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
                except Exception:
                    pass
    return df


def load_data(path: str = None) -> pd.DataFrame:
    """
    Load CSV from provided path or default DATA_PATH / UPLOADED_PATH.
    Tries multiple strategies for parsing dates and numeric columns.
    """
    # choose path (uploaded has priority)
    candidate_paths = []
    if path:
        candidate_paths.append(path)
    if os.path.exists(UPLOADED_PATH):
        candidate_paths.append(UPLOADED_PATH)
    if os.path.exists(DATA_PATH):
        candidate_paths.append(DATA_PATH)

    for p in candidate_paths: # try each path
        try:
            # read with pandas; don't force parse_dates to allow flexible parsing later
            df = pd.read_csv(p, dtype=str)  # read as strings initially
            # strip BOM and whitespace from column names
            df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
            # coerce numeric columns where possible
            for col in df.columns:
                # try numeric conversion - remove common formatting
                # Remove currency symbols, commas, spaces, and other common formatting
                cleaned = df[col].astype(str).str.replace(r"[\$\£\€\₹Rs\s,]", "", regex=True).str.strip()
                # Remove any remaining non-numeric characters except decimal point and minus sign
                cleaned = cleaned.str.replace(r"[^\d\.\-]", "", regex=True)
                coerced = pd.to_numeric(cleaned, errors="coerce")
                # if many non-NaN numeric -> replace
                if coerced.notna().sum() / max(1, len(df)) > 0.6:
                    df[col] = coerced
            # try parsing date-like columns
            df = try_parse_dates(df)
            return df
        except Exception:
            continue

    raise FileNotFoundError(f"No data file found. Checked: {candidate_paths}")


# Simple KPI endpoint
@app.get("/kpis")
def kpis(path: Optional[str] = None):
    """
    Returns simple KPIs: total_revenue (if detected), total_orders (if order_id exists),
    avg_order_value, total_customers.
    """
    try:
        df = load_data(path=path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # detect likely revenue column
    revenue_cols = [c for c in df.columns if re.search(r"revenue|price|amount|total|cost", c, re.I)]
    revenue_col = revenue_cols[0] if revenue_cols else None
    order_col = next((c for c in df.columns if re.search(r"order|invoice|txn|id", c, re.I)), None)
    customer_col = next((c for c in df.columns if re.search(r"customer|client", c, re.I)), None)

    total_revenue = None
    avg_order_value = None
    total_orders = None
    total_customers = None

    if revenue_col:
        # Ensure numeric type before summing
        numeric_values = pd.to_numeric(df[revenue_col], errors="coerce").dropna()
        if len(numeric_values) > 0:
            total_revenue = float(numeric_values.sum())
        else:
            total_revenue = None
    if order_col:
        total_orders = int(df[order_col].dropna().nunique())
    if order_col and revenue_col:
        try:
            # Ensure both columns are properly typed
            df_grouped = df.copy()
            df_grouped[revenue_col] = pd.to_numeric(df_grouped[revenue_col], errors="coerce")
            df_grouped = df_grouped.dropna(subset=[order_col, revenue_col])
            if len(df_grouped) > 0:
                avg_order_value = float(df_grouped.groupby(order_col)[revenue_col].sum().mean())
            else:
                avg_order_value = None
        except Exception:
            avg_order_value = None
    if customer_col:
        total_customers = int(df[customer_col].dropna().nunique())

    return {
        "revenue_col": revenue_col,
        "order_col": order_col,
        "customer_col": customer_col,
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "total_customers": total_customers,
    }



# Timeseries endpoint
@app.get("/timeseries")
def timeseries(metric: str = "revenue", from_date: Optional[str] = None, to_date: Optional[str] = None, path: Optional[str] = None):
    """
    Returns a timeseries aggregated by day for a given metric.
    metric: 'revenue' or 'orders' or any numeric column name
    """
    try:
        df = load_data(path=path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # find date-like column
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not date_cols:
        # fallback: find by name
        date_cols = [c for c in df.columns if re.search(r"date|time|day|dt", c, re.I)]
    if not date_cols:
        raise HTTPException(status_code=400, detail="No date-like column found for timeseries.")
    date_col = date_cols[0]

    # parse from/to
    if from_date:
        try:
            from_dt = pd.to_datetime(from_date)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid from_date")
    else:
        from_dt = None
    if to_date:
        try:
            to_dt = pd.to_datetime(to_date)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid to_date")
    else:
        to_dt = None

    # filter
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if from_dt is not None:
        df = df[df[date_col] >= from_dt]
    if to_dt is not None:
        df = df[df[date_col] <= to_dt]

    if metric == "revenue":
        # detect revenue column
        revs = [c for c in df.columns if re.search(r"revenue|price|amount|total|cost", c, re.I)]
        if not revs:
            raise HTTPException(status_code=400, detail="No revenue-like column found")
        rev_col = revs[0]
        # Ensure numeric type
        df_ts = df.copy()
        df_ts[rev_col] = pd.to_numeric(df_ts[rev_col], errors="coerce")
        df_ts = df_ts.dropna(subset=[date_col, rev_col])
        ts = df_ts.groupby(df_ts[date_col].dt.date)[rev_col].sum().reset_index()
        # After reset_index(), the date column is the first column (index 0)
        labels = ts.iloc[:, 0].astype(str).tolist()
        values = ts[rev_col].round(2).tolist()
    elif metric == "orders":
        # detect order id
        order_col = next((c for c in df.columns if re.search(r"order|invoice|txn|id", c, re.I)), None)
        if not order_col:
            raise HTTPException(status_code=400, detail="No order id-like column found")
        ts = df.groupby(df[date_col].dt.date)[order_col].nunique().reset_index()
        # After reset_index(), the date column is the first column (index 0)
        labels = ts.iloc[:, 0].astype(str).tolist()
        values = ts[order_col].astype(int).tolist()
    else:
        # if metric is a numeric column name
        if metric not in df.columns:
            raise HTTPException(status_code=400, detail=f"Metric column '{metric}' not found")
        # Ensure numeric type
        df_ts = df.copy()
        df_ts[metric] = pd.to_numeric(df_ts[metric], errors="coerce")
        df_ts = df_ts.dropna(subset=[date_col, metric])
        ts = df_ts.groupby(df_ts[date_col].dt.date)[metric].sum().reset_index()
        # After reset_index(), the date column is the first column (index 0)
        labels = ts.iloc[:, 0].astype(str).tolist()
        values = ts[metric].round(2).tolist()

    return {"metric": metric, "labels": labels, "values": values}


# Breakdown endpoint
@app.get("/breakdown")
def breakdown(by: str = "product", top_n: int = 10, path: Optional[str] = None):
    """
    Returns a breakdown (group by) for a categorical column.
    by: any column name (product|category|country|device|...)
    """
    try:
        df = load_data(path=path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if by not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{by}' not found in data")

    # try to summarize by revenue if present, else by counts
    revs = [c for c in df.columns if re.search(r"revenue|price|amount|total|cost", c, re.I)]
    if revs:
        # Ensure the revenue column is numeric
        df_agg = df.copy()
        df_agg[revs[0]] = pd.to_numeric(df_agg[revs[0]], errors="coerce")
        df_agg = df_agg.dropna(subset=[by, revs[0]])
        if len(df_agg) > 0:
            agg = df_agg.groupby(by)[revs[0]].sum().reset_index().sort_values(revs[0], ascending=False).head(top_n)
            items = agg.to_dict(orient="records")
            return {"by": by, "metric": revs[0], "items": items}
        else:
            # Fallback to counts if no valid numeric data
            agg = df[by].value_counts().head(top_n).reset_index()
            agg.columns = [by, "count"]
            items = agg.to_dict(orient="records")
            return {"by": by, "metric": "count", "items": items}
    else:
        agg = df[by].value_counts().head(top_n).reset_index()
        agg.columns = [by, "count"]
        items = agg.to_dict(orient="records")
        return {"by": by, "metric": "count", "items": items}



# Infer endpoint (heuristic)
def infer_column_type(series: pd.Series) -> str:
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
    except Exception:
        pass
    non_null = series.dropna().astype(str)
    sample = non_null.head(200).tolist()
    date_like = sum(1 for s in sample if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", s))
    if len(sample) > 0 and date_like / len(sample) > 0.5:
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    name = series.name.lower()
    if any(tok in name for tok in ["price", "amount", "revenue", "total", "cost", "paid"]):
        return "currency"
    if any(re.search(r"[\$\£\€\₹]|lkr\b|rs\b", s.lower()) for s in sample):
        return "currency"
    nunique = series.nunique(dropna=True)
    total_nonnull = len(non_null)
    if total_nonnull == 0:
        return "unknown"
    uniq_ratio = nunique / total_nonnull
    if nunique <= 50 or uniq_ratio < 0.05:
        return "categorical"
    return "text"


def summarize_dataframe_for_infer(df: pd.DataFrame, sample_size: int = 500) -> List[Dict[str, Any]]:
    df_sample = df.head(sample_size)
    cols_meta = []
    for c in df_sample.columns:
        s = df[c]
        ctype = infer_column_type(s)
        cols_meta.append({
            "name": c,
            "type": ctype,
            "n_missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique(dropna=True)),
            "samples": s.dropna().astype(str).head(5).tolist()
        })
    return cols_meta


def recommend_from_meta(meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    date_col = None
    money_col = None
    numeric_cols = []
    cat_cols = []
    for c in meta:
        n = c["name"]
        t = c["type"]
        if t == "datetime" or re.search(r"date|time|day|dt", n, re.I):
            if not date_col:
                date_col = n
        if t in ("currency", "numeric"):
            numeric_cols.append(n)
            if re.search(r"revenue|price|amount|total|cost", n, re.I) and not money_col:
                money_col = n
        if t == "categorical":
            cat_cols.append(n)
    if not money_col and numeric_cols:
        money_col = numeric_cols[0] if numeric_cols else None

    kpis = []
    charts = []
    if money_col:
        kpis.append({"name": f"total_{money_col}", "expr": f"SUM({money_col})", "reason": "monetary column detected"})
    if date_col and money_col:
        charts.append({"type": "line", "title": f"Time series of {money_col}", "x": date_col, "y": money_col, "aggregation": "sum"})
    if cat_cols and money_col:
        charts.append({"type": "bar", "title": f"{money_col} by {cat_cols[0]}", "x": cat_cols[0], "y": money_col, "aggregation": "sum"})
    if numeric_cols:
        charts.append({"type": "histogram", "title": f"Distribution of {numeric_cols[0]}", "x": numeric_cols[0]})
    confidence = 0.8 if date_col and money_col else 0.5 if money_col or date_col else 0.3
    return {"kpis": kpis, "charts": charts, "confidence": confidence, "mappings": {"date": date_col, "money": money_col, "categories": cat_cols}}


@app.get("/infer")
def infer_demo(path: Optional[str] = None):
    """
    Heuristic inference endpoint.
    Reads uploaded_data.csv (preferred) or sample_sales.csv and returns metadata + recommended charts.
    """
    try:
        df = load_data(path=path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    meta = summarize_dataframe_for_infer(df)
    rec = recommend_from_meta(meta)
    return {"rows": len(df), "cols": len(df.columns), "columns": meta, "recommendation": rec}



# Upload endpoint
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Accepts a CSV upload and saves it to disk as 'uploaded_data.csv'.
    Overwrites previous uploaded_data.csv. Tries to parse CSV into a DataFrame.
    """
    contents = await file.read()
    # quick sanity checks
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # try to parse with pandas directly (utf-8)
    try:
        s = contents.decode("utf-8")
        df = pd.read_csv(io.StringIO(s))
    except Exception:
        # fallback: try latin-1
        try:
            s = contents.decode("latin-1")
            df = pd.read_csv(io.StringIO(s))
        except Exception:
            # save raw bytes as uploaded file so user can debug
            with open(UPLOADED_PATH, "wb") as f:
                f.write(contents)
            return {"detail": f"Saved raw upload to {UPLOADED_PATH} but could not parse with pandas."}

    # inspect and save
    df.to_csv(UPLOADED_PATH, index=False)
    return {"detail": f"Uploaded and saved to {UPLOADED_PATH}", "rows": len(df), "cols": len(df.columns)}
