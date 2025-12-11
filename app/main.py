from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Optional
from datetime import datetime
import os

app = FastAPI(title="Sample Analytics Backend (Demo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000","http://192.168.8.124:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sample_sales.csv")  # default CSV in same folder

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df

@app.get("/kpis")
def kpis():
    """
    Returns simple KPIs: total_revenue, total_orders, avg_order_value, total_customers
    """
    try:
        df = load_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    total_revenue = float(df["revenue"].sum())
    total_orders = int(df["order_id"].nunique())
    avg_order_value = float(df["revenue"].sum() / df["order_id"].nunique())
    total_customers = int(df["customer_id"].nunique())
    country_counts = df["country"].value_counts().to_dict()

    return {"total_revenue": total_revenue, "total_orders": total_orders, "avg_order_value": avg_order_value, "total_customers": total_customers, "country_counts": country_counts}


@app.get("/timeseries")
def timeseries(metric: str = "revenue", from_date: Optional[str] = None, to_date: Optional[str] = None):
    """
    Returns a timeseries aggregated by day for a given metric.
    metric: 'revenue' or 'orders'
    """
    try:
        df = load_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if from_date:
        df = df[df["date"] >= pd.to_datetime(from_date)]
    if to_date:
        df = df[df["date"] <= pd.to_datetime(to_date)]
    if metric == "revenue":
        ts = df.groupby(df["date"].dt.date)["revenue"].sum().reset_index()
        labels = ts["date"].astype(str).tolist()
        values = ts["revenue"].round(2).tolist()
    elif metric == "orders":
        ts = df.groupby(df["date"].dt.date)["order_id"].nunique().reset_index()
        labels = ts["date"].astype(str).tolist()
        values = ts["order_id"].astype(int).tolist()
    else:
        raise HTTPException(status_code=400, detail="Unsupported metric")
    return {"metric": metric, "labels": labels, "values": values}

@app.get("/breakdown")
def breakdown(by: str = "product", top_n: int = 10):
    """
    Returns a breakdown (group by) for a categorical column.
    by: 'product' | 'category' | 'country' | 'device'
    """
    try:
        df = load_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if by not in ["product", "category", "country", "device"]:
        raise HTTPException(status_code=400, detail="Unsupported breakdown column")
    br = df.groupby(by)["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(top_n)
    items = br.to_dict(orient="records")
    return {"by": by, "items": items}


@app.get("/devices/pie")
def devices_pie():
    """
    Returns device distribution as counts and percentage share for pie chart.
    """
    try:
        df = load_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "device" not in df.columns:
        raise HTTPException(status_code=400, detail="Column 'device' not found in data")

    counts_series = df["device"].value_counts()
    labels = counts_series.index.tolist()
    counts = counts_series.tolist()
    total = int(counts_series.sum())
    percentages = [round((c / total) * 100, 2) if total else 0 for c in counts]

    subtitle = f"{total} orders across {len(labels)} devices"

    return {
        "labels": labels,
        "counts": counts,
        "percentages": percentages,
        "total": total,
        "subtitle": subtitle,
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Accepts a CSV upload and saves it to disk as 'uploaded_data.csv'.
    For demo purposes, this will overwrite the existing sample.
    """
    contents = await file.read()
    try:
        df = pd.read_csv(pd.compat.StringIO(contents.decode("utf-8")), parse_dates=["date"])
    except Exception:
        # fallback: try to write file as-is
        path = "uploaded_data.csv"
        with open(path, "wb") as f:
            f.write(contents)
        return {"detail": f"Saved uploaded file to {path}. Could not parse with pandas."}
    # save parsed DataFrame
    path = "uploaded_data.csv"
    df.to_csv(path, index=False)
    return {"detail": f"Uploaded and saved to {path}", "rows": len(df)}
