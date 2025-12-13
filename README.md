# 📊 Generic Sales Analytics Dashboard – Backend (FastAPI)

## 📌 Project Overview

This project is about **generic analytics dashboard** built using **FastAPI**. It can analyze **most of the sales-related CSV files** and automatically generate data required for dashboards such as:

- KPI cards  
- Time-series charts  
- Category breakdown charts  
- Automatic chart recommendations  

Unlike traditional dashboards that depend on a fixed schema, this system is **dataset-agnostic**.  
It dynamically analyzes the uploaded CSV files and adapts its logic based on detected columns and data types.
This backend is designed to be connected to a **React frontend** for visualization.

---

## 🎯 Project Objectives

- Accept **Sales CSV file** (e-commerce, SaaS, POS, subscriptions, invoices, etc.)
- Automatically detect:
  - Date columns
  - Revenue / monetary columns
  - Categorical columns
- Generate:
  - Business KPIs
  - Time-series data
  - Breakdown data
  - Chart recommendations
- Work without requiring a predefined database schema
- Provide clean REST APIs for frontend dashboards

---

## 🧠 System Workflow (High-Level)

CSV Upload -> Robust CSV Parsing -> Automatic Column Type Detection -> KPI / Timeseries / Breakdown Calculations -> Chart Recommendations -> JSON Responses to Frontend

---

## 🛠️ Technologies Used


**FastAPI** - Backend REST API framework
**Python** - Core programming language
**Pandas** - Data processing and analytics
**Regex** - Column and data type detection
**CORS Middleware** - Frontend-backend communication

---

## 📂 Project Structure

---

## 🔍 Automatic Data Detection

The backend automatically detects and processes:

### 📅 Date Columns
- Identified using regex and sampling
- Converted to `datetime` format
- Required for time-series analysis

### 💰 Revenue / Monetary Columns
Detected using:
- Column names (`revenue`, `price`, `amount`, `total`, `cost`)
- Currency symbols
- Numeric conversion validation

### 🏷️ Categorical Columns
- Low cardinality detection
- Unique value ratio analysis

### 🔢 Numeric Columns
- Safe numeric coercion
- Invalid values are ignored gracefully

---

## Frontend Integration

- Designed for React-based dashboards
- Compatible with:
### Vite
### Next.js

- Works with chart libraries such as:
### Plotly
### Recharts
### Chart.js

---

Maheesha
Computer Science Undergraduate
SLIIT