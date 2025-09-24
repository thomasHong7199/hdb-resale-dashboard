# app.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="HDB Resale Dashboard", layout="wide")

# -------------------------
# Data loading
# -------------------------
@st.cache_data(show_spinner="Loading data…")
def load_data():
    """
    Try small committed files first (for Streamlit Cloud),
    then let user upload a file.
    """
    candidates = [
        "Data/sample_resale.csv",          # <-- small sample (recommended for cloud)
        "data/sample_resale.csv",
        "Data/hdb_resale_cleaned.csv",
        "data/hdb_resale_cleaned.csv",
        "Data/resale2017data_cleaned.csv",
        "data/resale2017data_cleaned.csv",
    ]

    tried = []
    for p in candidates:
        tried.append(p)
        if os.path.exists(p):
            st.success(f"Loaded local file: `{p}`")
            return pd.read_csv(p)

    st.info("No local CSV found. Upload a CSV to continue.")
    file = st.file_uploader("Upload HDB resale CSV", type=["csv"])
    if file:
        st.success("Loaded uploaded file.")
        return pd.read_csv(file)

    st.error("No data file found. Looked for:\n" + "\n".join(f"- {p}" for p in tried))
    st.stop()

# -------------------------
# Helpers
# -------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df

def ensure_price_psm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # common variants
    price = "resale_price" if "resale_price" in df.columns else None
    area = (
        "floor_area_sqm" if "floor_area_sqm" in df.columns
        else ("floor_area" if "floor_area" in df.columns else None)
    )
    if "price_psm" not in df.columns and price and area:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["price_psm"] = pd.to_numeric(df[price], errors="coerce") / \
                              pd.to_numeric(df[area], errors="coerce")
    return df

def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # accept month like "2019-05" or full date
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    elif "resale_registration_date" in df.columns:
        df["month"] = pd.to_datetime(df["resale_registration_date"], errors="coerce")
    return df

# -------------------------
# Main
# -------------------------
st.title("🏠 HDB Resale Dashboard")

df = load_data()
df = normalize_columns(df)
df = ensure_price_psm(df)
df = coerce_dates(df)

# ---- KPIs
st.subheader("📊 Key Metrics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    mean_price = df["resale_price"].mean() if "resale_price" in df.columns else np.nan
    st.metric("Average Price (SGD)", f"{mean_price:,.0f}" if pd.notna(mean_price) else "N/A")
with c2:
    med_psm = df["price_psm"].median() if "price_psm" in df.columns else np.nan
    st.metric("Median Price / sqm", f"{med_psm:,.0f}" if pd.notna(med_psm) else "N/A")
with c3:
    max_price = df["resale_price"].max() if "resale_price" in df.columns else np.nan
    st.metric("Highest Transaction", f"{max_price:,.0f}" if pd.notna(max_price) else "N/A")
with c4:
    st.metric("Total Transactions", f"{len(df):,}")

# ---- Preview
st.subheader("📋 Dataset Preview")
st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
st.dataframe(df.head(20), use_container_width=True)

st.divider()

# ---- Charts
# 1) Trend line
st.subheader("📈 Average Resale Price Over Time")
if "month" in df.columns and "resale_price" in df.columns:
    trend = (
        df.dropna(subset=["month"])
          .groupby("month", as_index=False)["resale_price"]
          .mean()
          .sort_values("month")
    )
    fig = px.line(trend, x="month", y="resale_price",
                  labels={"month":"Month", "resale_price":"Avg Price (SGD)"},
                  title="Average Resale Price Over Time")
    st.plotly_chart(fig, use_container_width=True, key="trend_line")
else:
    st.info("Need columns 'month' and 'resale_price' to plot the trend.")

# 2) Donut by flat type (total value)
st.subheader("🍩 Market Share by Flat Type (Total Value)")
if {"flat_type", "resale_price"}.issubset(df.columns):
    share = (df.groupby("flat_type", as_index=False)["resale_price"]
               .sum()
               .sort_values("resale_price", ascending=False))
    fig = px.pie(share, names="flat_type", values="resale_price", hole=0.45)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True, key="donut_flat_type")
else:
    st.info("Need columns 'flat_type' and 'resale_price' to draw donut chart.")

# 3) Top towns by avg $/sqm
st.subheader("📍 Top Towns by Avg $/sqm")
if {"town", "price_psm"}.issubset(df.columns):
    by_town = (df.groupby("town", as_index=False)["price_psm"]
                 .mean()
                 .sort_values("price_psm", ascending=False)
                 .head(20))
    fig = px.bar(by_town, x="price_psm", y="town",
                 labels={"price_psm":"Avg $/sqm", "town":"Town"},
                 title="Top 20 Towns by Avg $/sqm")
    st.plotly_chart(fig, use_container_width=True, key="town_bar_psm")
else:
    st.info("Need columns 'town' and 'price_psm' to plot town comparison.")
