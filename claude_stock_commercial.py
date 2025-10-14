#!/usr/bin/env python3
"""S&P 500 Stock Analyzer - Complete Working Version with VADER Sentiment"""

import io
from datetime import date, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except:
    VADER_AVAILABLE = False

try:
    from newsapi import NewsApiClient
except:
    NewsApiClient = None

st.set_page_config(page_title="S&P 500 Analyzer", page_icon="📊", layout="wide")

# Google Analytics - Simple direct injection
components.html("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-598BZYJEBM"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-598BZYJEBM');
</script>
""", height=0)

# Force light theme at root level
st.markdown("""
<script>
// Force light theme by overriding Streamlit's theme detection
window.parent.document.documentElement.setAttribute('data-theme', 'light');
</script>
""", unsafe_allow_html=True)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* NUCLEAR OPTION - Force everything to light mode */
* {
  color-scheme: light !important;
}
html, body, #root, .stApp {
  color-scheme: light !important;
  background: #F9FAFB !important;
}
/* Force Streamlit container to light */
[data-testid="stAppViewContainer"] {
  background: #F9FAFB !important;
}
[data-testid="stApp"] {
  background: #F9FAFB !important;
}

html,body,.stApp{font-family:'Inter',sans-serif!important;background:#F9FAFB}

/* Fix top header bar */
header[data-testid="stHeader"]{background:#fff!important}
.stApp > header{background:#fff!important}
button[kind="header"]{color:#1F2937!important}
div[data-testid="stToolbar"]{background:#fff!important}

/* Fix hamburger menu and buttons in header */
button[data-testid="baseButton-header"]{color:#1F2937!important}
button[data-testid="baseButton-header"] svg{color:#1F2937!important;fill:#1F2937!important}
section[data-testid="stSidebarNav"]{background:#fff!important}

/* Fix mobile menu */
div[data-testid="stSidebarNavItems"]{background:#fff!important}
div[data-testid="stSidebarNavItems"] a{color:#1F2937!important}
div[data-testid="stSidebarNavItems"] span{color:#1F2937!important}

h1{font-size:2.5rem!important;font-weight:700!important;background:linear-gradient(135deg,#667eea,#764ba2);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
h2,h3{color:#1F2937!important}

/* Fix selectbox styling - CRITICAL - More specific selectors */
[data-baseweb="select"]{background:#fff!important}
[data-baseweb="select"] > div{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
[data-baseweb="select"] [role="button"]{background:#fff!important;color:#1F2937!important}
[data-baseweb="select"] input{background:#fff!important;color:#1F2937!important}
[data-baseweb="select"] span{color:#1F2937!important}
[data-baseweb="select"] svg{color:#1F2937!important;fill:#1F2937!important}
div[data-baseweb="select"] > div{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
div[data-baseweb="select"] span{color:#1F2937!important}
div[data-baseweb="select"] svg{color:#1F2937!important}
/* Force all select boxes */
.stSelectbox > div > div{background:#fff!important;color:#1F2937!important}
.stSelectbox [data-baseweb="select"]{background:#fff!important}
.stSelectbox label{color:#1F2937!important}

/* Multi-select styling */
.stMultiSelect > div > div{background:#fff!important;color:#1F2937!important}
.stMultiSelect [data-baseweb="select"]{background:#fff!important}
.stMultiSelect label{color:#1F2937!important}
.stMultiSelect span{color:#1F2937!important}
.stMultiSelect [data-baseweb="tag"]{background:#E5E7EB!important;color:#1F2937!important}

/* Number input styling */
.stNumberInput > div > div{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
.stNumberInput input{background:#fff!important;color:#1F2937!important}
.stNumberInput label{color:#1F2937!important}
.stNumberInput button{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
.stNumberInput [data-baseweb="input"]{background:#fff!important}
.stNumberInput [data-baseweb="input"] > div{background:#fff!important}
.stNumberInput [data-baseweb="input"] input{background:#fff!important;color:#1F2937!important}

/* Date input styling */
.stDateInput > div > div{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
.stDateInput input{background:#fff!important;color:#1F2937!important}
.stDateInput label{color:#1F2937!important}

/* Text input styling */
.stTextInput > div > div{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
.stTextInput input{background:#fff!important;color:#1F2937!important}
.stTextInput label{color:#1F2937!important}

/* Checkbox styling */
.stCheckbox{color:#1F2937!important}
.stCheckbox > label{color:#1F2937!important}
.stCheckbox span{color:#1F2937!important}

/* Radio button styling */
.stRadio > label{color:#1F2937!important}
.stRadio [role="radiogroup"]{color:#1F2937!important}
.stRadio [role="radiogroup"] label{color:#1F2937!important}
.stRadio [role="radiogroup"] span{color:#1F2937!important}

/* Slider styling */
.stSlider > label{color:#1F2937!important}
.stSlider [data-baseweb="slider"]{background:#fff!important}

/* Dropdown menu - Multiple selectors to override everything */
[data-baseweb="popover"]{background:#fff!important}
[data-baseweb="popover"] > div{background:#fff!important}
div[data-baseweb="popover"]{background:#fff!important}
ul[role="listbox"]{background:#fff!important}
ul[role="listbox"] li{background:#fff!important;color:#1F2937!important}
ul[role="listbox"] li:hover{background:#F3F4F6!important;color:#1F2937!important}
ul[role="listbox"] span{color:#1F2937!important}
ul[role="listbox"] div{color:#1F2937!important}
/* Target the menu container */
[role="listbox"]{background:#fff!important}
[role="listbox"] *{color:#1F2937!important}
/* Target layers */
div[class*="layer"]{background:transparent!important}
div[class*="Layer"]{background:transparent!important}
/* Selectbox menu */
.stSelectbox [role="listbox"]{background:#fff!important}
.stSelectbox ul{background:#fff!important}
.stSelectbox li{background:#fff!important;color:#1F2937!important}
/* Extra layer targeting */
div[data-baseweb="menu"]{background:#fff!important}
div[data-baseweb="menu"] > div{background:#fff!important}
div[data-baseweb="menu"] ul{background:#fff!important}
div[data-baseweb="menu"] li{background:#fff!important;color:#1F2937!important}

/* Calendar/date picker dropdown */
[data-baseweb="calendar"]{background:#fff!important;color:#1F2937!important}
[data-baseweb="calendar"] *{color:#1F2937!important}

/* Fix the Select Stock label */
label{color:#1F2937!important}
.stSelectbox label{color:#1F2937!important}
.stSelectbox > label{color:#1F2937!important}

/* Fix metrics */
[data-testid="stMetric"]{background:#fff;padding:1rem;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.1);border:1px solid #E5E7EB}
[data-testid="stMetricValue"]{font-size:1.75rem!important;font-weight:700!important;color:#1F2937!important}
[data-testid="stMetricLabel"]{color:#6B7280!important}

/* Fix expander styling */
div[data-testid="stExpander"]{background:#fff!important;border-radius:12px;border:1px solid #E5E7EB;margin-bottom:1rem}
div[data-testid="stExpander"] summary{background:#fff!important;color:#1F2937!important;padding:1rem!important}
div[data-testid="stExpander"] summary:hover{background:#F3F4F6!important}
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"]{background:#fff!important;padding:1rem!important}
div[data-testid="stExpander"] p, div[data-testid="stExpander"] li, div[data-testid="stExpander"] span{color:#1F2937!important}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#667eea,#764ba2)!important;color:#fff!important;
border:none!important;border-radius:8px!important;padding:.5rem 1.5rem!important;font-weight:600!important}

/* Badges */
.badge{display:inline-block;padding:.25rem .75rem;border-radius:9999px;font-size:.75rem;font-weight:600;text-transform:uppercase}
.badge-success{background:#D1FAE5;color:#065F46}
.badge-warning{background:#FEF3C7;color:#92400E}
.badge-danger{background:#FEE2E2;color:#991B1B}
.compact-metric [data-testid="stMetricValue"]{font-size:1.2rem!important}

/* Fix all text visibility */
p, span, div, label{color:#1F2937!important}
.stMarkdown{color:#1F2937!important}
[data-testid="stCaption"]{color:#6B7280!important}

/* Sidebar fixes - CRITICAL FOR MOBILE */
section[data-testid="stSidebar"]{background:#fff!important}
section[data-testid="stSidebar"] *{color:#1F2937!important}
section[data-testid="stSidebar"] h2{color:#1F2937!important}
section[data-testid="stSidebar"] label{color:#1F2937!important}
section[data-testid="stSidebar"] p{color:#1F2937!important}
section[data-testid="stSidebar"] span{color:#1F2937!important}
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"]{color:#1F2937!important}
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p{color:#1F2937!important}
/* Fix radio buttons in sidebar */
section[data-testid="stSidebar"] div[role="radiogroup"] label{color:#1F2937!important}
section[data-testid="stSidebar"] div[role="radiogroup"] span{color:#1F2937!important}
section[data-testid="stSidebar"] label[data-baseweb="radio"]{color:#1F2937!important}
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div{color:#1F2937!important}

/* Input fields */
input, textarea{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}

/* Number input */
div[data-baseweb="input"] > div{background:#fff!important}
div[data-baseweb="input"] input{color:#1F2937!important}

/* Checkbox */
label[data-baseweb="checkbox"] span{color:#1F2937!important}

/* Mobile responsive */
@media (max-width: 768px) {
  h1{font-size:1.75rem!important}
  h2{font-size:1.5rem!important}
  h3{font-size:1.25rem!important}
  [data-testid="stMetric"]{padding:0.75rem!important}
  [data-testid="stMetricValue"]{font-size:1.5rem!important}
  
  /* Extra mobile fixes for selectbox */
  [data-baseweb="select"] > div{
    background:#fff!important;
    color:#1F2937!important;
  }
  
  /* Force all dropdown menus white on mobile */
  [role="listbox"]{background:#fff!important}
  [role="listbox"] li{background:#fff!important;color:#1F2937!important}
  [role="listbox"] *{color:#1F2937!important}
  ul{background:#fff!important}
  li{background:#fff!important;color:#1F2937!important}
}

/* NUCLEAR OPTION - Override all Streamlit layers and portals */
body > div[class*="layer"],
body > div[class*="Layer"],
#root > div[class*="layer"],
[data-baseweb="layer"]{
  background:transparent!important;
}
[data-baseweb="menu"]{background:#fff!important}
[data-baseweb="menu"] ul{background:#fff!important}
[data-baseweb="menu"] li{background:#fff!important;color:#1F2937!important}
</style>""", unsafe_allow_html=True)

DEFAULT_VAL_PATH = "val_output/undervaluation_scored.csv"

@st.cache_data(ttl=3600)
def get_sp500_data():
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        df = pd.read_html(r.text)[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df[["Symbol", "Security", "GICS Sector"]]
    except:
        r = requests.get("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")
        df = pd.read_csv(io.StringIO(r.text))
        df = df.rename(columns={"Name": "Security", "Sector": "GICS Sector"})
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df[["Symbol", "Security", "GICS Sector"]]

@st.cache_data(ttl=3600)
def load_valuation_scores(csv_path):
    try:
        import os
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        
        # Normalize column names
        if "ticker" in df.columns and "Symbol" not in df.columns:
            df["Symbol"] = df["ticker"].str.upper()
        elif "Ticker" in df.columns and "Symbol" not in df.columns:
            df["Symbol"] = df["Ticker"].str.upper()
        elif "Symbol" in df.columns:
            df["Symbol"] = df["Symbol"].str.upper()
        
        if "sector" in df.columns and "GICS Sector" not in df.columns:
            df["GICS Sector"] = df["sector"]
        elif "Sector" in df.columns and "GICS Sector" not in df.columns:
            df["GICS Sector"] = df["Sector"]
        
        sp = get_sp500_data()
        if "Security" not in df.columns:
            df = df.merge(sp[["Symbol", "Security"]], on="Symbol", how="left")
        if "GICS Sector" not in df.columns:
            df = df.merge(sp[["Symbol", "GICS Sector"]], on="Symbol", how="left")
        
        df["Sector"] = df.get("GICS Sector", "Unknown")
        return df
    except Exception as e:
        st.error(f"Error loading valuation data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_sentiment_model():
    """Load VADER sentiment analyzer"""
    if VADER_AVAILABLE:
        return SentimentIntensityAnalyzer()
    return None

@st.cache_data(ttl=21600)
def analyze_news_sentiment(api_key, company_name, min_articles=5):
    if not api_key or not NewsApiClient:
        return "News API not configured.", "N/A", 0, []
    try:
        newsapi = NewsApiClient(api_key=api_key)
        from_date = (date.today() - timedelta(days=28)).strftime("%Y-%m-%d")
        all_articles = newsapi.get_everything(
            q=f'"{company_name}" AND (earnings OR forecast OR guidance OR outlook OR analyst OR target OR upgrade OR downgrade)', 
            language="en", from_param=from_date, sort_by="relevancy", page_size=50)
        
        # Check for rate limit error
        if isinstance(all_articles, dict) and all_articles.get("status") == "error":
            if "rate limit" in str(all_articles.get("message", "")).lower():
                return "Daily news limit reached. Try again tomorrow.", "N/A", 0, []
        
        if all_articles["totalResults"] == 0:
            return "No articles found.", "N/A", 0, []
        
        exclude_keywords = ['up', 'down', 'gains', 'falls', 'rises', 'drops', 'climbs', 'slides', 
                          'jumps', 'dips', 'rallies', 'plunges', 'surges', 'tumbles', 'soars',
                          'today', 'yesterday', 'trading', 'premarket', 'afterhours']
        forward_keywords = ['forecast', 'outlook', 'guidance', 'expects', 'projected', 'target',
                          'estimate', 'consensus', 'upgrade', 'downgrade', 'rating', 'analyst',
                          'quarter', 'earnings', 'revenue', 'fy2024', 'fy2025', 'next year',
                          'growth', 'expansion', 'strategy', 'future', 'plans']
        
        articles = []
        for a in all_articles["articles"][:40]:
            title_lower = a["title"].lower()
            desc_lower = (a.get("description") or "").lower()
            combined = title_lower + " " + desc_lower
            
            if any(kw in combined for kw in exclude_keywords):
                continue
            
            if any(kw in combined for kw in forward_keywords):
                articles.append(a)
        
        article_list = [{"title": a["title"], "link": a["url"],
                        "publisher": a["source"]["name"], "image_url": a.get("urlToImage")}
                       for a in articles]
        
        if len(articles) < min_articles:
            return f"Only {len(articles)} forward-looking articles", "N/A", len(articles), article_list
        
        sentiment_model = load_sentiment_model()
        if not sentiment_model:
            return "Sentiment model unavailable.", "N/A", len(articles), article_list
        
        # Use VADER for sentiment
        pos, neg, neu = 0, 0, 0
        for a in articles[:20]:
            try:
                text = a["title"] + " " + (a.get("description") or "")
                scores = sentiment_model.polarity_scores(text)
                compound = scores['compound']
                
                # Compound score is between -1 (negative) and 1 (positive)
                if compound >= 0.05:
                    pos += 1
                elif compound <= -0.05:
                    neg += 1
                else:
                    neu += 1
            except:
                pass
        
        total = min(20, len(articles))
        return (pos/total*100, neg/total*100, total, article_list)
    except Exception as e:
        return f"Error: {e}", "N/A", 0, []

@st.cache_data(ttl=3600, show_spinner=False)
def get_price_targets_cached(symbol):
    try:
        if symbol.startswith("^"):
            return {"last": None, "mean": None, "high": None, "low": None, "n": None, "upside": None}
        t = yf.Ticker(symbol)
        hist = t.history(period="1d")
        last = float(hist["Close"][-1]) if not hist.empty else None
        try:
            pt = t.analyst_price_targets or t.get_analyst_price_targets() or {}
        except:
            pt = {}
        mean = pt.get("mean") or pt.get("targetMeanPrice")
        low = pt.get("low") or pt.get("targetLowPrice")
        high = pt.get("high") or pt.get("targetHighPrice")
        n = pt.get("numberOfAnalysts") or pt.get("numAnalysts")
        upside = (mean/last-1) if (mean and last) else None
        return {"last": last, "mean": mean, "high": high, "low": low, "n": int(n) if n else None, "upside": upside}
    except:
        return {"last": None, "mean": None, "high": None, "low": None, "n": None, "upside": None}

def fmt_pct(x):
    return f"{float(x):.2%}" if x is not None and not pd.isna(x) else "N/A"

def fmt_usd(x):
    return f"${float(x):,.2f}" if x is not None and not pd.isna(x) else "N/A"

def fmt_float(x):
    return f"{float(x):.2f}" if x is not None and not pd.isna(x) else "N/A"

# MAIN APP
st.title("🔍 Key Stock Investment Metrics")
st.caption("AI-Powered Valuation, Price Targets & Sentiment Analysis")

# Legal disclaimer and credits
st.markdown("""
<div style='background:#FEF3C7;border-left:4px solid #F59E0B;padding:1rem;border-radius:8px;margin-bottom:1.5rem'>
    <strong>⚠️ For Personal Use Only</strong><br>
    This tool is for educational and personal research purposes only. Not intended for commercial use. 
    Not financial advice. Always do your own research and consult with a financial advisor before making investment decisions.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<details style='margin-bottom:1rem'>
<summary style='cursor:pointer;color:#6B7280;font-size:0.85rem'>📚 Credits & Data Sources</summary>
<div style='padding:0.5rem;font-size:0.85rem;color:#6B7280'>
    <strong>Libraries & APIs:</strong><br>
    • <a href="https://streamlit.io" target="_blank">Streamlit</a> - Interactive web framework<br>
    • <a href="https://github.com/ranaroussi/yfinance" target="_blank">yfinance</a> - Yahoo Finance data<br>
    • <a href="https://plotly.com" target="_blank">Plotly</a> - Interactive visualizations<br>
    • <a href="https://github.com/cjhutto/vaderSentiment" target="_blank">VADER Sentiment</a> - News sentiment analysis<br>
    • <a href="https://newsapi.org" target="_blank">NewsAPI</a> - News articles<br>
    • <strong>pandas, numpy</strong> - Data processing<br><br>
    <strong>Data Sources:</strong><br>
    • Stock prices: Yahoo Finance<br>
    • Analyst ratings: Yahoo Finance<br>
    • News articles: NewsAPI.org<br>
    • S&P 500 list: Wikipedia
</div>
</details>
""", unsafe_allow_html=True)

st.sidebar.header("🧭 Navigation")
app_mode = st.sidebar.radio("Mode", ("Single Stock Analysis", "Multi-Stock Comparison", "Top Undervalued Stocks"))

# Get API keys (hidden from UI)
try:
    news_api_key = st.secrets.get("NEWS_API_KEY")
except:
    news_api_key = None

val_path = DEFAULT_VAL_PATH

sp500_df = get_sp500_data()
index_dict = {"^GSPC": "S&P 500", "^NDX": "Nasdaq-100"}

# SINGLE STOCK ANALYSIS
if app_mode == "Single Stock Analysis":
    st.subheader("🔍 Stock Selection")
    company_dict = pd.Series(sp500_df["Security"].values, index=sp500_df["Symbol"]).to_dict()
    full_list = {**index_dict, **company_dict}
    
    selected = st.selectbox("Select Stock", list(full_list.keys()),
                           index=list(full_list.keys()).index("AAPL") if "AAPL" in full_list else 0,
                           format_func=lambda s: f"{full_list.get(s,s)} ({s})")
    
    st.markdown("---")
    st.session_state.selected_symbol = selected
    
    # Track stock selection in Google Analytics
    if selected:
        components.html(f"""
        <script>
          if (typeof gtag !== 'undefined') {{
            gtag('event', 'stock_view', {{
              'stock_symbol': '{selected}',
              'mode': 'single_stock'
            }});
          }}
        </script>
        """, height=0)
    
    if selected:
        ticker = yf.Ticker(selected)
        company = full_list.get(selected, selected)
        st.header(f"📈 {company}")
        st.caption(f"**{selected}**")
        
        # FINANCIAL METRICS
        if selected not in index_dict:
            with st.expander("💰 Financial Metrics", expanded=True):
                try:
                    info = ticker.info
                    qf = ticker.quarterly_financials
                    c1, c2, c3, c4 = st.columns(4)
                    try:
                        c1.metric("Q Revenue", f"${qf.loc['Total Revenue'][0]/1e9:.2f}B")
                    except:
                        c1.metric("Q Revenue", "N/A")
                    c2.metric("Margin", fmt_pct(info.get("profitMargins", 0)))
                    c3.metric("EPS", fmt_float(info.get("trailingEps", 0)))
                    c4.metric("P/E", fmt_float(info.get("trailingPE", 0)))
                except:
                    st.warning("⚠️ Data unavailable")
        
        # VALUATION GAUGE AND ANALYST RATINGS
        if selected not in index_dict:
            col_left, col_right = st.columns(2)
            
            with col_left:
                try:
                    val_df = load_valuation_scores(val_path)
                    
                    if val_df.empty:
                        st.subheader("📊 Valuation Gauge")
                        st.info("Valuation data not available")
                    elif "Symbol" not in val_df.columns:
                        st.subheader("📊 Valuation Gauge")
                        st.error("❌ CSV missing 'Symbol' column")
                    else:
                        row = val_df[val_df["Symbol"].str.upper() == selected.upper()]
                        
                        if not row.empty and "undervaluation_score" in row.columns:
                            score = float(row.iloc[0]["undervaluation_score"])
                            sector_name = row.iloc[0].get("Sector", row.iloc[0].get("GICS Sector", "Unknown"))
                            
                            st.subheader("📊 Valuation Gauge")
                            
                            badge = "success" if score <= 3 else "warning" if score <= 7 else "danger"
                            label = "Undervalued" if score <= 3 else "Fairly Valued" if score <= 7 else "Overvalued"
                            st.markdown(f'<span class="badge badge-{badge}">{label}</span>', unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style="margin-top:1rem">
                                <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#6B7280;margin-bottom:0.25rem">
                                    <span>Undervalued</span>
                                    <span>Fairly Valued</span>
                                    <span>Overvalued</span>
                                </div>
                                <div style="width:100%;height:8px;background:linear-gradient(to right, #10B981, #FCD34D, #EF4444);border-radius:4px;position:relative">
                                    <div style="position:absolute;left:{(score-1)/9*100}%;top:-4px;width:16px;height:16px;background:#1F2937;border-radius:50%;border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.2)"></div>
                                </div>
                                <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#9CA3AF;margin-top:0.5rem">
                                    <span>1</span>
                                    <span>5</span>
                                    <span>10</span>
                                </div>
                            </div>
                            <div style='margin-top:1rem;text-align:center;padding:1rem;background:#F9FAFB;border-radius:8px'>
                                <div style='font-size:2rem;font-weight:700;color:#1F2937'>{score:.1f}<span style='font-size:1.2rem;color:#6B7280'>/10</span></div>
                                <div style='font-size:0.9rem;color:#6B7280;margin-top:0.5rem'>Sector: {sector_name}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add explanation
                            with st.expander("ℹ️ How is this calculated?"):
                                st.markdown("""
                                **Undervaluation Score Methodology:**
                                
                                The score (1-10) is calculated based on multiple valuation metrics:
                                
                                - **Price-to-Earnings (P/E) Ratio** - vs sector average
                                - **Price-to-Book (P/B) Ratio** - vs historical values
                                - **Price-to-Sales (P/S) Ratio** - vs competitors
                                - **PEG Ratio** - P/E relative to growth rate
                                - **Enterprise Value multiples** - EV/EBITDA, EV/Sales
                                - **Dividend Yield** - compared to peers
                                
                                **Score Interpretation:**
                                - **1-3**: Potentially undervalued (green zone)
                                - **4-7**: Fairly valued (yellow zone)
                                - **8-10**: Potentially overvalued (red zone)
                                
                                *Lower scores suggest better value, but always consider company fundamentals, 
                                growth prospects, and market conditions. This is not financial advice.*
                                """)
                        else:
                            st.info(f"No valuation data for {selected}")
                            
                except Exception as e:
                    st.warning(f"⚠️ Valuation: {str(e)}")
            
            with col_right:
                st.subheader("⭐ Analyst Ratings")
                try:
                    recs = ticker.recommendations_summary
                    if recs is not None and not recs.empty:
                        s = recs.iloc[-1]
                        
                        # Calculate weighted average rating (1=Strong Buy to 5=Strong Sell)
                        rating_values = {
                            "strongBuy": 1,
                            "buy": 2,
                            "hold": 3,
                            "sell": 4,
                            "strongSell": 5
                        }
                        
                        total_ratings = 0
                        weighted_sum = 0
                        
                        for rating_key, rating_val in rating_values.items():
                            if rating_key in s.index and s[rating_key] > 0:
                                count = int(s[rating_key])
                                total_ratings += count
                                weighted_sum += count * rating_val
                        
                        if total_ratings > 0:
                            avg_rating = weighted_sum / total_ratings
                            
                            # Create barometer gauge
                            st.markdown(f"""
                            <div style="margin-top:1rem">
                                <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#6B7280;margin-bottom:0.25rem">
                                    <span>Strong Buy</span>
                                    <span>Hold</span>
                                    <span>Strong Sell</span>
                                </div>
                                <div style="width:100%;height:12px;background:linear-gradient(to right, #10B981, #34D399, #FCD34D, #FB923C, #EF4444);border-radius:6px;position:relative">
                                    <div style="position:absolute;left:{(avg_rating-1)/4*100}%;top:-2px;width:20px;height:20px;background:#1F2937;border-radius:50%;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3)"></div>
                                </div>
                                <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#9CA3AF;margin-top:0.5rem">
                                    <span>1</span>
                                    <span>2</span>
                                    <span>3</span>
                                    <span>4</span>
                                    <span>5</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show rating label
                            if avg_rating <= 1.5:
                                label = "Strong Buy"
                                color = "#10B981"
                            elif avg_rating <= 2.5:
                                label = "Buy"
                                color = "#34D399"
                            elif avg_rating <= 3.5:
                                label = "Hold"
                                color = "#FCD34D"
                            elif avg_rating <= 4.5:
                                label = "Sell"
                                color = "#FB923C"
                            else:
                                label = "Strong Sell"
                                color = "#EF4444"
                            
                            st.markdown(f"""
                            <div style='margin-top:1rem;text-align:center;padding:1rem;background:#F9FAFB;border-radius:8px'>
                                <div style='font-size:1.75rem;font-weight:700;color:{color}'>{label}</div>
                                <div style='font-size:0.9rem;color:#6B7280;margin-top:0.5rem'>
                                    Avg Rating: {avg_rating:.2f} | {total_ratings} analysts
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show breakdown
                            with st.expander("📊 Rating Breakdown"):
                                for rating_key in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
                                    if rating_key in s.index and s[rating_key] > 0:
                                        label_name = rating_key.replace("strong", "Strong ")
                                        st.write(f"**{label_name}:** {int(s[rating_key])}")
                        else:
                            st.info("No rating data")
                    else:
                        st.info("No ratings available")
                except Exception as e:
                    st.warning(f"⚠️ No ratings: {str(e)}")
        
        # NEWS SENTIMENT
        if selected not in index_dict:
            if news_api_key and NewsApiClient:
                with st.expander("📰 News Sentiment (VADER Analysis)", expanded=False):
                    st.caption("Earnings, forecasts, guidance, analyst ratings - excludes daily price moves")
                    bull, bear, n, _ = analyze_news_sentiment(news_api_key, company, 5)
                    if isinstance(bull, str):
                        if "limit reached" in bull.lower():
                            st.warning(f"ℹ️ {bull}")
                        else:
                            st.info(f"ℹ️ {bull}")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🟢 Bullish", f"{bull:.1f}%")
                        c2.metric("🔴 Bearish", f"{bear:.1f}%")
                        c3.metric("📊 Neutral", f"{100-bull-bear:.1f}%")
                        st.caption(f"Based on {n} forward-looking articles (VADER sentiment)")
            else:
                with st.expander("📰 News Sentiment", expanded=False):
                    st.info("News API not configured. Add NEWS_API_KEY to Streamlit secrets.")
        
        # PRICE TARGETS
        if selected not in index_dict:
            with st.expander("🎯 Price Targets", expanded=True):
                pt = get_price_targets_cached(selected)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("💵 Current", fmt_usd(pt["last"]))
                c2.metric("🎯 Mean", fmt_usd(pt["mean"]))
                with c3:
                    st.markdown('<div class="compact-metric">', unsafe_allow_html=True)
                    st.metric("📊 High/Low", f"{fmt_usd(pt['high'])}/{fmt_usd(pt['low'])}")
                    st.markdown('</div>', unsafe_allow_html=True)
                c4.metric("📈 Upside", fmt_pct(pt["upside"]))
                if pt["n"]:
                    st.caption(f"**{pt['n']} analysts**")
        
        # PRICE HISTORY
        with st.expander("📈 Price History", expanded=False):
            hist = ticker.history(period="5y")
            if not hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'],
                    mode='lines', line=dict(color='#3B82F6', width=2)))
                fig.update_layout(title=f"{selected} - 5Y", xaxis_title="Date",
                    yaxis_title="Price", height=400, margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
        
        # NEWS ARTICLES SECTION
        if selected not in index_dict and news_api_key and NewsApiClient:
            st.markdown("---")
            st.subheader("📰 Latest News & Analysis")
            try:
                _, _, _, articles = analyze_news_sentiment(news_api_key, company, 5)
                if articles:
                    st.caption(f"Found {len(articles)} forward-looking articles")
                    for i, a in enumerate(articles[:10]):
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if a.get("image_url"):
                                st.image(a["image_url"], width=120)
                        with col2:
                            st.markdown(f"**[{a['title']}]({a['link']})**")
                            st.caption(f"📰 {a['publisher']}")
                        
                        if i < min(9, len(articles) - 1):
                            st.divider()
                else:
                    st.info("No recent articles found")
            except Exception as e:
                st.warning(f"Could not load news: {str(e)}")

# MULTI-STOCK COMPARISON
elif app_mode == "Multi-Stock Comparison":
    st.subheader("📊 Comparison")
    company_dict = pd.Series(sp500_df["Security"].values, index=sp500_df["Symbol"]).to_dict()
    full_list = {**index_dict, **company_dict}
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        symbols = st.multiselect("Stocks", list(full_list.keys()), ["AAPL","MSFT","^GSPC"],
            format_func=lambda s: f"{full_list.get(s,s)} ({s})")
    with c2:
        start = st.date_input("Start", date(2020,1,1))
    with c3:
        end = st.date_input("End", date.today())
    
    st.markdown("---")
    
    if symbols:
        df = pd.DataFrame()
        for s in symbols:
            h = yf.Ticker(s).history(period="max", start=start, end=end)
            if not h.empty:
                df[s] = h["Close"]
        if not df.empty:
            norm = (df / df.iloc[0] - 1) * 100
            fig = go.Figure()
            colors = ['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6','#EC4899']
            for i, col in enumerate(norm.columns):
                fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode='lines',
                    name=col, line=dict(color=colors[i % len(colors)], width=2)))
            fig.update_layout(title="Performance (%)", xaxis_title="Date",
                yaxis_title="Return (%)", height=600)
            st.plotly_chart(fig, use_container_width=True)

# TOP UNDERVALUED STOCKS
elif app_mode == "Top Undervalued Stocks":
    st.header("🏆 Top Undervalued Stocks Right Now")
    
    try:
        val_df = load_valuation_scores(val_path)
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()
    
    if val_df.empty:
        st.warning("No valuation data available")
        st.stop()
    
    sectors = ["All"] + sorted(val_df["Sector"].dropna().unique().tolist())
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        sector = st.selectbox("Sector", sectors)
    with c2:
        mc_top = st.number_input("Top N by Market Cap", 0, 200, 0, 10)
    with c3:
        top_n = st.selectbox("Show Top", [10, 20, 50, 100])
    with c4:
        targets = st.checkbox("Show Price Targets", True)
    
    c_reload, _ = st.columns([1, 5])
    with c_reload:
        if st.button("🔄 Reload"):
            load_valuation_scores.clear()
            st.rerun()
    
    st.markdown("---")
    
    df = val_df.copy()
    if sector != "All":
        df = df[df["Sector"] == sector]
    if "mkt_cap" in df.columns and mc_top > 0:
        df = df.sort_values("mkt_cap", ascending=False).head(mc_top)
    
    # Always sort by undervaluation (lowest score = most undervalued)
    df = df.sort_values("undervaluation_score", ascending=True).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    out = df.head(top_n).copy()
    
    if targets and not out.empty:
        with st.spinner("Loading targets..."):
            out["Last"] = None
            out["PT Mean"] = None
            out["PT High"] = None
            out["PT Low"] = None
            out["Upside"] = None
            for i, row in out.iterrows():
                pt = get_price_targets_cached(row["Symbol"])
                out.at[i, "Last"] = pt["last"]
                out.at[i, "PT Mean"] = pt["mean"]
                out.at[i, "PT High"] = pt["high"]
                out.at[i, "PT Low"] = pt["low"]
                out.at[i, "Upside"] = pt["upside"]
    
    cols = ["Rank", "Symbol", "Security", "Sector", "undervaluation_score"]
    if targets:
        cols += ["Last", "PT Mean", "PT High", "PT Low", "Upside"]
    disp = out[[c for c in cols if c in out.columns]].copy()
    
    for c in ["Last", "PT Mean", "PT High", "PT Low"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(fmt_usd)
    if "Upside" in disp.columns:
        disp["Upside"] = disp["Upside"].apply(fmt_pct)
    
    st.subheader(f"Top {top_n} Most Undervalued Stocks {f'in {sector}' if sector != 'All' else ''}")
    st.dataframe(disp, use_container_width=True, height=600, hide_index=True)
    
    if not out.empty:
        fig = px.bar(out, x="Symbol", y="undervaluation_score", title="Undervaluation Scores (Lower = Better Value)",
            color="undervaluation_score", color_continuous_scale=["#10B981","#FCD34D","#EF4444"])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.download_button("⬇️ Download CSV", disp.to_csv(index=False),
        f"top_undervalued_{sector.replace(' ','_')}_top{top_n}.csv", "text/csv")

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#9CA3AF;font-size:0.8rem;padding:0.5rem'>
    Built by ANGAD ARORA
</div>
""", unsafe_allow_html=True)
