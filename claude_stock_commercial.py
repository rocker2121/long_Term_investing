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

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,.stApp{font-family:'Inter',sans-serif!important;background:#F9FAFB}
h1{font-size:2.5rem!important;font-weight:700!important;background:linear-gradient(135deg,#667eea,#764ba2);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
[data-testid="stMetric"]{background:#fff;padding:1rem;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.1);border:1px solid #E5E7EB}
[data-testid="stMetricValue"]{font-size:1.75rem!important;font-weight:700!important;color:#1F2937!important}
div[data-testid="stExpander"]{background:#fff;border-radius:12px;border:1px solid #E5E7EB;margin-bottom:1rem}
.stButton>button{background:linear-gradient(135deg,#667eea,#764ba2)!important;color:#fff!important;
border:none!important;border-radius:8px!important;padding:.5rem 1.5rem!important;font-weight:600!important}
.badge{display:inline-block;padding:.25rem .75rem;border-radius:9999px;font-size:.75rem;font-weight:600;text-transform:uppercase}
.badge-success{background:#D1FAE5;color:#065F46}
.badge-warning{background:#FEF3C7;color:#92400E}
.badge-danger{background:#FEE2E2;color:#991B1B}
.compact-metric [data-testid="stMetricValue"]{font-size:1.2rem!important}
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

@st.cache_data(ttl=1800)
def analyze_news_sentiment(api_key, company_name, min_articles=5):
    if not api_key or not NewsApiClient:
        return "News API not configured.", "N/A", 0, []
    try:
        newsapi = NewsApiClient(api_key=api_key)
        from_date = (date.today() - timedelta(days=28)).strftime("%Y-%m-%d")
        all_articles = newsapi.get_everything(
            q=f'"{company_name}" AND (earnings OR forecast OR guidance OR outlook OR analyst OR target OR upgrade OR downgrade)', 
            language="en", from_param=from_date, sort_by="relevancy", page_size=50)
        
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

@st.cache_data(ttl=1800, show_spinner=False)
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
st.title("📊 S&P 500 Stock Analyzer")
st.caption("AI-Powered Valuation, Price Targets & Sentiment Analysis")

st.sidebar.header("🧭 Navigation")
app_mode = st.sidebar.radio("Mode", ("Single Stock Analysis", "Multi-Stock Comparison",
                                     "News Drilldown", "Valuation Rankings"))

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
val_path = st.sidebar.text_input("Valuation CSV", DEFAULT_VAL_PATH)

try:
    news_api_key = st.secrets.get("NEWS_API_KEY")
    if news_api_key and NewsApiClient:
        st.sidebar.success("✅ News API configured")
    else:
        st.sidebar.info("💡 News API: Not configured (optional)")
except:
    st.sidebar.info("💡 News API: Not configured (optional)")
    news_api_key = None

if VADER_AVAILABLE:
    st.sidebar.success("✅ VADER sentiment analysis ready")
else:
    st.sidebar.warning("⚠️ VADER not available")

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
                        st.markdown('<div style="background:#fff;padding:1.5rem;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,.1);height:100%">',
                                    unsafe_allow_html=True)
                        st.subheader("📊 Valuation Gauge")
                        st.info("Valuation data not available")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif "Symbol" not in val_df.columns:
                        st.markdown('<div style="background:#fff;padding:1.5rem;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,.1);height:100%">',
                                    unsafe_allow_html=True)
                        st.subheader("📊 Valuation Gauge")
                        st.error("❌ CSV missing 'Symbol' column")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        row = val_df[val_df["Symbol"].str.upper() == selected.upper()]
                        
                        if not row.empty and "undervaluation_score" in row.columns:
                            score = float(row.iloc[0]["undervaluation_score"])
                            sector_name = row.iloc[0].get("Sector", row.iloc[0].get("GICS Sector", "Unknown"))
                            
                            st.markdown('<div style="background:#fff;padding:1.5rem;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,.1);height:100%">',
                                        unsafe_allow_html=True)
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
                                <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#9CA3AF;margin-top:0.25rem">
                                    <span>1</span>
                                    <span>5</span>
                                    <span>10</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"<div style='margin-top:1rem;text-align:center'><strong style='font-size:1.5rem'>{score:.1f}/10</strong><br><span style='font-size:0.85rem;color:#6B7280'>Sector: {sector_name}</span></div>", 
                                       unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"No valuation data for {selected}")
                            
                except Exception as e:
                    st.warning(f"⚠️ Valuation: {str(e)}")
            
            with col_right:
                st.markdown('<div style="background:#fff;padding:1.5rem;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,.1);height:100%">',
                            unsafe_allow_html=True)
                st.subheader("⭐ Analyst Ratings")
                try:
                    recs = ticker.recommendations_summary
                    if recs is not None and not recs.empty:
                        s = recs.iloc[-1]
                        cols = [c for c in ["strongBuy","buy","hold","sell","strongSell"] if c in s.index]
                        if cols:
                            rating_data = []
                            rating_labels = []
                            rating_colors = []
                            color_map = {
                                "strongBuy": "#10B981",
                                "buy": "#34D399", 
                                "hold": "#FCD34D",
                                "sell": "#FB923C",
                                "strongSell": "#EF4444"
                            }
                            for c in cols:
                                if s[c] > 0:
                                    rating_data.append(int(s[c]))
                                    rating_labels.append(c.replace("strong", "Strong "))
                                    rating_colors.append(color_map.get(c, "#6B7280"))
                            
                            if rating_data:
                                fig = go.Figure(data=[go.Pie(
                                    labels=rating_labels,
                                    values=rating_data,
                                    marker=dict(colors=rating_colors),
                                    hole=0.4,
                                    textposition='inside',
                                    textinfo='label+percent'
                                )])
                                fig.update_layout(
                                    showlegend=True,
                                    height=280,
                                    margin=dict(l=20, r=20, t=20, b=20),
                                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No rating data")
                        else:
                            st.info("No rating columns")
                    else:
                        st.info("No ratings available")
                except:
                    st.warning("⚠️ No ratings")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # NEWS SENTIMENT
        if selected not in index_dict:
            if news_api_key and NewsApiClient:
                with st.expander("📰 News Sentiment (VADER Analysis)", expanded=True):
                    st.caption("Earnings, forecasts, guidance, analyst ratings - excludes daily price moves")
                    bull, bear, n, _ = analyze_news_sentiment(news_api_key, company, 5)
                    if isinstance(bull, str):
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

# NEWS DRILLDOWN
elif app_mode == "News Drilldown":
    st.header("📰 News Details")
    if "selected_symbol" in st.session_state:
        sym = st.session_state.selected_symbol
        company = index_dict.get(sym) or sp500_df.set_index("Symbol")["Security"].get(sym, sym)
        st.subheader(f"{company} ({sym})")
        _, _, _, articles = analyze_news_sentiment(news_api_key, company, 5)
        if articles:
            for i, a in enumerate(articles):
                c1, c2 = st.columns([1, 4])
                with c1:
                    if a.get("image_url"):
                        st.image(a["image_url"], width=150)
                with c2:
                    st.markdown(f"**[{a['title']}]({a['link']})**")
                    st.caption(f"📰 {a['publisher']}")
                if i < len(articles) - 1:
                    st.divider()
        else:
            st.warning("⚠️ No articles")
    else:
        st.info("👈 Select stock first")

# VALUATION RANKINGS
elif app_mode == "Valuation Rankings":
    st.header("🏆 Rankings")
    
    try:
        val_df = load_valuation_scores(val_path)
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()
    
    if val_df.empty:
        st.warning("No valuation data available")
        st.stop()
    
    sectors = ["All"] + sorted(val_df["Sector"].dropna().unique().tolist())
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        sector = st.selectbox("Sector", sectors)
    with c2:
        mc_top = st.number_input("Top N Mkt", 0, 200, 0, 10)
    with c3:
        view = st.selectbox("Sort", ["Undervalued", "Overvalued"])
    with c4:
        top_n = st.selectbox("Show", [10, 20, 50, 100])
    with c5:
        targets = st.checkbox("Targets", True)
    
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
    
    asc = view == "Undervalued"
    df = df.sort_values("undervaluation_score", ascending=asc).reset_index(drop=True)
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
    
    st.subheader(f"Top {top_n} {view} {f'in {sector}' if sector != 'All' else ''}")
    st.dataframe(disp, use_container_width=True, height=600, hide_index=True)
    
    if not out.empty:
        fig = px.bar(out, x="Symbol", y="undervaluation_score", title="Scores",
            color="undervaluation_score", color_continuous_scale=["#10B981","#FCD34D","#EF4444"])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.download_button("⬇️ CSV", disp.to_csv(index=False),
        f"rankings_{view.lower()}_{sector.replace(' ','_')}_top{top_n}.csv", "text/csv")

st.markdown("---")
st.caption("⚠️ For informational purposes only. Not financial advice.")
st.caption("Built by ANGAD ARORA")
