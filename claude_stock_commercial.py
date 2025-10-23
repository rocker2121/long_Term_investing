undervaluation_scored.csv#!/usr/bin/env python3
"""IntelliInvest - Complete S&P 500 Stock Analyzer with Authentication
   
   Features:
   - Email + Password authentication
   - Portfolio tracking (persistent across devices)
   - Stock analysis with valuation scores
   - News sentiment analysis (VADER)
   - Price targets & analyst ratings
   - Multi-stock comparison
   - Top undervalued stocks finder
   - Monthly email updates (automated script available)
   - Google Analytics & PostHog tracking
   
   Uses ONLY FREE services!
"""

import io
import json
import uuid
import hashlib
from datetime import date, timedelta, datetime

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

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except:
    GSPREAD_AVAILABLE = False

# --------------------------------------------------------------------------------------
# PAGE SETUP
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="IntelliInvest", page_icon="📊", layout="wide")

# --------------------------------------------------------------------------------------
# ANALYTICS CONFIG
# --------------------------------------------------------------------------------------
ENABLE_GA_MP = True
ENABLE_POSTHOG_CLIENT = True
ENABLE_POSTHOG_SERVER = True

GA_MEASUREMENT_ID = st.secrets.get("GA_MEASUREMENT_ID", "G-598BZYJEBM")
GA_API_SECRET = st.secrets.get("GA_API_SECRET", "PUT_YOUR_GA_API_SECRET")
POSTHOG_KEY = st.secrets.get("POSTHOG_KEY", "phc_your_project_key_here")
POSTHOG_HOST = st.secrets.get("POSTHOG_HOST", "https://app.posthog.com")
APP_URL = "https://intellinvest.streamlit.app/"
DEFAULT_VAL_PATH = st.secrets.get("VALUATION_CSV_URL",
    "https://raw.githubusercontent.com/angadarora2024/IntelliInvest/main/undervaluation_scored.csv")

# --------------------------------------------------------------------------------------
# AUTHENTICATION FUNCTIONS
# --------------------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hash password with SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash: str, provided_password: str) -> bool:
    """Verify password against stored hash"""
    return stored_hash == hash_password(provided_password)

def get_google_sheet():
    """Get Google Sheets client"""
    if not GSPREAD_AVAILABLE:
        return None
    
    try:
        creds_dict = st.secrets.get("gspread", None)
        if not creds_dict:
            return None
        
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        try:
            sheet = client.open("IntelliInvest_Users").sheet1
        except:
            spreadsheet = client.create("IntelliInvest_Users")
            sheet = spreadsheet.sheet1
            sheet.append_row(["Email", "Password_Hash", "Portfolio", "User_ID", "Created_At", "Last_Login", "Status"])
        
        return sheet
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {str(e)}")
        return None

def create_user(email: str, password: str, portfolio: list) -> bool:
    """Create new user account"""
    sheet = get_google_sheet()
    if not sheet:
        return False
    
    try:
        try:
            existing = sheet.find(email, in_column=1)
            if existing:
                return False
        except:
            pass
        
        user_id = str(uuid.uuid4())
        password_hash = hash_password(password)
        portfolio_str = ",".join(portfolio)
        timestamp = datetime.now().isoformat()
        
        sheet.append_row([email, password_hash, portfolio_str, user_id, timestamp, timestamp, "active"])
        return True
        
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return False

def login_user(email: str, password: str) -> dict:
    """Login user and return their data"""
    sheet = get_google_sheet()
    if not sheet:
        return None
    
    try:
        cell = sheet.find(email, in_column=1)
        if not cell:
            return None
        
        row_values = sheet.row_values(cell.row)
        stored_hash = row_values[1]
        
        if not verify_password(stored_hash, password):
            return None
        
        sheet.update_cell(cell.row, 6, datetime.now().isoformat())
        
        portfolio_str = row_values[2]
        portfolio = [s.strip() for s in portfolio_str.split(',') if s.strip()]
        
        return {
            "email": email,
            "portfolio": portfolio,
            "user_id": row_values[3],
            "created_at": row_values[4],
            "row_num": cell.row
        }
        
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return None

def update_user_portfolio(email: str, portfolio: list) -> bool:
    """Update user's portfolio"""
    sheet = get_google_sheet()
    if not sheet:
        return False
    
    try:
        cell = sheet.find(email, in_column=1)
        if not cell:
            return False
        
        portfolio_str = ",".join(portfolio)
        sheet.update_cell(cell.row, 3, portfolio_str)
        return True
        
    except Exception as e:
        st.error(f"Error updating portfolio: {str(e)}")
        return False

def init_session():
    """Initialize session state"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_data" not in st.session_state:
        st.session_state.user_data = None
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

def logout():
    """Logout user"""
    st.session_state.logged_in = False
    st.session_state.user_data = None
    st.session_state.portfolio = []

# --------------------------------------------------------------------------------------
# ANALYTICS HELPERS
# --------------------------------------------------------------------------------------
def _ensure_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]

def ga_mp_send(event_name: str, params: dict):
    if not ENABLE_GA_MP:
        return
    try:
        cid = _ensure_session_id()
        url = f"https://www.google-analytics.com/mp/collect?measurement_id={GA_MEASUREMENT_ID}&api_secret={GA_API_SECRET}"
        payload = {"client_id": cid, "events": [{"name": event_name, "params": params}]}
        requests.post(url, json=payload, timeout=4)
    except Exception:
        pass

def posthog_server_capture(event_name: str, properties: dict):
    if not ENABLE_POSTHOG_SERVER:
        return
    try:
        distinct_id = _ensure_session_id()
        url = f"{POSTHOG_HOST}/capture/"
        payload = {
            "api_key": POSTHOG_KEY,
            "event": event_name,
            "properties": properties,
            "distinct_id": distinct_id,
        }
        requests.post(url, data=json.dumps(payload),
                      headers={"Content-Type": "application/json"}, timeout=4)
    except Exception:
        pass

def posthog_client_boot():
    if not ENABLE_POSTHOG_CLIENT:
        return
    components.html(f"""
<!doctype html><html><head><meta charset="utf-8"></head><body>
<script>
  (function() {{
    try {{
      if (!window.parent.posthog) {{
        var s = document.createElement('script'); s.async = true;
        s.src = '{POSTHOG_HOST}/static/array.js';
        s.onload = function() {{
          try {{
            window.posthog = window.posthog || [];
            window.posthog.init('{POSTHOG_KEY}', {{ api_host: '{POSTHOG_HOST}' }});
            window.parent.posthog = window.posthog;
          }} catch(e) {{}}
        }};
        document.head.appendChild(s);
      }}
    }} catch (e) {{}}
  }})();
</script>
</body></html>
""", height=0)

posthog_client_boot()

if st.session_state.get("_sent_app_start") is None:
    st.session_state["_sent_app_start"] = True
    ga_mp_send("page_view", {"page_title": "IntelliInvest", "page_location": APP_URL})
    posthog_server_capture("app_start", {"app": "intellinvest"})

# --------------------------------------------------------------------------------------
# STYLING
# --------------------------------------------------------------------------------------
st.markdown("""
<script>
window.parent.document.documentElement.setAttribute('data-theme', 'light');
</script>
""", unsafe_allow_html=True)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { color-scheme: light !important; }
html, body, #root, .stApp { color-scheme: light !important; background: #F9FAFB !important; }
html,body,.stApp{font-family:'Inter',sans-serif!important;background:#F9FAFB}
h1{font-size:2.5rem!important;font-weight:700!important;background:linear-gradient(135deg,#667eea,#764ba2);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
h2,h3{color:#1F2937!important}
.stMetric{background:#fff;padding:1rem;border-radius:0.75rem;border:1px solid #E5E7EB}
button[kind="primary"]{background:linear-gradient(135deg,#667eea,#764ba2)!important;color:#fff!important}
[data-testid="stExpander"]{background:#fff;border:1px solid #E5E7EB;border-radius:0.75rem}
</style>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_sp500_list():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        df.columns = df.columns.str.strip()
        if "Symbol" not in df.columns and len(df.columns) > 0:
            df.rename(columns={df.columns[0]: "Symbol"}, inplace=True)
        if "Security" not in df.columns and len(df.columns) > 1:
            df.rename(columns={df.columns[1]: "Security"}, inplace=True)
        if "GICS Sector" in df.columns:
            df.rename(columns={"GICS Sector": "Sector"}, inplace=True)
        return df[["Symbol", "Security", "Sector"]]
    except:
        return pd.DataFrame(columns=["Symbol", "Security", "Sector"])

@st.cache_data(ttl=3600)
def load_valuation_scores(path: str) -> pd.DataFrame:
    try:
        resp = requests.get(path, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.dropna(subset=["Symbol"], inplace=True)
        df.sort_values("undervaluation_score", ascending=True, inplace=True)
        return df
    except Exception as e:
        st.warning(f"Could not load valuation data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_current_price(ticker: str) -> float:
    try:
        t = yf.Ticker(ticker)
        data = t.fast_info
        return round(data.get("last_price", 0.0), 2)
    except:
        return 0.0

@st.cache_resource
def load_sentiment_model():
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

        pos, neg, neu = 0, 0, 0
        for a in articles[:20]:
            try:
                text = a["title"] + " " + (a.get("description") or "")
                scores = sentiment_model.polarity_scores(text)
                compound = scores['compound']
                if compound >= 0.05: pos += 1
                elif compound <= -0.05: neg += 1
                else: neu += 1
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
            info = t.info
            mean = info.get("targetMeanPrice")
            low = info.get("targetLowPrice")
            high = info.get("targetHighPrice")
            n = info.get("numberOfAnalystOpinions")
        except:
            mean, low, high, n = None, None, None, None
        upside = ((mean/last)-1)*100 if (mean and last) else None
        return {"last": last, "mean": mean, "high": high, "low": low,
                "n": int(n) if n else None, "upside": upside}
    except:
        return {"last": None, "mean": None, "high": None, "low": None, "n": None, "upside": None}

@st.cache_data(ttl=3600)
def get_analyst_ratings_cached(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        rec = t.recommendations
        if rec is not None and not rec.empty:
            latest = rec.iloc[-1]
            buy = latest.get("strongBuy", 0) + latest.get("buy", 0)
            hold = latest.get("hold", 0)
            sell = latest.get("sell", 0) + latest.get("strongSell", 0)
            total = buy + hold + sell
            if total == 0:
                return {"buy": 0, "hold": 0, "sell": 0, "n": 0, "avg": 3.0}
            buy_pct = (buy / total) * 100
            hold_pct = (hold / total) * 100
            sell_pct = (sell / total) * 100
            avg = (buy * 1 + hold * 3 + sell * 5) / total
            return {"buy": buy_pct, "hold": hold_pct, "sell": sell_pct, "n": total, "avg": avg}
        return {"buy": 0, "hold": 0, "sell": 0, "n": 0, "avg": 3.0}
    except:
        return {"buy": 0, "hold": 0, "sell": 0, "n": 0, "avg": 3.0}

@st.cache_data(ttl=3600)
def get_fundamentals_cached(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "rev": info.get("totalRevenue", 0) / 1e9,
            "margin": info.get("profitMargins", 0) * 100,
            "eps": info.get("trailingEps", 0),
            "pe": info.get("trailingPE", 0),
            "pb": info.get("priceToBook", 0),
            "div": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
        }
    except:
        return {"rev": 0, "margin": 0, "eps": 0, "pe": 0, "pb": 0, "div": 0}

def fmt_usd(val): return f"${val:,.2f}" if val else "N/A"
def fmt_pct(val): return f"{val:+.1f}%" if val else "N/A"

# --------------------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------------------
sp500_df = load_sp500_list()
val_path = DEFAULT_VAL_PATH
try:
    val_df = load_valuation_scores(val_path)
except:
    val_df = pd.DataFrame()

index_dict = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"}
news_api_key = st.secrets.get("NEWS_API_KEY", None)

# Initialize session
init_session()

# --------------------------------------------------------------------------------------
# AUTHENTICATION UI
# --------------------------------------------------------------------------------------
if not st.session_state.logged_in:
    st.title("📊 IntelliInvest")
    
    st.markdown("""
    <div style='text-align:center;padding:2rem;'>
        <h2>Smart Stock Analysis with AI-Powered Insights</h2>
        <p style='color:#6B7280;font-size:1.1rem;'>
            Track your favorite stocks • Get valuation insights • Receive monthly updates
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "✨ Sign Up"])
    
    with tab1:
        st.subheader("Welcome Back!")
        
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Login", use_container_width=True, type="primary"):
                if login_email and login_password:
                    with st.spinner("Logging in..."):
                        user_data = login_user(login_email, login_password)
                        
                        if user_data:
                            st.session_state.logged_in = True
                            st.session_state.user_data = user_data
                            st.session_state.portfolio = user_data["portfolio"]
                            
                            ga_mp_send("user_login", {"email": login_email})
                            st.success("✅ Welcome back!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid email or password")
                else:
                    st.warning("Please enter both email and password")
    
    with tab2:
        st.subheader("Create Your Account")
        st.caption("Track stocks and get personalized insights")
        
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
        signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")
        
        st.caption("Password requirements: At least 6 characters")
        
        if st.button("Create Account", use_container_width=True, type="primary"):
            if not signup_email or "@" not in signup_email:
                st.error("❌ Please enter a valid email")
            elif len(signup_password) < 6:
                st.error("❌ Password must be at least 6 characters")
            elif signup_password != signup_password_confirm:
                st.error("❌ Passwords don't match")
            else:
                with st.spinner("Creating your account..."):
                    success = create_user(signup_email, signup_password, [])
                    
                    if success:
                        user_data = login_user(signup_email, signup_password)
                        if user_data:
                            st.session_state.logged_in = True
                            st.session_state.user_data = user_data
                            st.session_state.portfolio = []
                            
                            ga_mp_send("user_signup", {"email": signup_email})
                            st.success("✅ Account created! Welcome to IntelliInvest!")
                            st.balloons()
                            st.rerun()
                    else:
                        st.error("❌ Email already registered. Please login instead.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#9CA3AF;font-size:0.8rem;padding:1rem'>
        ⚠️ <strong>Disclaimer:</strong> Educational purposes only. Not financial advice.
        Always consult a licensed financial advisor.<br><br>
        Built by ANGAD ARORA
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()

# --------------------------------------------------------------------------------------
# MAIN APP (Only shown when logged in)
# --------------------------------------------------------------------------------------
st.sidebar.title("📊 IntelliInvest")

st.sidebar.info(f"""
👤 **{st.session_state.user_data['email']}**

📊 Tracking {len(st.session_state.portfolio)} stocks
""")

if st.sidebar.button("🚪 Logout", use_container_width=True):
    logout()
    st.rerun()

st.sidebar.markdown("---")

app_mode = st.sidebar.radio("", ["🔍 Stock Selection", "📊 Multi-Stock Comparison",
                                  "🏆 Top Undervalued Stocks", "💼 My Portfolio"])

# --------------------------------------------------------------------------------------
# STOCK SELECTION PAGE
# --------------------------------------------------------------------------------------
if app_mode == "🔍 Stock Selection":
    st.title("🔍 Stock Selection & Analysis")
    
    company_dict = pd.Series(sp500_df["Security"].values, index=sp500_df["Symbol"]).to_dict()
    full_list = {**index_dict, **company_dict}
    
    selected = st.selectbox("Select Stock", list(full_list.keys()), 
                           index=list(full_list.keys()).index("AAPL") if "AAPL" in full_list else 0,
                           format_func=lambda s: f"{full_list.get(s,s)} ({s})")
    
    st.markdown("---")
    
    if selected not in index_dict:
        company = full_list.get(selected, selected)
        sector = sp500_df[sp500_df["Symbol"] == selected]["Sector"].iloc[0] \
            if selected in sp500_df["Symbol"].values else "Unknown"
        
        st.subheader(f"{company} ({selected})")
        st.caption(f"Sector: {sector}")
        
        # FUNDAMENTALS
        with st.expander("📊 Fundamentals", expanded=True):
            fund = get_fundamentals_cached(selected)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Revenue (B)", f"${fund['rev']:.1f}B")
            c2.metric("Margin", f"{fund['margin']:.1f}%")
            c3.metric("EPS", f"${fund['eps']:.2f}")
            c4.metric("P/E", f"{fund['pe']:.1f}")
            c5.metric("P/B", f"{fund['pb']:.2f}")
            c6.metric("Div Yield", f"{fund['div']:.2f}%")
        
        # ANALYST RATINGS
        with st.expander("👥 Analyst Ratings", expanded=True):
            ratings = get_analyst_ratings_cached(selected)
            if ratings["n"]:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🟢 Buy", f"{ratings['buy']:.0f}%")
                c2.metric("🟡 Hold", f"{ratings['hold']:.0f}%")
                c3.metric("🔴 Sell", f"{ratings['sell']:.0f}%")
                c4.metric("📊 Total", ratings['n'])
                st.caption(f"**Avg Rating:** {ratings['avg']:.1f}/5 (1=Strong Buy, 5=Strong Sell)")
            else:
                st.info("No analyst ratings available")
        
        # NEWS SENTIMENT
        if news_api_key and NewsApiClient:
            with st.expander("📰 News Sentiment (VADER Analysis)", expanded=False):
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
        
        # PRICE TARGETS
        with st.expander("🎯 Price Targets", expanded=True):
            pt = get_price_targets_cached(selected)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💵 Current", fmt_usd(pt["last"]))
            c2.metric("🎯 Mean", fmt_usd(pt["mean"]))
            c3.metric("📊 High/Low", f"{fmt_usd(pt['high'])}/{fmt_usd(pt['low'])}")
            c4.metric("📈 Upside", fmt_pct(pt["upside"]))
            if pt["n"]:
                st.caption(f"**{pt['n']} analysts**")
        
        # PRICE HISTORY
        with st.expander("📈 Price History", expanded=False):
            hist = yf.Ticker(selected).history(period="5y")
            if not hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'],
                    mode='lines', line=dict(color='#3B82F6', width=2)))
                fig.update_layout(title=f"{selected} - 5Y", xaxis_title="Date",
                    yaxis_title="Price", height=400)
                st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# MULTI-STOCK COMPARISON PAGE
# --------------------------------------------------------------------------------------
elif app_mode == "📊 Multi-Stock Comparison":
    st.title("📊 Multi-Stock Comparison")
    
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

# --------------------------------------------------------------------------------------
# TOP UNDERVALUED STOCKS PAGE
# --------------------------------------------------------------------------------------
elif app_mode == "🏆 Top Undervalued Stocks":
    st.title("🏆 Top Undervalued Stocks")
    
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

    df = df.sort_values("undervaluation_score", ascending=True).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    out = df.head(top_n).copy()

    if targets and not out.empty:
        with st.spinner("Loading targets..."):
            out["Last"] = None; out["PT Mean"] = None; out["PT High"] = None; out["PT Low"] = None; out["Upside"] = None
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
        if c in disp.columns: disp[c] = disp[c].apply(fmt_usd)
    if "Upside" in disp.columns: disp["Upside"] = disp["Upside"].apply(fmt_pct)

    st.subheader(f"Top {top_n} Most Undervalued Stocks {f'in {sector}' if sector != 'All' else ''}")
    st.dataframe(disp, use_container_width=True, height=600, hide_index=True)

    if not out.empty:
        fig = px.bar(out, x="Symbol", y="undervaluation_score", title="Undervaluation Scores (Lower = Better Value)",
            color="undervaluation_score", color_continuous_scale=["#10B981","#FCD34D","#EF4444"])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.download_button("⬇️ Download CSV", disp.to_csv(index=False),
        f"top_undervalued_{sector.replace(' ','_')}_top{top_n}.csv", "text/csv")

# --------------------------------------------------------------------------------------
# MY PORTFOLIO PAGE
# --------------------------------------------------------------------------------------
elif app_mode == "💼 My Portfolio":
    st.title("💼 My Portfolio")
    
    st.markdown(f"""
    Welcome back, **{st.session_state.user_data['email']}**! 
    Track your favorite stocks and monitor their valuations.
    """)
    
    # Add stock section
    st.markdown("### ➕ Add Stock to Track")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        all_symbols = sp500_df["Symbol"].tolist()
        ticker_input = st.selectbox(
            "Select a stock",
            [""] + all_symbols,
            format_func=lambda x: f"{x} - {sp500_df[sp500_df['Symbol']==x]['Security'].iloc[0]}" if x and x in all_symbols else "Select a stock..."
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Add", use_container_width=True, type="primary"):
            if ticker_input and ticker_input not in st.session_state.portfolio:
                st.session_state.portfolio.append(ticker_input)
                update_user_portfolio(st.session_state.user_data['email'], st.session_state.portfolio)
                st.success(f"✅ Added {ticker_input}")
                ga_mp_send("portfolio_add", {"ticker": ticker_input})
                st.rerun()
            elif ticker_input in st.session_state.portfolio:
                st.warning(f"⚠️ {ticker_input} already in portfolio")
    
    st.markdown("---")
    
    # Display portfolio
    if st.session_state.portfolio:
        st.markdown("### 📊 Your Tracked Stocks")
        
        for ticker in st.session_state.portfolio:
            try:
                price = get_current_price(ticker)
                
                if not val_df.empty and ticker in val_df["Symbol"].values:
                    score = val_df[val_df["Symbol"] == ticker]["undervaluation_score"].iloc[0]
                else:
                    score = 5.0
                
                company = sp500_df[sp500_df["Symbol"] == ticker]["Security"].iloc[0] if ticker in sp500_df["Symbol"].values else ticker
                
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{ticker}** - {company}")
                
                with col2:
                    if score <= 3.5:
                        color, label = "🟢", "Undervalued"
                    elif score <= 6.5:
                        color, label = "🟡", "Fair Value"
                    else:
                        color, label = "🔴", "Overvalued"
                    st.metric("Valuation", f"{color} {score}/10")
                    st.caption(label)
                
                with col3:
                    st.metric("Price", fmt_usd(price))
                
                with col4:
                    st.write("")
                    if st.button("🗑️", key=f"del_{ticker}"):
                        st.session_state.portfolio.remove(ticker)
                        update_user_portfolio(st.session_state.user_data['email'], st.session_state.portfolio)
                        ga_mp_send("portfolio_remove", {"ticker": ticker})
                        st.rerun()
                
                st.divider()
            except:
                pass
        
        # Email preferences
        st.markdown("### 📧 Email Preferences")
        st.info("""
        ✅ **You're subscribed to monthly updates!**
        
        You'll receive monthly emails with:
        - Valuation changes for your tracked stocks
        - Undervalued opportunities
        - Price target updates
        """)
        
    else:
        st.info("👆 Add stocks above to start tracking")

# --------------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#9CA3AF;font-size:0.8rem;padding:0.5rem'>
    Built by ANGAD ARORA | 
    <a href="https://github.com/angadarora2024" style="color:#9CA3AF">GitHub</a> | 
    <a href="https://linkedin.com/in/angadarora" style="color:#9CA3AF">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
