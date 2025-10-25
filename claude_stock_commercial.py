#!/usr/bin/env python3
"""S&P 500 Stock Analyzer
   - VADER Sentiment
   - GA4 Measurement Protocol (server-side)
   - PostHog client (bridged) + server fallback
   - OPTIONAL: Portfolio tracking with login (4th page only)
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
st.set_page_config(page_title="S&P 500 Analyzer", page_icon="📊", layout="wide")

# --------------------------------------------------------------------------------------
# ANALYTICS CONFIG (edit these or set via st.secrets)
# --------------------------------------------------------------------------------------
# Toggle analytics modules
ENABLE_GA_MP = True                 # GA4 Measurement Protocol (server-side)
ENABLE_POSTHOG_CLIENT = True        # PostHog client (JS) via components bridge
ENABLE_POSTHOG_SERVER = True        # PostHog server-side capture

# Prefer secrets if available
GA_MEASUREMENT_ID = st.secrets.get("GA_MEASUREMENT_ID", "G-598BZYJEBM")  # <- your GA4 MEASUREMENT ID
GA_API_SECRET     = st.secrets.get("GA_API_SECRET", "PUT_YOUR_GA_API_SECRET")  # <- create in GA4: Admin > Data Streams > Measurement Protocol API secret

POSTHOG_KEY  = st.secrets.get("POSTHOG_KEY",  "phc_your_project_key_here")
POSTHOG_HOST = st.secrets.get("POSTHOG_HOST", "https://app.posthog.com")   # or your self-hosted domain

APP_URL = "https://intellinvest.streamlit.app/"

# --------------------------------------------------------------------------------------
# ANALYTICS HELPERS
# --------------------------------------------------------------------------------------
def _ensure_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]

def ga_mp_send(event_name: str, params: dict):
    """Send GA4 event via Measurement Protocol (server-side)."""
    if not ENABLE_GA_MP:
        return
    try:
        cid = _ensure_session_id()
        url = f"https://www.google-analytics.com/mp/collect?measurement_id={GA_MEASUREMENT_ID}&api_secret={GA_API_SECRET}"
        payload = {
            "client_id": cid,
            "events": [{"name": event_name, "params": params}],
        }
        requests.post(url, json=payload, timeout=4)
    except Exception:
        pass  # keep silent in prod

def posthog_server_capture(event_name: str, properties: dict):
    """Server-side event to PostHog."""
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
                      headers={"Content-Type": "application/json"},
                      timeout=4)
    except Exception:
        pass

def posthog_client_boot():
    """Load PostHog client inside a tiny components iframe and bridge it to the app window."""
    if not ENABLE_POSTHOG_CLIENT:
        return
    components.html(f"""
<!doctype html><html><head><meta charset="utf-8"></head><body>
<script>
  (function() {{
    try {{
      if (!window.parent.posthog) {{
        // Load lightweight PostHog loader in this iframe
        var s = document.createElement('script'); s.async = true;
        s.src = '{POSTHOG_HOST}/static/array.js';
        s.onload = function() {{
          try {{
            window.posthog = window.posthog || [];
            window.posthog.init('{POSTHOG_KEY}', {{ api_host: '{POSTHOG_HOST}' }});
            window.parent.posthog = window.posthog; // bridge to parent
            console.log('[PostHog] client initialized & bridged');
          }} catch(e) {{ console.error('[PostHog] init error', e); }}
        }};
        document.head.appendChild(s);
      }} else {{
        console.log('[PostHog] already available on parent');
      }}
    }} catch (e) {{
      console.warn('[PostHog] boot error', e);
    }}
  }})();
</script>
</body></html>
""", height=0)

# Initialize client analytics (safe on every rerun)
posthog_client_boot()

# Send a one-time "app_start" (per session) + GA page_view
if st.session_state.get("_sent_app_start") is None:
    st.session_state["_sent_app_start"] = True
    # GA page_view (server)
    ga_mp_send("page_view", {"page_title": "SP500 Analyzer", "page_location": APP_URL})
    # PostHog (server)
    posthog_server_capture("app_start", {"app": "sp500_analyzer"})
    # PostHog (client) via bridged call
    components.html("""
<!doctype html><html><head><meta charset="utf-8"></head><body>
<script>
  (function fire(){
    function send(){
      try {
        if (window.parent && window.parent.posthog) {
          window.parent.posthog.capture('app_start', {app: 'sp500_analyzer'});
          return true;
        }
      } catch(e){}
      return false;
    }
    if(!send()){
      let t=0; const it=setInterval(function(){
        t++; if (send() || t>10) clearInterval(it);
      }, 300);
    }
  })();
</script>
</body></html>
""", height=0)

# --------------------------------------------------------------------------------------
# LIGHT THEME / STYLES (unchanged)
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
[data-testid="stAppViewContainer"], [data-testid="stApp"] { background: #F9FAFB !important; }
html,body,.stApp{font-family:'Inter',sans-serif!important;background:#F9FAFB}
header[data-testid="stHeader"], .stApp > header{background:#fff!important}
button[kind="header"], button[data-testid="baseButton-header"]{color:#1F2937!important}
button[data-testid="baseButton-header"] svg{color:#1F2937!important;fill:#1F2937!important}
section[data-testid="stSidebarNav"]{background:#fff!important}
div[data-testid="stSidebarNavItems"]{background:#fff!important}
div[data-testid="stSidebarNavItems"] a, div[data-testid="stSidebarNavItems"] span{color:#1F2937!important}
h1{font-size:2.5rem!important;font-weight:700!important;background:linear-gradient(135deg,#667eea,#764ba2);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
h2,h3{color:#1F2937!important}
[data-baseweb="select"]{background:#fff!important}
[data-baseweb="select"] > div{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
[data-baseweb="select"] [role="button"], [data-baseweb="select"] input, [data-baseweb="select"] span,
[data-baseweb="select"] svg{background:#fff!important;color:#1F2937!important;fill:#1F2937!important}
.stSelectbox > div > div{background:#fff!important;color:#1F2937!important}
.stSelectbox label{color:#1F2937!important}
.stMultiSelect > div > div, .stMultiSelect [data-baseweb="select"]{background:#fff!important;color:#1F2937!important}
.stMultiSelect label, .stMultiSelect span{color:#1F2937!important}
.stMultiSelect [data-baseweb="tag"]{background:#E5E7EB!important;color:#1F2937!important}
.stNumberInput > div > div, .stNumberInput input{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
.stDateInput > div > div, .stDateInput input{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
.stTextInput > div > div, .stTextInput input{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
.stCheckbox, .stCheckbox > label, .stCheckbox span{color:#1F2937!important}
.stRadio > label, .stRadio [role="radiogroup"] label, .stRadio [role="radiogroup"] span{color:#1F2937!important}
[data-testid="stMetric"]{background:#fff;padding:1rem;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.1);border:1px solid #E5E7EB}
[data-testid="stMetricValue"]{font-size:1.75rem!important;font-weight:700!important;color:#1F2937!important}
[data-testid="stMetricLabel"]{color:#6B7280!important}
div[data-testid="stExpander"]{background:#fff!important;border-radius:12px;border:1px solid #E5E7EB;margin-bottom:1rem}
div[data-testid="stExpander"] summary{background:#fff!important;color:#1F2937!important;padding:1rem!important}
div[data-testid="stExpander"] summary:hover{background:#F3F4F6!important}
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"]{background:#fff!important;padding:1rem!important}
.stButton>button{background:linear-gradient(135deg,#667eea,#764ba2)!important;color:#fff!important;
border:none!important;border-radius:8px!important;padding:.5rem 1.5rem!important;font-weight:600!important}
.badge{display:inline-block;padding:.25rem .75rem;border-radius:9999px;font-size:.75rem;font-weight:600;text-transform:uppercase}
.badge-success{background:#D1FAE5;color:#065F46}
.badge-warning{background:#FEF3C7;color:#92400E}
.badge-danger{background:#FEE2E2;color:#991B1B}
.compact-metric [data-testid="stMetricValue"]{font-size:1.2rem!important}
p, span, div, label{color:#1F2937!important}
[data-testid="stCaption"]{color:#6B7280!important}
section[data-testid="stSidebar"]{background:#fff!important}
section[data-testid="stSidebar"] *{color:#1F2937!important}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span{color:#1F2937!important}
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"]{color:#1F2937!important}
section[data-testid="stSidebar"] label[data-baseweb="radio"]{color:#1F2937!important}
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div{color:#1F2937!important}
input, textarea{background:#fff!important;color:#1F2937!important;border:1px solid #E5E7EB!important}
div[data-baseweb="input"] > div{background:#fff!important}
div[data-baseweb="input"] input{color:#1F2937!important}
@media (max-width: 768px) {
  h1{font-size:1.75rem!important}
  h2{font-size:1.5rem!important}
  h3{font-size:1.25rem!important}
  [data-testid="stMetric"]{padding:0.75rem!important}
  [data-testid="stMetricValue"]{font-size:1.5rem!important}
  [role="listbox"], ul, li{background:#fff!important;color:#1F2937!important}
}
body > div[class*="layer"], #root > div[class*="layer"], [data-baseweb="layer"]{ background:transparent!important; }
[data-baseweb="menu"]{background:#fff!important}
[data-baseweb="menu"] ul{background:#fff!important}
[data-baseweb="menu"] li{background:#fff!important;color:#1F2937!important}
</style>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# AUTHENTICATION FUNCTIONS (for Portfolio page only)
# --------------------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash: str, provided_password: str) -> bool:
    return stored_hash == hash_password(provided_password)

def get_google_sheet():
    if not GSPREAD_AVAILABLE:
        return None
    try:
        creds_dict = st.secrets.get("gspread", None)
        if not creds_dict:
            return None
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        try:
            sheet = client.open("IntelliInvest_Users").sheet1
        except:
            spreadsheet = client.create("IntelliInvest_Users")
            sheet = spreadsheet.sheet1
            sheet.append_row(["Email", "Password_Hash", "Portfolio", "User_ID", "Created_At", "Last_Login", "Status"])
        return sheet
    except:
        return None

def create_user(email: str, password: str, portfolio: list) -> bool:
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
    except:
        return False

def login_user(email: str, password: str) -> dict:
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
        return {"email": email, "portfolio": portfolio, "user_id": row_values[3], "created_at": row_values[4], "row_num": cell.row}
    except:
        return None

def update_user_portfolio(email: str, portfolio: list) -> bool:
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
    except:
        return False

# OAuth Functions
def generate_oauth_url():
    """Generate Google OAuth URL"""
    try:
        oauth_config = st.secrets.get("oauth", {})
        if not oauth_config:
            return None
        
        client_id = oauth_config.get("google_client_id")
        redirect_uri = oauth_config.get("redirect_uri")
        
        if not client_id or not redirect_uri:
            return None
        
        import urllib.parse
        
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "select_account"
        }
        
        query_string = urllib.parse.urlencode(params)
        return f"https://accounts.google.com/o/oauth2/v2/auth?{query_string}"
    except:
        return None

def exchange_code_for_token(code):
    """Exchange OAuth code for user info"""
    try:
        oauth_config = st.secrets.get("oauth", {})
        if not oauth_config:
            return None
        
        # Exchange code for token
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "code": code,
            "client_id": oauth_config.get("google_client_id"),
            "client_secret": oauth_config.get("google_client_secret"),
            "redirect_uri": oauth_config.get("redirect_uri"),
            "grant_type": "authorization_code"
        }
        
        response = requests.post(token_url, data=data)
        token_data = response.json()
        
        if "access_token" not in token_data:
            return None
        
        # Get user info
        userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        user_response = requests.get(userinfo_url, headers=headers)
        user_info = user_response.json()
        
        return {
            "email": user_info.get("email"),
            "name": user_info.get("name"),
            "picture": user_info.get("picture")
        }
    except Exception as e:
        return None

def get_or_create_oauth_user(email, name=None):
    """Get existing user or create new one for OAuth login"""
    sheet = get_google_sheet()
    if not sheet:
        return None
    
    try:
        records = sheet.get_all_records()
        
        # Check if user exists
        user = next((u for u in records if u['Email'].lower() == email.lower()), None)
        
        if user:
            # Update last login
            for idx, record in enumerate(records, start=2):
                if record['Email'].lower() == email.lower():
                    sheet.update_cell(idx, 6, datetime.now().isoformat())  # Last_Login
                    break
            
            portfolio_str = user.get('Portfolio', '')
            return {
                "email": user['Email'],
                "portfolio": portfolio_str.split(',') if portfolio_str else [],
                "user_id": user.get('User_ID', str(uuid.uuid4())),
                "auth_method": "oauth"
            }
        else:
            # Create new user with OAuth
            user_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            new_row = [
                email,
                "OAUTH_USER",  # No password for OAuth users
                "",  # Empty portfolio
                user_id,
                now,  # Created_At
                now,  # Last_Login
                "active"
            ]
            
            sheet.append_row(new_row)
            
            return {
                "email": email,
                "portfolio": [],
                "user_id": user_id,
                "auth_method": "oauth"
            }
    except Exception as e:
        return None

def init_auth_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_data" not in st.session_state:
        st.session_state.user_data = None
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

def logout():
    st.session_state.logged_in = False
    st.session_state.user_data = None
    st.session_state.portfolio = []

DEFAULT_VAL_PATH = "val_output/undervaluation_scored.csv"

# --------------------------------------------------------------------------------------
# DATA HELPERS (unchanged)
# --------------------------------------------------------------------------------------
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
            pt = t.analyst_price_targets or t.get_analyst_price_targets() or {}
        except:
            pt = {}
        mean = pt.get("mean") or pt.get("targetMeanPrice")
        low = pt.get("low") or pt.get("targetLowPrice")
        high = pt.get("high") or pt.get("targetHighPrice")
        n = pt.get("numberOfAnalysts") or pt.get("numAnalysts")
        upside = (mean/last-1) if (mean and last) else None
        return {"last": last, "mean": mean, "high": high, "low": low,
                "n": int(n) if n else None, "upside": upside}
    except:
        return {"last": None, "mean": None, "high": None, "low": None, "n": None, "upside": None}

def fmt_pct(x):   return f"{float(x):.2%}"  if x is not None and not pd.isna(x) else "N/A"
def fmt_usd(x):   return f"${float(x):,.2f}" if x is not None and not pd.isna(x) else "N/A"
def fmt_float(x): return f"{float(x):.2f}"   if x is not None and not pd.isna(x) else "N/A"

# --------------------------------------------------------------------------------------
# APP UI
# --------------------------------------------------------------------------------------
st.title("🔍 Key Stock Investment Metrics")
st.caption("AI-Powered Valuation, Price Targets & Sentiment Analysis")

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
    • Streamlit • yfinance • Plotly • VADER • NewsAPI • pandas, numpy<br><br>
    <strong>Data Sources:</strong><br>
    • Stock prices & analyst ratings: Yahoo Finance<br>
    • News: NewsAPI.org<br>
    • S&P 500 list: Wikipedia
</div>
</details>
""", unsafe_allow_html=True)

st.sidebar.header("🧭 Navigation")
app_mode = st.sidebar.radio("Mode", ("Single Stock Analysis", "Multi-Stock Comparison", "Top Undervalued Stocks", "My Portfolio")

)

# Initialize auth session
init_auth_session()

# Show user info if logged in
if st.session_state.logged_in and app_mode == "My Portfolio":
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    👤 **{st.session_state.user_data['email']}**
    📊 Tracking {len(st.session_state.portfolio)} stocks
    """)
    if st.sidebar.button("🚪 Logout"):
        logout()
        st.rerun()

# Get API keys (hidden from UI)
try:
    news_api_key = st.secrets.get("NEWS_API_KEY")
except:
    news_api_key = None

val_path = DEFAULT_VAL_PATH
sp500_df = get_sp500_data()
index_dict = {"^GSPC": "S&P 500", "^NDX": "Nasdaq-100"}

# --------------------------------------------------------------------------------------
# SINGLE STOCK ANALYSIS
# --------------------------------------------------------------------------------------
if app_mode == "Single Stock Analysis":
    st.subheader("🔍 Stock Selection")
    company_dict = pd.Series(sp500_df["Security"].values, index=sp500_df["Symbol"]).to_dict()
    full_list = {**index_dict, **company_dict}

    selected = st.selectbox(
        "Select Stock",
        list(full_list.keys()),
        index=list(full_list.keys()).index("AAPL") if "AAPL" in full_list else 0,
        format_func=lambda s: f"{full_list.get(s,s)} ({s})"
    )

    st.markdown("---")
    st.session_state.selected_symbol = selected

    # Deduped analytics on selection change
    if "last_tracked_symbol" not in st.session_state:
        st.session_state["last_tracked_symbol"] = None

    if selected and st.session_state["last_tracked_symbol"] != selected:
        st.session_state["last_tracked_symbol"] = selected

        # GA4 (server)
        ga_mp_send("stock_view", {"stock_symbol": selected, "mode": "single_stock"})

        # PostHog (server)
        posthog_server_capture("stock_view", {"stock_symbol": selected, "mode": "single_stock"})

        # PostHog (client) via bridged call
        components.html(f"""
<!doctype html><html><head><meta charset="utf-8"></head><body>
<script>
  (function send(){{
    function go(){{
      try {{
        if (window.parent && window.parent.posthog) {{
          window.parent.posthog.capture('stock_view', {{
            stock_symbol: '{selected}', mode: 'single_stock'
          }});
          return true;
        }}
      }} catch(e){{}}
      return false;
    }}
    if (!go()) {{
      let tries=0; const t=setInterval(function(){{
        tries++; if (go()||tries>10) clearInterval(t);
      }}, 300);
    }}
  }})();
</script>
</body></html>
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

        # VALUATION & ANALYST RATINGS
        if selected not in index_dict:
            col_left, col_right = st.columns(2)

            with col_left:
                try:
                    val_df = load_valuation_scores(val_path)
                    st.subheader("📊 Valuation Gauge")
                    if val_df.empty or "Symbol" not in val_df.columns:
                        st.info("Valuation data not available")
                    else:
                        row = val_df[val_df["Symbol"].str.upper() == selected.upper()]
                        if not row.empty and "undervaluation_score" in row.columns:
                            score = float(row.iloc[0]["undervaluation_score"])
                            sector_name = row.iloc[0].get("Sector", row.iloc[0].get("GICS Sector", "Unknown"))
                            badge = "success" if score <= 3 else "warning" if score <= 7 else "danger"
                            label = "Undervalued" if score <= 3 else "Fairly Valued" if score <= 7 else "Overvalued"
                            st.markdown(f'<span class="badge badge-{badge}">{label}</span>', unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="margin-top:1rem">
                              <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#6B7280;margin-bottom:0.25rem">
                                <span>Undervalued</span><span>Fairly Valued</span><span>Overvalued</span>
                              </div>
                              <div style="width:100%;height:8px;background:linear-gradient(to right, #10B981, #FCD34D, #EF4444);border-radius:4px;position:relative">
                                <div style="position:absolute;left:{(score-1)/9*100}%;top:-4px;width:16px;height:16px;background:#1F2937;border-radius:50%;border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.2)"></div>
                              </div>
                              <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#9CA3AF;margin-top:0.5rem">
                                <span>1</span><span>5</span><span>10</span>
                              </div>
                            </div>
                            <div style='margin-top:1rem;text-align:center;padding:1rem;background:#F9FAFB;border-radius:8px'>
                              <div style='font-size:2rem;font-weight:700;color:#1F2937'>{score:.1f}<span style='font-size:1.2rem;color:#6B7280'>/10</span></div>
                              <div style='font-size:0.9rem;color:#6B7280;margin-top:0.5rem'>Sector: {sector_name}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("ℹ️ How is this calculated?"):
                                st.markdown("""
**Undervaluation Score (1–10)** combines:
- P/E vs sector, P/B vs history, P/S vs peers
- PEG, EV/EBITDA, EV/Sales
- Dividend yield vs peers

1–3: Undervalued · 4–7: Fair · 8–10: Overvalued
*Not financial advice.*
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
                        rating_values = {"strongBuy": 1, "buy": 2, "hold": 3, "sell": 4, "strongSell": 5}
                        total_ratings, weighted_sum = 0, 0
                        for key, val in rating_values.items():
                            if key in s.index and s[key] > 0:
                                cnt = int(s[key]); total_ratings += cnt; weighted_sum += cnt * val
                        if total_ratings > 0:
                            avg = weighted_sum / total_ratings
                            st.markdown(f"""
                            <div style="margin-top:1rem">
                              <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#6B7280;margin-bottom:0.25rem">
                                <span>Strong Buy</span><span>Hold</span><span>Strong Sell</span>
                              </div>
                              <div style="width:100%;height:12px;background:linear-gradient(to right,#10B981,#34D399,#FCD34D,#FB923C,#EF4444);border-radius:6px;position:relative">
                                <div style="position:absolute;left:{(avg-1)/4*100}%;top:-2px;width:20px;height:20px;background:#1F2937;border-radius:50%;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3)"></div>
                              </div>
                              <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#9CA3AF;margin-top:0.5rem">
                                <span>1</span><span>2</span><span>3</span><span>4</span><span>5</span>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                            if   avg <= 1.5: label, color = "Strong Buy", "#10B981"
                            elif avg <= 2.5: label, color = "Buy", "#34D399"
                            elif avg <= 3.5: label, color = "Hold", "#FCD34D"
                            elif avg <= 4.5: label, color = "Sell", "#FB923C"
                            else:           label, color = "Strong Sell", "#EF4444"
                            st.markdown(f"""
                            <div style='margin-top:1rem;text-align:center;padding:1rem;background:#F9FAFB;border-radius:8px'>
                              <div style='font-size:1.75rem;font-weight:700;color:{color}'>{label}</div>
                              <div style='font-size:0.9rem;color:#6B7280;margin-top:0.5rem'>
                                Avg Rating: {avg:.2f} | {total_ratings} analysts
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("📊 Rating Breakdown"):
                                for key in ["strongBuy","buy","hold","sell","strongSell"]:
                                    if key in s.index and s[key] > 0:
                                        st.write(f"**{key.replace('strong','Strong ')}:** {int(s[key])}")
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
            hist = yf.Ticker(selected).history(period="5y")
            if not hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'],
                    mode='lines', line=dict(color='#3B82F6', width=2)))
                fig.update_layout(title=f"{selected} - 5Y", xaxis_title="Date",
                    yaxis_title="Price", height=400, margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)

        # NEWS ARTICLES
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

# --------------------------------------------------------------------------------------
# MULTI-STOCK COMPARISON
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# TOP UNDERVALUED STOCKS
# --------------------------------------------------------------------------------------
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
# MY PORTFOLIO (NEW - OPTIONAL 4TH PAGE)
# --------------------------------------------------------------------------------------
elif app_mode == "My Portfolio":
    st.title("💼 My Portfolio")
    
    # Handle OAuth callback
    query_params = st.query_params
    if "code" in query_params and not st.session_state.logged_in:
        code = query_params["code"]
        user_info = exchange_code_for_token(code)
        
        if user_info and user_info.get("email"):
            user_data = get_or_create_oauth_user(user_info["email"], user_info.get("name"))
            
            if user_data:
                st.session_state.logged_in = True
                st.session_state.user_data = user_data
                st.session_state.portfolio = user_data["portfolio"]
                # Clear query params
                st.query_params.clear()
                st.success("✅ Logged in with Google!")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Could not create user account")
        else:
            st.error("❌ OAuth login failed")
    
    # If not logged in, show login/signup
    if not st.session_state.logged_in:
        st.info("""
        👋 **Welcome to Portfolio Tracking!**
        
        Create a free account to:
        - Track your favorite stocks
        - Save portfolio across devices
        - Get monthly email updates
        """)
        
        tab1, tab2, tab3 = st.tabs(["🔐 Login", "✨ Sign Up", "🔑 Forgot Password"])
        
        with tab1:
            st.subheader("Welcome Back!")
            
            # OAuth Login Button
            oauth_url = generate_oauth_url()
            if oauth_url:
                st.markdown("#### 🚀 Quick Login")
                if st.button("🔐 Login with Google", type="primary", use_container_width=True):
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={oauth_url}">', unsafe_allow_html=True)
                    st.info("Redirecting to Google...")
                
                st.markdown("---")
                st.markdown("#### 📧 Or Login with Email")
            
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", type="secondary" if oauth_url else "primary"):
                if login_email and login_password:
                    user_data = login_user(login_email, login_password)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.user_data = user_data
                        st.session_state.portfolio = user_data["portfolio"]
                        st.success("✅ Welcome back!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password")
                else:
                    st.warning("Please enter both fields")
        
        with tab2:
            st.subheader("Create Your Account")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
            signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")
            
            if st.button("Create Account", type="primary"):
                if not signup_email or "@" not in signup_email:
                    st.error("❌ Please enter a valid email")
                elif len(signup_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                elif signup_password != signup_password_confirm:
                    st.error("❌ Passwords don't match")
                else:
                    success = create_user(signup_email, signup_password, [])
                    if success:
                        user_data = login_user(signup_email, signup_password)
                        if user_data:
                            st.session_state.logged_in = True
                            st.session_state.user_data = user_data
                            st.session_state.portfolio = []
                            st.success("✅ Account created!")
                            st.balloons()
                            st.rerun()
                    else:
                        st.error("❌ Email already registered")
        
        with tab3:
            st.subheader("🔑 Reset Your Password")
            st.info("💡 Enter your email to receive a temporary password.")
            
            reset_email = st.text_input("Email Address", key="reset_email")
            
            if st.button("Send Reset Email", type="primary"):
                if reset_email and "@" in reset_email:
                    sheet = get_google_sheet()
                    if sheet:
                        try:
                            records = sheet.get_all_records()
                            user = next((u for u in records if u['Email'].lower() == reset_email.lower()), None)
                            
                            if user:
                                # Generate temporary password
                                import random
                                import string
                                temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
                                temp_password_hash = hashlib.sha256(temp_password.encode()).hexdigest()
                                
                                # Update password in sheet
                                for idx, record in enumerate(records, start=2):
                                    if record['Email'].lower() == reset_email.lower():
                                        sheet.update_cell(idx, 2, temp_password_hash)
                                        break
                                
                                # Send email
                                try:
                                    import smtplib
                                    from email.mime.text import MIMEText
                                    from email.mime.multipart import MIMEMultipart
                                    
                                    email_config = st.secrets.get("email", {})
                                    
                                    if email_config:
                                        msg = MIMEMultipart()
                                        msg['From'] = email_config["sender_email"]
                                        msg['To'] = reset_email
                                        msg['Subject'] = f"{email_config.get('app_name', 'App')} - Password Reset"
                                        
                                        body = f"""Hello,

Your password has been reset for {email_config.get('app_name', 'our app')}.

Your temporary password is: {temp_password}

Please login and change your password immediately for security.

If you didn't request this reset, please contact support immediately.

Best regards,
{email_config.get('app_name', 'App')} Team
"""
                                        
                                        msg.attach(MIMEText(body, 'plain'))
                                        
                                        # Send email
                                        server = smtplib.SMTP(email_config["smtp_server"], int(email_config["smtp_port"]))
                                        server.starttls()
                                        server.login(email_config["sender_email"], email_config["sender_password"])
                                        server.send_message(msg)
                                        server.quit()
                                        
                                        st.success(f"✅ Password reset email sent to {reset_email}!")
                                        st.info("📧 Please check your email (and spam folder) for the temporary password.")
                                    else:
                                        st.error("❌ Email service not configured")
                                        st.info("Please contact support for password reset.")
                                        
                                except Exception as e:
                                    st.error(f"❌ Could not send email: {str(e)}")
                                    st.warning("Please contact support for manual password reset.")
                            else:
                                # Security: Don't reveal if email exists
                                st.info(f"If an account exists for {reset_email}, a password reset email has been sent.")
                                st.caption("Please check your email (including spam folder).")
                        except Exception as e:
                            st.error("❌ Error processing request. Please try again later.")
                    else:
                        st.error("❌ Database connection error. Please try again later.")
                else:
                    st.warning("⚠️ Please enter a valid email address")
        st.stop()
    
    # USER IS LOGGED IN - show portfolio
    st.markdown(f"Welcome back, **{st.session_state.user_data['email']}**!")
    
    # DEBUG: Show if Google Sheets is working
    with st.expander("🔧 Debug: Check Database Connection"):
        if not GSPREAD_AVAILABLE:
            st.error("❌ gspread library not installed")
        else:
            st.success("✅ gspread library installed")
        
        creds = st.secrets.get("gspread", None)
        if not creds:
            st.error("❌ No Google Sheets credentials in secrets")
            st.info("💡 Your data is NOT being saved! Add gspread credentials to secrets.toml")
        else:
            st.success("✅ Credentials found in secrets")
            st.code(f"Service Account: {creds.get('client_email', 'N/A')}")
        
        sheet = get_google_sheet()
        if not sheet:
            st.error("❌ Cannot connect to Google Sheet")
            st.warning("Your portfolio is NOT being saved!")
        else:
            st.success("✅ Connected to Google Sheet!")
            try:
                url = sheet.spreadsheet.url
                st.code(url)
                st.markdown(f"[Open Sheet in Browser]({url})")
            except:
                st.error("Could not get sheet URL")
    
    st.markdown("---")
    
    # Add stock
    st.markdown("### ➕ Add Stock")
    col1, col2 = st.columns([4, 1])
    with col1:
        all_symbols = sp500_df["Symbol"].tolist()
        ticker_input = st.selectbox(
            "Select stock",
            [""] + all_symbols,
            format_func=lambda x: f"{x} - {sp500_df[sp500_df['Symbol']==x]['Security'].iloc[0]}" if x and x in all_symbols else "Select..."
        )
    with col2:
        st.write(""); st.write("")
        if st.button("Add", type="primary"):
            if ticker_input and ticker_input not in st.session_state.portfolio:
                st.session_state.portfolio.append(ticker_input)
                update_user_portfolio(st.session_state.user_data['email'], st.session_state.portfolio)
                st.success(f"✅ Added {ticker_input}")
                st.rerun()
            elif ticker_input in st.session_state.portfolio:
                st.warning(f"⚠️ Already in portfolio")
    
    st.markdown("---")
    
    # Display portfolio
    if st.session_state.portfolio:
        st.markdown(f"### 📊 Tracking {len(st.session_state.portfolio)} Stocks")
        
        # Performance Summary Table
        st.markdown("#### 📈 Performance Overview")
        
        performance_data = []
        
        for ticker in st.session_state.portfolio:
            try:
                t = yf.Ticker(ticker)
                
                # Get current price
                try:
                    current_price = t.fast_info.last_price
                except:
                    hist = t.history(period="1d")
                    current_price = float(hist["Close"][-1]) if not hist.empty else None
                
                if not current_price:
                    continue
                
                # Get company name
                company = sp500_df[sp500_df["Symbol"] == ticker]["Security"].iloc[0] if ticker in sp500_df["Symbol"].values else ticker
                
                # Get historical data for performance calculation
                hist_1y = t.history(period="1y")
                
                if hist_1y.empty:
                    continue
                
                # Calculate performance metrics
                # 1 Day
                try:
                    price_1d_ago = hist_1y["Close"][-2] if len(hist_1y) >= 2 else current_price
                    pct_1d = ((current_price / price_1d_ago) - 1) * 100
                except:
                    pct_1d = 0
                
                # 1 Week
                try:
                    price_1w_ago = hist_1y["Close"][-6] if len(hist_1y) >= 6 else current_price
                    pct_1w = ((current_price / price_1w_ago) - 1) * 100
                except:
                    pct_1w = 0
                
                # 1 Month
                try:
                    price_1m_ago = hist_1y["Close"][-22] if len(hist_1y) >= 22 else current_price
                    pct_1m = ((current_price / price_1m_ago) - 1) * 100
                except:
                    pct_1m = 0
                
                # YTD (Year to Date)
                try:
                    from datetime import datetime
                    year_start = datetime(datetime.now().year, 1, 1)
                    ytd_data = hist_1y[hist_1y.index >= year_start]
                    if not ytd_data.empty:
                        price_ytd = ytd_data["Close"].iloc[0]
                        pct_ytd = ((current_price / price_ytd) - 1) * 100
                    else:
                        pct_ytd = 0
                except:
                    pct_ytd = 0
                
                # Since Added (using 1 month as proxy - in future we'll track actual add date)
                pct_since_added = pct_1m
                
                performance_data.append({
                    "Symbol": ticker,
                    "Company": company,
                    "Price": current_price,
                    "1 Day": pct_1d,
                    "1 Week": pct_1w,
                    "1 Month": pct_1m,
                    "YTD": pct_ytd,
                    "Since Added": pct_since_added
                })
                
            except Exception as e:
                continue
        
        if performance_data:
            # Create DataFrame
            perf_df = pd.DataFrame(performance_data)
            
            # Display summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_1d = perf_df["1 Day"].mean()
                st.metric("Portfolio 1D", f"{avg_1d:+.2f}%", 
                         delta=f"{avg_1d:.2f}%")
            
            with col2:
                avg_1w = perf_df["1 Week"].mean()
                st.metric("Portfolio 1W", f"{avg_1w:+.2f}%",
                         delta=f"{avg_1w:.2f}%")
            
            with col3:
                avg_1m = perf_df["1 Month"].mean()
                st.metric("Portfolio 1M", f"{avg_1m:+.2f}%",
                         delta=f"{avg_1m:.2f}%")
            
            with col4:
                avg_ytd = perf_df["YTD"].mean()
                st.metric("Portfolio YTD", f"{avg_ytd:+.2f}%",
                         delta=f"{avg_ytd:.2f}%")
            
            st.markdown("---")
            
            # Display detailed table
            st.markdown("#### 📋 Performance Table")
            
            # Format for display
            display_df = perf_df.copy()
            display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:,.2f}")
            
            for col in ["1 Day", "1 Week", "1 Month", "YTD", "Since Added"]:
                display_df[col] = display_df[col].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=min(400, len(display_df) * 35 + 38)
            )
            
            # Chart: Performance Comparison
            st.markdown("#### 📊 Performance by Period")
            
            chart_data = perf_df[["Symbol", "1 Day", "1 Week", "1 Month", "YTD"]].set_index("Symbol")
            
            fig = go.Figure()
            
            colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
            periods = ["1 Day", "1 Week", "1 Month", "YTD"]
            
            for i, period in enumerate(periods):
                fig.add_trace(go.Bar(
                    name=period,
                    x=chart_data.index,
                    y=chart_data[period],
                    marker_color=colors[i]
                ))
            
            fig.update_layout(
                barmode='group',
                title="",
                xaxis_title="Stock",
                yaxis_title="Return (%)",
                height=400,
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        
        # Individual Stock Cards with Remove Button
        st.markdown("### 🏢 Manage Stocks")
        
        for ticker in st.session_state.portfolio:
            try:
                price = yf.Ticker(ticker).fast_info.last_price
                company = sp500_df[sp500_df["Symbol"]==ticker]["Security"].iloc[0] if ticker in sp500_df["Symbol"].values else ticker
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{ticker}** - {company}")
                with col2:
                    st.metric("Current Price", f"${price:,.2f}")
                with col3:
                    if st.button("🗑️", key=f"del_{ticker}"):
                        st.session_state.portfolio.remove(ticker)
                        update_user_portfolio(st.session_state.user_data['email'], st.session_state.portfolio)
                        st.rerun()
                st.divider()
            except:
                pass
        
        st.info("✅ You'll receive monthly email updates about these stocks!")
    else:
        st.info("👆 Add stocks above to start tracking")

# --------------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#9CA3AF;font-size:0.8rem;padding:0.5rem'>
    Built by ANGAD ARORA
</div>
""", unsafe_allow_html=True)
