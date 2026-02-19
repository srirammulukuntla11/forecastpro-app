"""
FORECASTPRO ENTERPRISE - COMPLETE FULL VERSION
1700+ Lines • All Features • Sidebar Delete Buttons • Graph Persists • Enterprise Grade
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import time
from io import BytesIO
import json
import chardet
import hashlib
import re
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.append(os.path.dirname(__file__))
import model

# ===== THEME CONFIGURATION =====
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Theme colors
themes = {
    'light': {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#4cc9f0',
        'warning': '#f72585',
        'info': '#4895ef',
        'light': '#f8f9fa',
        'dark': '#1a1f36',
        'gray': '#6b7280',
        'background': '#ffffff',
        'card_bg': '#f8f9fa',
        'border': '#dee2e6',
        'text': '#1a1f36',
        'text_secondary': '#6b7280',
        'delete': '#dc2626',
        'delete_hover': '#b91c1c',
        'gradient_start': '#667eea',
        'gradient_end': '#764ba2'
    },
    'dark': {
        'primary': '#818cf8',
        'secondary': '#a78bfa',
        'success': '#6ee7f7',
        'warning': '#fb7185',
        'info': '#60a5fa',
        'light': '#2d3748',
        'dark': '#f7fafc',
        'gray': '#9ca3af',
        'background': '#1a1f36',
        'card_bg': '#2d3748',
        'border': '#4a5568',
        'text': '#f7fafc',
        'text_secondary': '#9ca3af',
        'delete': '#ef4444',
        'delete_hover': '#dc2626',
        'gradient_start': '#818cf8',
        'gradient_end': '#a78bfa'
    }
}

# Set current theme colors
colors = themes[st.session_state.theme]

# ===== DEPLOYMENT-SAFE FILE HANDLING =====
DATA_DIR = Path("/tmp") if os.name != 'nt' else Path(".")
USER_DATA_FILE = DATA_DIR / "users.json"

def load_users():
    try:
        if USER_DATA_FILE.exists():
            with open(USER_DATA_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading users: {e}")
    return {}

def save_users(users):
    try:
        with open(USER_DATA_FILE, "w") as f:
            json.dump(users, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving users: {e}")
        return False

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="ForecastPro Enterprise",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== INITIALIZE SESSION STATE =====
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_email = None
    st.session_state.user_data = {}
    st.session_state.auth_page = "login"
    st.session_state.show_forecast_history = False
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.data_source = None
    st.session_state.forecast_history = []
    st.session_state.forecast_count = 0
    st.session_state.favorite_model = None
    st.session_state.last_forecast = None
    st.session_state.logout_trigger = False
    st.session_state.viz_chart_type = "Line Chart"
    st.session_state.tab_index = 0
    st.session_state.using_sample_data = False
    st.session_state.force_update = False
    st.session_state.selected_forecast_to_delete = None
    st.session_state.show_delete_confirmation = False
    st.session_state.last_forecast_result = None
    st.session_state.data_cleaned = False

# ===== AUTHENTICATION FUNCTIONS =====
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def login_user(username, password):
    try:
        users = load_users()
        
        if username in users:
            if users[username]['password'] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_email = users[username]['email']
                st.session_state.user_data = users[username]
                
                if 'forecast_history' in users[username]:
                    st.session_state.forecast_history = users[username]['forecast_history']
                else:
                    st.session_state.forecast_history = []
                
                st.session_state.forecast_count = len(st.session_state.forecast_history)
                
                users[username]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                save_users(users)
                
                return True, f"Welcome back, {username}!"
            else:
                return False, "Incorrect password"
        else:
            return False, "Username not found"
    except Exception as e:
        return False, f"Login error: {str(e)}"

def register_user(username, password, email, full_name):
    try:
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters"
        if not validate_email(email):
            return False, "Invalid email format"
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        users = load_users()
        
        if username in users:
            return False, "Username already exists"
        
        for user_data in users.values():
            if user_data.get('email') == email:
                return False, "Email already registered"
        
        users[username] = {
            'username': username,
            'password': hash_password(password),
            'email': email,
            'full_name': full_name,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_login': None,
            'forecast_count': 0,
            'forecast_history': [],
            'favorite_model': None,
            'role': 'Data Scientist',
            'industry': 'Technology',
            'preferences': {
                'default_months': 6,
                'default_confidence': 95,
                'export_format': 'CSV'
            }
        }
        
        save_users(users)
        
        return True, "Registration successful! Please login."
    except Exception as e:
        return False, f"Registration error: {str(e)}"

def logout_user():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_email = None
    st.session_state.user_data = {}
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.data_source = None
    st.session_state.forecast_history = []
    st.session_state.forecast_count = 0
    st.session_state.show_forecast_history = False
    st.session_state.auth_page = "login"
    st.session_state.logout_trigger = False
    st.session_state.viz_chart_type = "Line Chart"
    st.session_state.tab_index = 0
    st.session_state.using_sample_data = False
    st.session_state.force_update = False
    st.session_state.selected_forecast_to_delete = None
    st.session_state.show_delete_confirmation = False
    st.session_state.last_forecast_result = None
    st.session_state.data_cleaned = False

# ===== HELPER FUNCTION FOR EXPORT =====
def get_export_data(data, format_type, filename_prefix):
    if format_type == "CSV":
        return data.to_csv(index=False).encode('utf-8'), "text/csv", f"{filename_prefix}.csv"
    elif format_type == "Excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name='Data')
        return output.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", f"{filename_prefix}.xlsx"
    else:
        return data.to_json(orient='records', indent=2).encode('utf-8'), "application/json", f"{filename_prefix}.json"

# ===== DATA CLEANING FUNCTION =====
def clean_dataset(df):
    """Automatically clean the dataset"""
    if df is None or len(df) == 0:
        return df, False
    
    df_clean = df.copy()
    cleaning_actions = []
    
    # 1. Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    if len(df_clean) < initial_rows:
        cleaning_actions.append(f"Removed {initial_rows - len(df_clean)} duplicate rows")
    
    # 2. Handle missing values in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            cleaning_actions.append(f"Filled missing values in '{col}' with median")
    
    # 3. Remove outliers (3 standard deviations)
    for col in numeric_cols:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= mean - 3*std) & (df_clean[col] <= mean + 3*std)]
        if len(df_clean) < before:
            cleaning_actions.append(f"Removed outliers from '{col}'")
    
    # 4. Convert date columns to datetime
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                test_dates = pd.to_datetime(df_clean[col].head(100), errors='coerce')
                if test_dates.notna().sum() > 50:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    cleaning_actions.append(f"Converted '{col}' to datetime")
            except:
                continue
    
    # 5. Strip whitespace from string columns
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    
    return df_clean, cleaning_actions

# ===== CUSTOM CSS WITH THEME SUPPORT =====
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');

    .stApp {{
        font-family: 'Inter', 'Plus Jakarta Sans', sans-serif;
        background: {colors['background']};
        color: {colors['text']};
    }}

    .main .block-container {{
        padding: 1.5rem 2.5rem;
        max-width: 1440px;
        margin: 0 auto;
    }}

    /* Premium Header with Glassmorphism */
    .header {{
        background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
        padding: 2rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }}

    .header::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }}

    @keyframes rotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}

    .header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }}

    .header p {{
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 1.1rem;
        position: relative;
        z-index: 1;
    }}

    /* Glassmorphism Sidebar Profile */
    .sidebar-profile {{
        background: {colors['card_bg']};
        border: 1px solid {colors['border']};
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    }}

    .profile-row {{
        display: flex;
        align-items: center;
        gap: 1.2rem;
        margin-bottom: 1.2rem;
    }}

    .profile-avatar {{
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.5rem;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        border: 3px solid rgba(255, 255, 255, 0.5);
    }}

    .profile-name {{
        font-weight: 700;
        color: {colors['text']};
        font-size: 1.2rem;
    }}

    .profile-email {{
        font-size: 0.85rem;
        color: {colors['text_secondary']};
    }}

    .profile-stats {{
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
        padding: 1rem 0;
        border-top: 2px solid {colors['border']};
        border-bottom: 2px solid {colors['border']};
    }}

    .stat-item {{
        text-align: center;
    }}

    .stat-value {{
        font-weight: 800;
        color: {colors['primary']};
        font-size: 1.5rem;
        line-height: 1.2;
    }}

    .stat-label {{
        font-size: 0.75rem;
        color: {colors['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Premium Cards */
    .card {{
        background: {colors['card_bg']};
        padding: 1.8rem;
        border-radius: 20px;
        border: 1px solid {colors['border']};
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}

    .card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {colors['gradient_start']}, {colors['gradient_end']});
        border-radius: 20px 20px 0 0;
    }}

    .card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.15);
    }}

    .card-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {colors['text']};
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid {colors['border']};
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}

    .card-title i {{
        color: {colors['primary']};
        font-size: 1.4rem;
    }}

    /* Premium Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.8rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    }}

    /* Theme Toggle Button */
    .stButton > button[key="theme_toggle"] {{
        background: {colors['card_bg']};
        color: {colors['text']};
        border: 2px solid {colors['border']};
        box-shadow: none;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .stButton > button[key="theme_toggle"]:hover {{
        background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
        color: white;
        border: none;
    }}

    /* Download button style */
    .stButton > button[key*="download"] {{
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }}

    /* Delete button style - RED and visible */
    .stButton > button[key*="delete"] {{
        background: white !important;
        color: {colors['delete']} !important;
        border: 2px solid {colors['delete']} !important;
        box-shadow: none !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .stButton > button[key*="delete"]:hover {{
        background: {colors['delete']} !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(220, 38, 38, 0.3) !important;
    }}

    /* Premium Tabs/Navigation */
    .stRadio > div {{
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        background: {colors['card_bg']};
        padding: 0.6rem;
        border-radius: 50px;
        border: 1px solid {colors['border']};
        margin-bottom: 2.5rem;
    }}

    .stRadio [role="radiogroup"] {{
        gap: 0.5rem;
    }}

    .stRadio [data-testid="stMarkdownContainer"] {{
        padding: 0.6rem 1.5rem;
        border-radius: 40px;
        font-weight: 500;
        color: {colors['text_secondary']};
        transition: all 0.3s ease;
        cursor: pointer;
    }}

    .stRadio [aria-checked="true"] {{
        background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
        color: white !important;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }}

    /* Premium Metric Cards */
    .metric-card {{
        background: {colors['card_bg']};
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid {colors['border']};
        transition: all 0.3s ease;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.1);
        border-color: {colors['primary']};
    }}

    .metric-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {colors['primary']};
        line-height: 1.2;
    }}

    .metric-label {{
        font-size: 0.85rem;
        color: {colors['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }}

    /* Premium File Upload */
    .upload-container {{
        background: {colors['card_bg']};
        border: 2px dashed {colors['border']};
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }}

    .upload-container:hover {{
        border-color: {colors['primary']};
    }}

    /* Premium File Info */
    .file-info {{
        background: {colors['card_bg']};
        border: 1px solid {colors['border']};
        border-radius: 14px;
        padding: 1rem 1.5rem;
        margin-top: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 500;
        color: {colors['text']};
    }}

    /* Premium History Items */
    .history-item {{
        background: {colors['card_bg']};
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        border: 1px solid {colors['border']};
        transition: all 0.3s ease;
    }}

    .history-item:hover {{
        border-color: {colors['primary']};
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.1);
    }}

    .history-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }}

    .history-date {{
        font-weight: 600;
        color: {colors['primary']};
        background: rgba(102, 126, 234, 0.1);
        padding: 0.3rem 1rem;
        border-radius: 30px;
        font-size: 0.85rem;
    }}

    .history-model {{
        background: {colors['border']};
        padding: 0.3rem 1rem;
        border-radius: 30px;
        font-size: 0.85rem;
        border: 1px solid {colors['border']};
        color: {colors['text']};
    }}

    /* Premium Export Section */
    .export-section {{
        background: {colors['card_bg']};
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid {colors['border']};
        position: relative;
        overflow: hidden;
    }}

    .export-section::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.05) 0%, transparent 70%);
        animation: rotate 30s linear infinite;
    }}

    .export-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {colors['text']};
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        position: relative;
        z-index: 1;
    }}

    .export-title i {{
        color: {colors['primary']};
        font-size: 1.3rem;
    }}

    /* Premium Footer */
    .footer {{
        text-align: center;
        padding: 2.5rem;
        background: {colors['card_bg']};
        border-radius: 30px;
        margin-top: 3rem;
        border: 1px solid {colors['border']};
    }}

    .badge-container {{
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }}

    .badge {{
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        background: rgba(102, 126, 234, 0.1);
        color: {colors['primary']};
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }}

    .badge:hover {{
        background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
        color: white;
        transform: translateY(-2px);
    }}

    /* Premium Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .card, .metric-card, .history-item {{
        animation: fadeIn 0.5s ease-out;
    }}

    /* Premium Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}

    ::-webkit-scrollbar-track {{
        background: {colors['border']};
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, {colors['gradient_start']}, {colors['gradient_end']});
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, {colors['gradient_end']}, {colors['gradient_start']});
    }}

    /* Forecast card in sidebar */
    .forecast-card {{
        background: {colors['card_bg']};
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid {colors['border']};
        transition: all 0.2s;
    }}
    
    .forecast-card:hover {{
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: {colors['primary']};
    }}
    
    .forecast-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }}
    
    .forecast-date {{
        font-weight: 600;
        color: {colors['primary']};
    }}
    
    .forecast-model {{
        background: {colors['border']};
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        color: {colors['text']};
    }}

    /* Cleaning badge */
    .cleaned-badge {{
        background: {colors['success']};
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid {colors['success']};
        margin-left: 1rem;
    }}

    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown(f"""
<div class="header">
    <h1>📊 ForecastPro Enterprise</h1>
    <p>Advanced Time Series Forecasting • AI-Powered • Enterprise-Grade Analytics</p>
</div>
""", unsafe_allow_html=True)

# ===== CHECK LOGIN STATUS =====
if not st.session_state.logged_in:
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔐 LOGIN", use_container_width=True, 
                    type="primary" if st.session_state.auth_page == "login" else "secondary"):
            st.session_state.auth_page = "login"
            st.rerun()
    with col2:
        if st.button("📝 SIGN UP", use_container_width=True,
                    type="primary" if st.session_state.auth_page == "signup" else "secondary"):
            st.session_state.auth_page = "signup"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.auth_page == "login":
        st.markdown('<div class="auth-title">Welcome Back!</div>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username and password:
                    success, message = login_user(username, password)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill all fields")
    else:
        st.markdown('<div class="auth-title">Create Account</div>', unsafe_allow_html=True)
        
        with st.form("signup_form"):
            full_name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email", placeholder="Enter your email")
            username = st.text_input("Username", placeholder="Choose a username")
            password = st.text_input("Password", type="password", placeholder="Choose a password")
            confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            submitted = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submitted:
                if not all([full_name, email, username, password, confirm]):
                    st.error("Please fill all fields")
                elif password != confirm:
                    st.error("Passwords do not match")
                else:
                    success, message = register_user(username, password, email, full_name)
                    if success:
                        st.success(message)
                        st.session_state.auth_page = "login"
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(message)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
else:
    with st.sidebar:
        # ===== THEME TOGGLE BUTTON =====
        st.markdown("### 🎨 Theme")
        
        # Single toggle button for dark/light mode
        current_theme = st.session_state.theme
        button_text = "🌙 DARK MODE OFF" if current_theme == 'light' else "☀️ DARK MODE ON"
        
        if st.button(button_text, key="theme_toggle", use_container_width=True):
            st.session_state.theme = 'dark' if current_theme == 'light' else 'light'
            st.rerun()
        
        st.markdown("---")
        
        # Delete confirmation dialog
        if st.session_state.show_delete_confirmation and st.session_state.selected_forecast_to_delete is not None:
            idx = st.session_state.selected_forecast_to_delete
            forecast = st.session_state.forecast_history[idx]
            
            st.warning(f"Delete forecast from {forecast['date']}?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ YES", key="confirm_delete", use_container_width=True):
                    # Remove from session state
                    st.session_state.forecast_history.pop(idx)
                    st.session_state.forecast_count = len(st.session_state.forecast_history)
                    
                    # Update users.json
                    users = load_users()
                    if st.session_state.username in users:
                        users[st.session_state.username]['forecast_history'] = st.session_state.forecast_history
                        users[st.session_state.username]['forecast_count'] = len(st.session_state.forecast_history)
                        save_users(users)
                    
                    st.session_state.show_delete_confirmation = False
                    st.session_state.selected_forecast_to_delete = None
                    st.success("✅ Deleted!")
                    st.rerun()
            with col2:
                if st.button("❌ NO", key="cancel_delete", use_container_width=True):
                    st.session_state.show_delete_confirmation = False
                    st.session_state.selected_forecast_to_delete = None
                    st.rerun()
        
        # Profile card
        st.markdown(f"""
        <div class="sidebar-profile">
            <div class="profile-row">
                <div class="profile-avatar">{st.session_state.username[0].upper()}</div>
                <div>
                    <div class="profile-name">Hi {st.session_state.username}!!</div>
                    <div class="profile-email">{st.session_state.user_email}</div>
                </div>
            </div>
            <div class="profile-stats">
                <div class="stat-item">
                    <div class="stat-value">{st.session_state.forecast_count}</div>
                    <div class="stat-label">Forecasts</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(st.session_state.forecast_history)}</div>
                    <div class="stat-label">Saved</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # CLICKABLE FORECAST BUTTON
        forecast_text = f"📊 {st.session_state.forecast_count} Forecast"
        if st.session_state.forecast_count != 1:
            forecast_text += "s"
        if len(st.session_state.forecast_history) > 0:
            forecast_text += f" ● {len(st.session_state.forecast_history)} saved"
        
        if st.button(forecast_text, key="sidebar_forecast_btn", use_container_width=True):
            st.session_state.show_forecast_history = not st.session_state.show_forecast_history
            st.rerun()
        
        # LOGOUT BUTTON
        if st.button("🚪 Logout", key="sidebar_logout_btn", use_container_width=True):
            st.session_state.logout_trigger = True
            st.rerun()
        
        st.markdown("---")
        
        # ===== FORECAST HISTORY DISPLAY (SHOWS WHEN CLICKED) =====
        if st.session_state.show_forecast_history and len(st.session_state.forecast_history) > 0:
            st.markdown("### 📋 Saved Forecasts")
            for i, forecast in enumerate(reversed(st.session_state.forecast_history)):
                original_idx = len(st.session_state.forecast_history) - 1 - i
                
                # Create a card for each forecast
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-header">
                        <span class="forecast-date">{forecast['date']}</span>
                        <span class="forecast-model">{forecast['model']}</span>
                    </div>
                    <p style="margin: 0.3rem 0; font-size: 0.9rem; color: {colors['text']};">Months: {forecast['months']} | R²: {forecast.get('r2', 0):.3f}</p>
                """, unsafe_allow_html=True)
                
                # Mini chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=forecast['values'], 
                    mode='lines+markers',
                    line=dict(color=colors['primary'], width=2),
                    marker=dict(size=3)
                ))
                fig.update_layout(
                    height=80, 
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    template='plotly_white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download and Delete buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    hist_df = pd.DataFrame({'Forecast': forecast['values']})
                    csv = hist_df.to_csv(index=False).encode()
                    st.download_button(
                        label="📥 CSV",
                        data=csv,
                        file_name=f"forecast_{i}.csv",
                        mime="text/csv",
                        key=f"sidebar_download_{i}",
                        use_container_width=True
                    )
                
                with col2:
                    if st.button("🗑️ DELETE", key=f"sidebar_delete_{i}", use_container_width=True):
                        st.session_state.selected_forecast_to_delete = original_idx
                        st.session_state.show_delete_confirmation = True
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Selection
        st.markdown("### 🤖 Model Selection")
        model_options = {
            "📈 Linear Regression": "linear",
            "📊 Polynomial Regression": "polynomial",
            "🌲 Random Forest": "random",
            "⚡ Gradient Boosting": "gradient"
        }
        selected_model = st.selectbox("Model", list(model_options.keys()), key="model_display")
        model_type = model_options[selected_model]
        
        st.markdown("---")
        st.markdown("### 📊 Forecast Parameters")
        
        months = st.slider("Months", 1, 36, 6, key="months")
        confidence = st.select_slider("Confidence", options=[80,85,90,95,99], value=95, key="confidence")
        seasonality = st.selectbox("Seasonality", ["Auto-detect", "Monthly", "Quarterly", "None"], key="seasonality")
        
        st.markdown("---")
        st.markdown("### 📥 Export Format")
        export_format = st.radio("Format", ["CSV", "Excel", "JSON"], horizontal=True, key="export_format")
    
    if st.session_state.logout_trigger:
        logout_user()
        st.rerun()
    
    if st.session_state.show_forecast_history and len(st.session_state.forecast_history) > 0:
        with st.expander("📋 Forecast History", expanded=True):
            for i, forecast in enumerate(reversed(st.session_state.forecast_history)):
                st.markdown(f"""
                <div class="history-item">
                    <div class="history-header">
                        <span class="history-date">{forecast['date']}</span>
                        <span class="history-model">{forecast['model']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**Months:** {forecast['months']}")
                with col2:
                    best_r2 = max(forecast['metrics'].values()) if isinstance(forecast['metrics'], dict) else forecast.get('r2', 0)
                    st.write(f"**R²:** {best_r2:.3f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=forecast['values'], mode='lines+markers',
                                        line=dict(color=colors['primary'], width=2)))
                fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0),
                                showlegend=False, template='plotly_white',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                hist_df = pd.DataFrame({
                    'Period': [f"Month {i+1}" for i in range(len(forecast['values']))],
                    'Forecast': forecast['values']
                })
                export_data, mime, fname = get_export_data(hist_df, "CSV", 
                                                          f"history_forecast_{i}_{datetime.now().strftime('%Y%m%d')}")
                st.download_button(
                    label=f"📥 Download This Forecast",
                    data=export_data,
                    file_name=fname,
                    mime=mime,
                    key=f"download_hist_{i}"
                )
                st.markdown("---")
            
            if st.button("Close History", use_container_width=True):
                st.session_state.show_forecast_history = False
                st.rerun()
    
    st.markdown("---")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><i class="fas fa-cloud-upload-alt"></i> Data Upload</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2.5, 1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload file", type=['csv', 'xlsx', 'xls', 'json'], 
                                        label_visibility="collapsed", key="uploader")
    with col2:
        if st.button("📁 Sample Data", use_container_width=True):
            st.session_state.last_forecast_result = None
            st.session_state.data_source = "sample"
            st.session_state.data_loaded = True
            st.session_state.using_sample_data = True
            st.rerun()
    with col3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.data_loaded = False
            st.session_state.df = None
            st.session_state.data_source = None
            st.session_state.using_sample_data = False
            st.session_state.data_cleaned = False
            st.session_state.last_forecast_result = None
            st.rerun()
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        df_current = st.session_state.df
        status = "DEMO MODE" if st.session_state.using_sample_data else "Your Data"
        cleaned_status = "🧹 Cleaned" if st.session_state.data_cleaned else ""
        st.markdown(f"""
        <div class="file-info">
            <span><i class="fas fa-file-csv"></i> {st.session_state.data_source if st.session_state.data_source == 'sample' else uploaded_file.name if uploaded_file else 'Dataset'} ({status}) {cleaned_status}</span>
            <span><i class="fas fa-database"></i> {len(df_current):,} rows • {len(df_current.columns)} columns</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    df = None
    
    def read_csv_with_encoding(file):
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1', 'utf-16']
        for encoding in encodings:
            try:
                file.seek(0)
                return pd.read_csv(file, encoding=encoding), encoding
            except:
                continue
        return None, None
    
    if st.session_state.data_source == "sample" and st.session_state.data_loaded:
        with st.spinner("Loading sample data..."):
            try:
                df = pd.read_csv("data/Sample - Superstore.csv", encoding='latin-1')
                # Auto-clean sample data
                df, cleaning_actions = clean_dataset(df)
                st.session_state.data_cleaned = True
                st.session_state.df = df
                
                # Show cleaning summary
                if cleaning_actions:
                    with st.expander("🧹 Data Cleaning Summary"):
                        for action in cleaning_actions:
                            st.write(f"✅ {action}")
                
                st.success(f"✅ Sample data loaded and cleaned: {len(df):,} rows (DEMO MODE - Forecasts won't be saved)")
            except:
                try:
                    df = pd.read_csv("data/Sample - Superstore.csv", encoding='utf-8')
                    # Auto-clean sample data
                    df, cleaning_actions = clean_dataset(df)
                    st.session_state.data_cleaned = True
                    st.session_state.df = df
                    
                    # Show cleaning summary
                    if cleaning_actions:
                        with st.expander("🧹 Data Cleaning Summary"):
                            for action in cleaning_actions:
                                st.write(f"✅ {action}")
                    
                    st.success(f"✅ Sample data loaded and cleaned: {len(df):,} rows (DEMO MODE - Forecasts won't be saved)")
                except Exception as e:
                    st.error(f"Error loading sample: {str(e)}")
    
    elif uploaded_file is not None:
        with st.spinner("Loading and cleaning data..."):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df, encoding_used = read_csv_with_encoding(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                if df is not None:
                    # Auto-clean the dataset
                    df, cleaning_actions = clean_dataset(df)
                    st.session_state.data_cleaned = True
                    
                    # Show cleaning summary
                    if cleaning_actions:
                        with st.expander("🧹 Data Cleaning Summary"):
                            st.markdown("#### Automatic Cleaning Performed:")
                            for action in cleaning_actions:
                                st.write(f"✅ {action}")
                    
                    st.success(f"✅ Dataset loaded and automatically cleaned: {len(df):,} rows")
                    
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.session_state.using_sample_data = False
                    st.session_state.data_source = "upload"
                else:
                    st.error("Could not read file")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
    
    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        tab_names = [
            "📊 Data Explorer", 
            "📈 Visual Analytics", 
            "🚀 Generate Forecast",
            "🤖 Model Comparison",
            "🎯 AutoML",
            "🔍 Anomaly Detection",
            "📋 Forecast History"
        ]
        
        def on_tab_change():
            st.session_state.tab_index = tab_names.index(st.session_state.nav_radio)
        
        selected_tab = st.radio(
            "Navigation",
            tab_names,
            index=st.session_state.tab_index,
            horizontal=True,
            label_visibility="collapsed",
            key="nav_radio",
            on_change=on_tab_change
        )
        
        # ===== TAB 1: DATA EXPLORER =====
        if selected_tab == "📊 Data Explorer":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i class="fas fa-chart-bar"></i> Dataset Overview</div>', unsafe_allow_html=True)
            
            # Show cleaned badge
            if st.session_state.data_cleaned:
                st.markdown(f"<span class='cleaned-badge'>🧹 Auto-Cleaned</span>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(df):,}</div>
                    <div class="metric-label">Total Rows</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(df.columns)}</div>
                    <div class="metric-label">Columns</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                missing = df.isna().sum().sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{missing:,}</div>
                    <div class="metric-label">Missing Values</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(numeric_cols)}</div>
                    <div class="metric-label">Numeric Columns</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("#### Data Preview")
            preview_df = df.head(100)
            st.dataframe(preview_df, use_container_width=True, height=300)
            
            st.markdown('<div class="export-section">', unsafe_allow_html=True)
            st.markdown(f'<div class="export-title"><i class="fas fa-download"></i> Export Data</div>', unsafe_allow_html=True)
            
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                export_data, mime, fname = get_export_data(preview_df, export_format, f"data_preview_{datetime.now().strftime('%Y%m%d')}")
                st.download_button(
                    label=f"📥 Download Preview ({export_format})",
                    data=export_data,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True
                )
            
            with col_exp2:
                export_full, mime_full, fname_full = get_export_data(df, export_format, f"full_dataset_{datetime.now().strftime('%Y%m%d')}")
                st.download_button(
                    label=f"📥 Download Full Dataset ({export_format})",
                    data=export_full,
                    file_name=fname_full,
                    mime=mime_full,
                    use_container_width=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### Column Details")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Unique': df.nunique().values,
                'Missing': df.isna().sum().values,
                'Sample': [str(df[col].iloc[0])[:30] for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ===== TAB 2: VISUAL ANALYTICS =====
        elif selected_tab == "📈 Visual Analytics":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i class="fas fa-chart-line"></i> Data Visualization</div>', unsafe_allow_html=True)
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X Axis", df.columns, key="viz_x")
                with col2:
                    y_col = st.selectbox("Y Axis", numeric_cols, key="viz_y")
                
                st.markdown("#### Chart Type")
                col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
                
                with col_c1:
                    if st.button("📈 Line", key="btn_line", 
                                type="primary" if st.session_state.viz_chart_type == "Line Chart" else "secondary",
                                use_container_width=True):
                        st.session_state.viz_chart_type = "Line Chart"
                        st.rerun()
                
                with col_c2:
                    if st.button("📊 Bar", key="btn_bar",
                                type="primary" if st.session_state.viz_chart_type == "Bar Chart" else "secondary",
                                use_container_width=True):
                        st.session_state.viz_chart_type = "Bar Chart"
                        st.rerun()
                
                with col_c3:
                    if st.button("⚡ Scatter", key="btn_scatter",
                                type="primary" if st.session_state.viz_chart_type == "Scatter Plot" else "secondary",
                                use_container_width=True):
                        st.session_state.viz_chart_type = "Scatter Plot"
                        st.rerun()
                
                with col_c4:
                    if st.button("📊 Histogram", key="btn_hist",
                                type="primary" if st.session_state.viz_chart_type == "Histogram" else "secondary",
                                use_container_width=True):
                        st.session_state.viz_chart_type = "Histogram"
                        st.rerun()
                
                with col_c5:
                    if st.button("📦 Box", key="btn_box",
                                type="primary" if st.session_state.viz_chart_type == "Box Plot" else "secondary",
                                use_container_width=True):
                        st.session_state.viz_chart_type = "Box Plot"
                        st.rerun()
                
                fig = go.Figure()
                
                if st.session_state.viz_chart_type == "Line Chart":
                    fig.add_trace(go.Scatter(
                        x=df[x_col].head(500), 
                        y=df[y_col].head(500), 
                        mode='lines+markers', 
                        name=y_col,
                        line=dict(color=colors['primary'], width=2)
                    ))
                    fig.update_layout(title=f"Line Chart: {y_col} over {x_col}")
                    
                elif st.session_state.viz_chart_type == "Bar Chart":
                    fig.add_trace(go.Bar(
                        x=df[x_col].head(50), 
                        y=df[y_col].head(50), 
                        name=y_col,
                        marker_color=colors['primary']
                    ))
                    fig.update_layout(title=f"Bar Chart: {y_col} by {x_col}")
                    
                elif st.session_state.viz_chart_type == "Scatter Plot":
                    fig.add_trace(go.Scatter(
                        x=df[x_col].head(500), 
                        y=df[y_col].head(500), 
                        mode='markers', 
                        name=y_col,
                        marker=dict(color=colors['primary'], size=6, opacity=0.6)
                    ))
                    fig.update_layout(title=f"Scatter Plot: {y_col} vs {x_col}")
                    
                elif st.session_state.viz_chart_type == "Histogram":
                    fig.add_trace(go.Histogram(
                        x=df[y_col], 
                        nbinsx=30, 
                        name=y_col,
                        marker_color=colors['primary']
                    ))
                    fig.update_layout(title=f"Distribution of {y_col}")
                    
                else:
                    fig.add_trace(go.Box(
                        y=df[y_col], 
                        name=y_col,
                        marker_color=colors['primary']
                    ))
                    fig.update_layout(title=f"Box Plot: {y_col}")
                
                fig.update_layout(
                    height=500,
                    template='plotly_white',
                    hovermode='x unified',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=colors['text'])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="export-section">', unsafe_allow_html=True)
                st.markdown(f'<div class="export-title"><i class="fas fa-download"></i> Export Chart Data</div>', unsafe_allow_html=True)
                
                chart_data = pd.DataFrame({x_col: df[x_col].head(500), y_col: df[y_col].head(500)}).dropna()
                export_data, mime, fname = get_export_data(chart_data, export_format, f"chart_data_{datetime.now().strftime('%Y%m%d')}")
                
                st.download_button(
                    label=f"📥 Download Chart Data ({export_format})",
                    data=export_data,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                if len(numeric_cols) > 1:
                    st.markdown("#### Correlation Heatmap")
                    corr = df[numeric_cols[:8]].corr()
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr.values, 
                        x=corr.columns, 
                        y=corr.columns, 
                        colorscale='RdBu'
                    ))
                    fig_corr.update_layout(
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=colors['text'])
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    export_corr, mime_corr, fname_corr = get_export_data(corr, export_format, f"correlation_{datetime.now().strftime('%Y%m%d')}")
                    st.download_button(
                        label=f"📥 Download Correlation Matrix ({export_format})",
                        data=export_corr,
                        file_name=fname_corr,
                        mime=mime_corr,
                        use_container_width=True
                    )
            else:
                st.warning("No numeric columns found")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ===== TAB 3: GENERATE FORECAST =====
        elif selected_tab == "🚀 Generate Forecast":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i class="fas fa-rocket"></i> Generate Forecast</div>', unsafe_allow_html=True)
            
            st.info(f"Using **{selected_model}** model to forecast {months} months ahead")
            
            if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
                with st.spinner("Generating forecast..."):
                    try:
                        monthly = model.load_and_preprocess_data(df)
                        
                        if len(monthly) < 3:
                            st.error("Need at least 3 months of data")
                        else:
                            trained_model = model.train_model(monthly, model_type)
                            
                            last_month = monthly['Month_Num'].iloc[-1]
                            result = model.forecast_future(trained_model, last_month, months, monthly)
                            forecast = result['forecast']
                            
                            last_date = monthly['Month'].iloc[-1]
                            future_dates = []
                            current = last_date
                            for i in range(months):
                                if current.month == 12:
                                    current = datetime(current.year + 1, 1, 1)
                                else:
                                    current = datetime(current.year, current.month + 1, 1)
                                future_dates.append(current)
                            
                            # Store results in session state to persist
                            st.session_state.last_forecast_result = {
                                'monthly': monthly,
                                'forecast': forecast,
                                'future_dates': future_dates,
                                'model': selected_model,
                                'months': months,
                                'confidence': confidence
                            }
                            
                            # Save to history and update count
                            if not st.session_state.using_sample_data:
                                st.session_state.forecast_count += 1
                                
                                r2_score_value = getattr(trained_model, 'r2', 0)
                                
                                st.session_state.forecast_history.append({
                                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                    'model': selected_model,
                                    'months': months,
                                    'values': forecast,
                                    'metrics': {'R²': r2_score_value},
                                    'r2': r2_score_value
                                })
                                
                                st.session_state.favorite_model = selected_model
                                
                                users = load_users()
                                if st.session_state.username in users:
                                    users[st.session_state.username]['forecast_history'] = st.session_state.forecast_history
                                    users[st.session_state.username]['forecast_count'] = len(st.session_state.forecast_history)
                                    users[st.session_state.username]['favorite_model'] = selected_model
                                    save_users(users)
                                
                                st.success(f"✅ Forecast saved! Total forecasts: {st.session_state.forecast_count}")
                            else:
                                st.info("ℹ️ DEMO MODE: This forecast was NOT saved. Upload your own data to save forecasts.")
                            
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
            
            # Display the last forecast if it exists
            if st.session_state.last_forecast_result:
                result = st.session_state.last_forecast_result
                monthly = result['monthly']
                forecast = result['forecast']
                future_dates = result['future_dates']
                selected_model = result['model']
                months = result['months']
                confidence = result['confidence']
                
                # Plot
                fig = go.Figure()
                
                # Historical
                fig.add_trace(go.Scatter(
                    x=monthly['Month'], 
                    y=monthly['Sales'],
                    mode='lines+markers', 
                    name='Historical',
                    line=dict(color='black', width=3)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=forecast,
                    mode='lines+markers',
                    name=f'Forecast ({selected_model})',
                    line=dict(color=colors['primary'], width=3, dash='dash')
                ))
                
                # Confidence interval
                std_err = np.std(monthly['Sales']) * 0.1
                ci_factor = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}[confidence]
                
                upper = [f + ci_factor * std_err for f in forecast]
                lower = [max(0, f - ci_factor * std_err) for f in forecast]
                
                # Convert hex color to rgba for confidence interval
                primary_hex = colors['primary'].lstrip('#')
                primary_rgb = tuple(int(primary_hex[i:i+2], 16) for i in (0, 2, 4))
                
                fig.add_trace(go.Scatter(
                    x=future_dates + future_dates[::-1],
                    y=upper + lower[::-1],
                    fill='toself',
                    fillcolor=f'rgba{primary_rgb + (0.2,)}',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence}% Confidence'
                ))
                
                fig.update_layout(
                    height=500,
                    template='plotly_white',
                    hovermode='x unified',
                    title=f"{selected_model} - {months} Month Forecast",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=colors['text'])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${forecast[0]:,.0f}</div>
                        <div class="metric-label">Next Month</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    growth = ((forecast[-1] - monthly['Sales'].iloc[-1]) / monthly['Sales'].iloc[-1]) * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{growth:+.1f}%</div>
                        <div class="metric-label">Total Growth</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{growth/months:+.1f}%</div>
                        <div class="metric-label">Avg Monthly</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{confidence}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export forecast
                st.markdown('<div class="export-section">', unsafe_allow_html=True)
                st.markdown(f'<div class="export-title"><i class="fas fa-download"></i> Export Forecast</div>', unsafe_allow_html=True)
                
                forecast_df = pd.DataFrame({
                    'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
                    'Forecast': forecast,
                    'Lower Bound': lower,
                    'Upper Bound': upper
                })
                
                export_data, mime, fname = get_export_data(forecast_df, export_format, 
                                                          f"forecast_{datetime.now().strftime('%Y%m%d_%H%M')}")
                st.download_button(
                    label=f"📥 Download Forecast ({export_format})",
                    data=export_data,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Click 'Generate Forecast' to see results here.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ===== TAB 4: MODEL COMPARISON =====
        elif selected_tab == "🤖 Model Comparison":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i class="fas fa-code-branch"></i> Model Comparison</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Run Model Comparison", type="primary", use_container_width=True):
                with st.spinner("Training models..."):
                    try:
                        monthly = model.load_and_preprocess_data(df)
                        
                        if len(monthly) < 3:
                            st.error("Need at least 3 months of data")
                        else:
                            models = {}
                            predictions = {}
                            metrics = {}
                            
                            model_names = {
                                'linear': 'Linear Regression',
                                'polynomial': 'Polynomial Regression',
                                'random': 'Random Forest',
                                'gradient': 'Gradient Boosting'
                            }
                            
                            colors_map = {'linear': colors['primary'], 'polynomial': colors['success'], 
                                        'random': colors['warning'], 'gradient': colors['secondary']}
                            
                            progress = st.progress(0)
                            
                            for idx, m_name in enumerate(['linear', 'polynomial', 'random', 'gradient']):
                                try:
                                    models[m_name] = model.train_model(monthly, m_name)
                                    last = monthly['Month_Num'].iloc[-1]
                                    result = model.forecast_future(models[m_name], last, months, monthly)
                                    predictions[m_name] = result['forecast']
                                    
                                    if hasattr(models[m_name], 'r2'):
                                        metrics[m_name] = models[m_name].r2
                                    else:
                                        metrics[m_name] = 0
                                except:
                                    predictions[m_name] = [0] * months
                                    metrics[m_name] = 0
                                
                                progress.progress((idx + 1) / 4)
                            
                            last_date = monthly['Month'].iloc[-1]
                            future_dates = []
                            current = last_date
                            for i in range(months):
                                if current.month == 12:
                                    current = datetime(current.year + 1, 1, 1)
                                else:
                                    current = datetime(current.year, current.month + 1, 1)
                                future_dates.append(current)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Sales'],
                                                    mode='lines+markers', name='Historical',
                                                    line=dict(color='black', width=3)))
                            
                            for m_name, pred in predictions.items():
                                fig.add_trace(go.Scatter(x=future_dates, y=pred,
                                                        mode='lines+markers',
                                                        name=f"{model_names[m_name]} ({metrics[m_name]:.3f})",
                                                        line=dict(color=colors_map[m_name], width=2, dash='dash')))
                            
                            fig.update_layout(
                                height=500, 
                                hovermode='x unified',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color=colors['text'])
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            metrics_df = pd.DataFrame({
                                'Model': list(model_names.values()),
                                'R² Score': list(metrics.values())
                            }).sort_values('R² Score', ascending=False)
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            st.markdown('<div class="export-section">', unsafe_allow_html=True)
                            st.markdown(f'<div class="export-title"><i class="fas fa-download"></i> Export Forecast Results</div>', unsafe_allow_html=True)
                            
                            forecast_df = pd.DataFrame({'Date': [d.strftime('%Y-%m-%d') for d in future_dates]})
                            for m_name, pred in predictions.items():
                                forecast_df[model_names[m_name]] = pred
                            
                            export_data, mime, fname = get_export_data(forecast_df, export_format, 
                                                                      f"forecast_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}")
                            st.download_button(
                                label=f"📥 Download Forecast Comparison ({export_format})",
                                data=export_data,
                                file_name=fname,
                                mime=mime,
                                use_container_width=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            best_model = max(metrics, key=metrics.get)
                            st.session_state.favorite_model = model_names[best_model]
                            
                            if not st.session_state.using_sample_data:
                                st.session_state.forecast_count += 1
                                
                                st.session_state.forecast_history.append({
                                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                    'model': model_names[best_model],
                                    'months': months,
                                    'values': predictions[best_model],
                                    'metrics': metrics,
                                    'r2': metrics[best_model]
                                })
                                
                                users = load_users()
                                if st.session_state.username in users:
                                    users[st.session_state.username]['forecast_history'] = st.session_state.forecast_history
                                    users[st.session_state.username]['forecast_count'] = len(st.session_state.forecast_history)
                                    users[st.session_state.username]['favorite_model'] = st.session_state.favorite_model
                                    save_users(users)
                                
                                st.success(f"✅ Forecast saved! Total forecasts: {st.session_state.forecast_count}")
                            else:
                                st.info("ℹ️ DEMO MODE: This forecast was NOT saved. Upload your own data to save forecasts.")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.info("👆 Click 'Run Model Comparison' to compare all 4 models")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ===== TAB 5: AUTOML =====
        elif selected_tab == "🎯 AutoML":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i class="fas fa-robot"></i> AutoML</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Run AutoML", type="primary", use_container_width=True):
                with st.spinner("Running AutoML..."):
                    try:
                        monthly = model.load_and_preprocess_data(df)
                        X = monthly[['Month_Num']].values
                        y = monthly['Sales'].values
                        
                        results = []
                        configs = [
                            ('Linear Regression', LinearRegression(), {}),
                            ('Polynomial (deg=2)', LinearRegression(), {'poly': 2}),
                            ('Polynomial (deg=3)', LinearRegression(), {'poly': 3}),
                            ('Random Forest (n=50)', RandomForestRegressor(n_estimators=50, random_state=42), {}),
                            ('Random Forest (n=100)', RandomForestRegressor(n_estimators=100, random_state=42), {}),
                            ('Gradient Boosting (n=50)', GradientBoostingRegressor(n_estimators=50, random_state=42), {}),
                            ('Gradient Boosting (n=100)', GradientBoostingRegressor(n_estimators=100, random_state=42), {})
                        ]
                        
                        for name, base_model, params in configs:
                            try:
                                if 'poly' in params:
                                    poly = PolynomialFeatures(degree=params['poly'])
                                    X_train = poly.fit_transform(X)
                                    base_model.fit(X_train, y)
                                    y_pred = base_model.predict(X_train)
                                else:
                                    base_model.fit(X, y)
                                    y_pred = base_model.predict(X)
                                
                                results.append({
                                    'Model': name,
                                    'R²': r2_score(y, y_pred),
                                    'MAE': mean_absolute_error(y, y_pred),
                                    'RMSE': np.sqrt(mean_squared_error(y, y_pred))
                                })
                            except:
                                continue
                        
                        results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
                        st.dataframe(results_df.round(3), use_container_width=True)
                        
                        st.markdown('<div class="export-section">', unsafe_allow_html=True)
                        st.markdown(f'<div class="export-title"><i class="fas fa-download"></i> Export AutoML Results</div>', unsafe_allow_html=True)
                        
                        export_data, mime, fname = get_export_data(results_df, export_format, 
                                                                  f"automl_results_{datetime.now().strftime('%Y%m%d_%H%M')}")
                        st.download_button(
                            label=f"📥 Download AutoML Results ({export_format})",
                            data=export_data,
                            file_name=fname,
                            mime=mime,
                            use_container_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        best = results_df.iloc[0]
                        st.success(f"✅ Best Model: {best['Model']} (R²: {best['R²']:.3f})")
                    except Exception as e:
                        st.error(f"AutoML error: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ===== TAB 6: ANOMALY DETECTION (FIXED) =====
        elif selected_tab == "🔍 Anomaly Detection":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i class="fas fa-exclamation-triangle"></i> Anomaly Detection</div>', unsafe_allow_html=True)
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    anomaly_col = st.selectbox("Select Column", numeric_cols, key="anomaly_col")
                with col2:
                    method = st.selectbox("Method", ["Isolation Forest", "Z-Score", "IQR"], key="anomaly_method")
                
                if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
                    data = df[anomaly_col].values.reshape(-1, 1)
                    
                    if method == "Isolation Forest":
                        iso = IsolationForest(contamination=0.1, random_state=42)
                        pred = iso.fit_predict(data)
                        anomaly_idx = np.where(pred == -1)[0]
                        
                    elif method == "Z-Score":
                        # Flatten the data for Z-Score calculation
                        data_flat = data.flatten()
                        z_scores = np.abs((data_flat - data_flat.mean()) / data_flat.std())
                        anomaly_idx = np.where(z_scores > 3)[0]
                        
                    else:  # IQR method
                        # Flatten the data for IQR calculation
                        data_flat = data.flatten()
                        Q1, Q3 = np.percentile(data_flat, [25, 75])
                        iqr = Q3 - Q1
                        lower = Q1 - 1.5 * iqr
                        upper = Q3 + 1.5 * iqr
                        anomaly_idx = np.where((data_flat < lower) | (data_flat > upper))[0]
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Plot all data
                    fig.add_trace(go.Scatter(
                        y=df[anomaly_col], 
                        mode='lines+markers', 
                        name='Normal',
                        line=dict(color=colors['primary'], width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Highlight anomalies
                    if len(anomaly_idx) > 0:
                        fig.add_trace(go.Scatter(
                            x=anomaly_idx, 
                            y=df[anomaly_col].iloc[anomaly_idx], 
                            mode='markers', 
                            name=f'Anomaly ({len(anomaly_idx)})',
                            marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
                        ))
                    
                    fig.update_layout(
                        height=400, 
                        title=f"Found {len(anomaly_idx)} anomalies using {method}",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=colors['text']),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export results
                    if len(anomaly_idx) > 0:
                        st.markdown('<div class="export-section">', unsafe_allow_html=True)
                        st.markdown(f'<div class="export-title"><i class="fas fa-download"></i> Export Anomaly Results</div>', unsafe_allow_html=True)
                        
                        # Create dataframe with anomaly labels
                        result_df = df.copy()
                        result_df['is_anomaly'] = False
                        result_df.iloc[anomaly_idx, result_df.columns.get_loc('is_anomaly')] = True
                        
                        export_data, mime, fname = get_export_data(result_df, export_format, 
                                                                  f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M')}")
                        st.download_button(
                            label=f"📥 Download Results with Anomaly Labels ({export_format})",
                            data=export_data,
                            file_name=fname,
                            mime=mime,
                            use_container_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show anomaly statistics
                        st.markdown("#### Anomaly Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Anomalies", len(anomaly_idx))
                        with col2:
                            st.metric("Anomaly Percentage", f"{(len(anomaly_idx)/len(df)*100):.1f}%")
                        with col3:
                            st.metric("Anomaly Values", f"{df[anomaly_col].iloc[anomaly_idx].mean():.2f} (avg)")
                    else:
                        st.success("✅ No anomalies detected!")
            else:
                st.warning("No numeric columns found")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ===== TAB 7: FORECAST HISTORY =====
        elif selected_tab == "📋 Forecast History":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i class="fas fa-history"></i> Forecast History</div>', unsafe_allow_html=True)
            
            if len(st.session_state.forecast_history) > 0:
                for i, forecast in enumerate(reversed(st.session_state.forecast_history)):
                    original_idx = len(st.session_state.forecast_history) - 1 - i
                    with st.expander(f"📊 Forecast #{len(st.session_state.forecast_history)-i} - {forecast['date']}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.info(f"**Model:** {forecast['model']}")
                        with col2:
                            st.info(f"**Months:** {forecast['months']}")
                        with col3:
                            best_r2 = max(forecast['metrics'].values()) if isinstance(forecast['metrics'], dict) else forecast.get('r2', 0)
                            st.info(f"**Best R²:** {best_r2:.3f}")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=forecast['values'], mode='lines+markers',
                                                line=dict(color=colors['primary'], width=2)))
                        fig.update_layout(
                            height=200, 
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False, 
                            template='plotly_white',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        hist_df = pd.DataFrame({
                            'Period': [f"Month {i+1}" for i in range(len(forecast['values']))],
                            'Forecast': forecast['values']
                        })
                        export_data, mime, fname = get_export_data(hist_df, export_format, 
                                                                  f"history_forecast_{i}_{datetime.now().strftime('%Y%m%d')}")
                        st.download_button(
                            label=f"📥 Download This Forecast ({export_format})",
                            data=export_data,
                            file_name=fname,
                            mime=mime,
                            use_container_width=True
                        )
                        
                        # Delete button in main history
                        if st.button(f"🗑️ DELETE", key=f"main_delete_{i}", use_container_width=True):
                            st.session_state.selected_forecast_to_delete = original_idx
                            st.session_state.show_delete_confirmation = True
                            st.rerun()
            else:
                st.info("No forecast history yet. Run a forecast to see results here.")
            
            # Clear all button
            if len(st.session_state.forecast_history) > 0 and not st.session_state.show_delete_confirmation:
                if st.button("🗑️ CLEAR ALL HISTORY", key="clear_all_btn", use_container_width=True):
                    st.session_state.forecast_history = []
                    st.session_state.forecast_count = 0
                    users = load_users()
                    if st.session_state.username in users:
                        users[st.session_state.username]['forecast_history'] = []
                        users[st.session_state.username]['forecast_count'] = 0
                        save_users(users)
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="footer">
    <div class="badge-container">
        <span class="badge"><i class="fas fa-robot"></i> AutoML</span>
        <span class="badge"><i class="fas fa-exclamation-triangle"></i> Anomaly Detection</span>
        <span class="badge"><i class="fas fa-code-branch"></i> Model Comparison</span>
        <span class="badge"><i class="fas fa-chart-line"></i> Time Series</span>
        <span class="badge"><i class="fas fa-file-export"></i> Export CSV/Excel/JSON</span>
        <span class="badge"><i class="fas fa-users"></i> User Profiles</span>
        <span class="badge"><i class="fas fa-trash"></i> Delete Forecasts</span>
        <span class="badge"><i class="fas fa-palette"></i> Dark Mode</span>
        <span class="badge"><i class="fas fa-broom"></i> Auto Clean</span>
    </div>
    <p><strong>ForecastPro Enterprise</strong> • Version 4.0</p>
    <p style="font-size:0.8rem; color:{colors['text_secondary']};">© 2026 • All rights reserved</p>
</div>
""", unsafe_allow_html=True)