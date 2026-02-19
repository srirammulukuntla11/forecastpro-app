"""
FORECASTING ENGINE - COMPLETE VERSION
All ML Models • Data Processing • Forecast Functions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(df):
    """
    Advanced data preprocessing with multiple strategies
    """
    df = df.copy()
    df.columns = [str(col).strip().replace('\n', ' ').replace('\r', '') for col in df.columns]
    
    # ===== ADVANCED DATE DETECTION =====
    date_col = None
    date_patterns = ['date', 'time', 'day', 'month', 'year', 'order', 'ship', 'created', 'timestamp']
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(pattern in col_lower for pattern in date_patterns):
            try:
                test_dates = pd.to_datetime(df[col].head(100), errors='coerce')
                if test_dates.notna().sum() > 50:
                    date_col = col
                    break
            except:
                continue
    
    if date_col is None:
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = str(df[col].iloc[0])
                if ('/' in sample or '-' in sample) and any(c.isdigit() for c in sample):
                    try:
                        test_dates = pd.to_datetime(df[col].head(100), errors='coerce')
                        if test_dates.notna().sum() > 50:
                            date_col = col
                            break
                    except:
                        continue
    
    if date_col is None:
        df['_synthetic_date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        date_col = '_synthetic_date'
    
    # ===== ADVANCED VALUE DETECTION =====
    value_col = None
    value_patterns = ['sales', 'revenue', 'profit', 'amount', 'total', 'price', 'value', 'cost', 'income']
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(pattern in col_lower for pattern in value_patterns):
            if pd.api.types.is_numeric_dtype(df[col]):
                value_col = col
                break
    
    if value_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            means = [(col, df[col].mean()) for col in numeric_cols if df[col].mean() > 0]
            if means:
                value_col = max(means, key=lambda x: x[1])[0]
            else:
                value_col = numeric_cols[0]
    
    if value_col is None:
        df['_synthetic_value'] = range(1, len(df) + 1)
        value_col = '_synthetic_value'
    
    # ===== DATA CLEANING =====
    try:
        df['_date_processed'] = pd.to_datetime(df[date_col], errors='coerce')
    except:
        df['_date_processed'] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors='coerce')
    
    df = df.dropna(subset=['_date_processed'])
    
    df['_date_processed'] = df['_date_processed'].apply(
        lambda x: x if x.year < 2026 else datetime(2025, 12, 31)
    )
    
    df['_value_processed'] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=['_value_processed'])
    
    # Remove outliers (3 sigma)
    mean = df['_value_processed'].mean()
    std = df['_value_processed'].std()
    df = df[(df['_value_processed'] >= mean - 3*std) & (df['_value_processed'] <= mean + 3*std)]
    
    # Sort by date
    df = df.sort_values('_date_processed')
    
    # Aggregate by month
    df['YearMonth'] = df['_date_processed'].dt.to_period('M')
    monthly = df.groupby('YearMonth')['_value_processed'].sum().reset_index()
    monthly = monthly.sort_values('YearMonth')
    
    # Create result
    result = pd.DataFrame()
    result['Month'] = monthly['YearMonth'].astype(str)
    result['Month'] = pd.to_datetime(result['Month'])
    result['Month_Num'] = range(1, len(result) + 1)
    result['Sales'] = monthly['_value_processed'].values
    
    # Add features for ML
    result['Sales_Lag1'] = result['Sales'].shift(1)
    result['Sales_Lag2'] = result['Sales'].shift(2)
    result['Sales_Lag3'] = result['Sales'].shift(3)
    result['Sales_Rolling_Mean_3'] = result['Sales'].rolling(window=3).mean()
    result['Sales_Rolling_Std_3'] = result['Sales'].rolling(window=3).std()
    
    # Fill NaN values
    result = result.fillna(method='bfill').fillna(method='ffill')
    
    return result

def train_model(monthly_sales, model_type='polynomial'):
    """
    Train multiple model types with advanced features
    """
    feature_cols = ['Month_Num', 'Sales_Lag1', 'Sales_Lag2', 'Sales_Rolling_Mean_3']
    X = monthly_sales[feature_cols].values
    y = monthly_sales['Sales'].values
    
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        
    elif model_type == 'polynomial':
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        model.poly = poly
        
    elif model_type == 'random':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X, y)
        
    elif model_type == 'gradient':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
    
    # Calculate metrics
    if hasattr(model, 'predict'):
        if model_type == 'polynomial' and hasattr(model, 'poly'):
            X_pred = model.poly.transform(X)
            y_pred = model.predict(X_pred)
        else:
            y_pred = model.predict(X)
        
        model.r2 = r2_score(y, y_pred)
        model.mae = mean_absolute_error(y, y_pred)
        model.rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return model

def forecast_future(model, last_month_num, months_ahead, monthly_sales):
    """
    Generate future predictions with advanced features
    """
    future = []
    current = last_month_num
    
    # Get last values for lag features
    if len(monthly_sales) >= 3:
        last_sales = monthly_sales['Sales'].values[-3:]
        last_rolling_mean = monthly_sales['Sales_Rolling_Mean_3'].values[-1]
    else:
        last_sales = monthly_sales['Sales'].values
        last_rolling_mean = monthly_sales['Sales'].mean()
    
    for i in range(months_ahead):
        current += 1
        
        # Create feature vector with lags
        if len(last_sales) >= 3:
            lag1 = last_sales[-1]
            lag2 = last_sales[-2]
            lag3 = last_sales[-3]
            rolling_mean = np.mean(last_sales[-3:])
        else:
            lag1 = last_sales[-1] if len(last_sales) >= 1 else 0
            lag2 = last_sales[-2] if len(last_sales) >= 2 else lag1
            lag3 = last_sales[-3] if len(last_sales) >= 3 else lag2
            rolling_mean = np.mean(last_sales)
        
        features = np.array([[current, lag1, lag2, rolling_mean]])
        
        # Make prediction based on model type
        try:
            if hasattr(model, 'poly'):
                features_poly = model.poly.transform(features)
                pred = model.predict(features_poly)[0]
            else:
                pred = model.predict(features)[0]
        except:
            # Fallback to simple prediction if features fail
            simple_features = np.array([[current]])
            if hasattr(model, 'poly'):
                simple_poly = model.poly.transform(simple_features)
                pred = model.predict(simple_poly)[0]
            else:
                pred = model.predict(simple_features)[0]
        
        pred = max(0, pred)  # No negative predictions
        future.append(pred)
        
        # Update last_sales for next iteration
        last_sales = np.append(last_sales[1:], pred) if len(last_sales) >= 3 else np.append(last_sales, pred)[-3:]
    
    return {'forecast': future}