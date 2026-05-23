import streamlit as st
import yfinance as yf
from datetime import datetime as dt, timedelta
import plotly.graph_objects as go

# Import modular components from our src package
from src import (
    load_animation,
    display_market_overview,
    plot_technical_indicator,
    display_metrics,
    mkt_cap,
    display_news,
    calculate_technical_indicators,
    calculate_sma,
    predict,
    is_nse_stock
)

st.set_page_config(
    page_title="QuantX: Stock Analysis and Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #6200ea; font-size: 3em;'>QuantX: Stock Analysis and Forecast</h1>
        <p style='color: #e0e0e0;'>Analyze, predict, and visualize stock market trends with AI</p>
    </div>
""", unsafe_allow_html=True)

# Custom CSS styling (matching modern, dark dashboard aesthetic)
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        border-radius: 4px;
        color: #e0e0e0;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2c2c2c;
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #6200ea;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        width: 150px;
    }
    
    .stButton>button:hover {
        background-color: #7c4dff;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(124, 77, 255, 0.4);
    }
    
    /* Input styling */
    .stTextInput>div>div>input,
    .stDateInput>div>div>input,
    .stNumberInput>div>div>input {
        background-color: #2c2c2c;
        color: #f0f0f0;
        border-radius: 10px;
        border: 1px solid #444;
        padding: 10px;
        transition: all 0.3s ease;
    }
    
    /* Hover effects */
    .element-container:hover {
        transform: translateY(-2px);
        transition: transform 0.3s ease;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    
    .stSpinner {
        animation: pulse 1.5s infinite;
    }
    
    /* Card effects */
    .css-1r6slb0 {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .css-1r6slb0:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Render Sidebar Market Overview
display_market_overview()

# App tabs layout
tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "📈 Technical Indicators", "🔮 Prediction"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_code = st.text_input("Enter stock code", key="stock_code", placeholder="eg: AAPL, TCS.NS || add .NS for NSE stocks")
        
        if st.button("Analyze", key="analyze_button"):
            if stock_code:
                st.sidebar.info(f"📊 Analyzing stock code {stock_code}")
                st.sidebar.info(f"📊 Fetching information of stock code {stock_code}")
                st.sidebar.info(f"📊 Fetching news related to stock code {stock_code}")
                load_animation()
                ticker = yf.Ticker(stock_code.upper())
                
                with st.expander("Company Information", expanded=True):
                    mkt_cap(ticker, stock_code)
                    display_metrics(ticker, stock_code)
                    info = ticker.info
                    st.write(info.get('longBusinessSummary', 'No description available.'))                    
                    
                    # Fetch 5 years of historical data
                    historical_data = ticker.history(period='5y')
                    short_sma, long_sma = calculate_sma(historical_data)                    
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=historical_data.index, y=short_sma, mode='lines', name='20-Day SMA', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=historical_data.index, y=long_sma, mode='lines', name='100-Day SMA', line=dict(color='red')))
                    
                    fig.update_layout(
                        title=f"{stock_code.upper()} Price and SMAs", 
                        xaxis_title="Date", 
                        yaxis_title=f"Price ({'₹' if is_nse_stock(stock_code) else '$'})", 
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Please enter a stock code.")
    
    with col2:
        if stock_code:
            display_news(yf.Ticker(stock_code))
        else:
            st.info("Enter a stock code")

with tab2:
    st.markdown("### 📈 Technical Indicators")
    
    start_date = st.date_input("Select start date", dt.now() - timedelta(days=150))
    end_date = st.date_input("Select end date", dt.now())
    
    indicator = st.selectbox("Select Technical Indicator", ["RSI", "MACD"])
    
    if st.button("Calculate"):
        if stock_code:
            load_animation()
            try:
                ticker = yf.Ticker(stock_code.upper())
                df = ticker.history(start=start_date, end=end_date)
                if df.empty:
                    st.warning("No data available for the selected date range.")
                else:
                    df = calculate_technical_indicators(df)
                    fig = plot_technical_indicator(df, indicator, stock_code, start_date=start_date, end_date=end_date)
                    st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a stock code first.")

with tab3:
    st.markdown("### 🔮 Stock Price Prediction")
    
    algorithm = st.selectbox("Select Forecasting Algorithm", [
        "Support Vector Regression (SVR)",
        "Linear Regression",
        "Random Forest"
    ], key="prediction_algorithm")
    
    forecast_days = st.slider("Number of days to forecast", 1, 60, 25)
    
    if st.button("Forecast"):
        if stock_code:
            load_animation()
            try:
                fig, metrics, forecast_data = predict(stock_code, forecast_days + 1, algorithm=algorithm)
                st.plotly_chart(fig, width="stretch")
                
                # Show evaluation metrics
                st.subheader("📊 Model Performance Metrics")
                cols = st.columns(3)
                cols[0].metric("📉 Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.4f}")
                cols[1].metric("🎯 Mean Absolute Error (MAE)", f"{metrics['MAE']:.4f}")
                cols[2].metric("📊 Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']:.2%}")
                
                st.success("Forecast generated successfully!")
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
        else:
            st.warning("Please enter a stock code first.")
