import streamlit as st
import numpy as np
import yfinance as yf
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import date, timedelta
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime as dt
from datetime import date, timedelta
import time
import numpy as np
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_animation():
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Loading... {i+1}%")
        time.sleep(0.01)
    status_text.empty()
    progress_bar.empty()

def display_market_overview():
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #6200ea;'>📈 Stock Predictor</h1>
            <p style='color: #e0e0e0;'>Market Overview</p>
        </div>
    """, unsafe_allow_html=True)
    
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ'
    }
    
    for symbol, name in indices.items():
        try:
            index = yf.download(symbol, period='1d')
            if not index.empty:
                change = ((index['Close'][-1] - index['Open'][0]) / index['Open'][0]) * 100
                color = 'green' if change >= 0 else 'red'
                arrow = '↑' if change >= 0 else '↓'
                st.sidebar.markdown(f"""
                    <div style='padding: 10px; background-color: #1e1e1e; border-radius: 5px; margin: 5px;'>
                        <p style='color: #e0e0e0;'>{name}</p>
                        <p style='color: {color};'>{arrow} {abs(change):.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        except Exception:
            continue

def predict(stock, days_n):
    try:
        st.sidebar.info(f"Downloading stock data for {stock}")
        df = yf.download(stock, period='1y')
        
        if df.empty:
            raise ValueError("No stock data available.")

        df.reset_index(inplace=True)
        df['Day'] = df.index
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month

        X = df[['Day', 'Day_of_Week', 'Month']].values
        Y = df['Close'].values.ravel()

        st.sidebar.info("Scaling Data for Better Model Performance")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.sidebar.info("Splitting Data into Training and Testing Sets")
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.1, shuffle=False)
        
        st.sidebar.info("Training Initial Model with Randomized Search for Hyperparameter Tuning")
        rsc = RandomizedSearchCV(
            estimator=SVR(kernel='rbf'),
            param_distributions={
                'C': [0.1, 1, 100, 1000],
                'epsilon': np.linspace(0.0001, 0.1, 10),
                'gamma': np.linspace(0.0001, 5, 10)
            },
            cv=5, n_iter=20, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1
        )
        rsc.fit(x_train, y_train)
        
        best_params_rsc = rsc.best_params_
        
        st.sidebar.info(f"Best Parameters from Randomized Search: "
                        f"C: {best_params_rsc['C']}, "
                        f"epsilon: {best_params_rsc['epsilon']:.4f}, "
                        f"gamma: {best_params_rsc['gamma']:.4f}")
        
        param_grid = {
            'C': [best_params_rsc['C'] * 0.5, best_params_rsc['C'], best_params_rsc['C'] * 1.5],
            'epsilon': [best_params_rsc['epsilon'] * 0.5, best_params_rsc['epsilon'], best_params_rsc['epsilon'] * 1.5],
            'gamma': [best_params_rsc['gamma'] * 0.5, best_params_rsc['gamma'], best_params_rsc['gamma'] * 1.5]
        }
        
        st.sidebar.info("Fine-tuning with Grid Search Based on Best Randomized Search Parameters")
        gsc = GridSearchCV(
            estimator=SVR(kernel='rbf'),
            param_grid=param_grid,
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1
        )
        gsc.fit(x_train, y_train)
        
        best_svr = gsc.best_estimator_
        best_params_gsc = gsc.best_params_
        
        st.sidebar.info(f"Best SVR Params from Grid Search: "
                        f"C: {best_params_gsc['C']}, "
                        f"epsilon: {best_params_gsc['epsilon']:.4f}, "
                        f"gamma: {best_params_gsc['gamma']:.4f}")

        test_predictions = best_svr.predict(x_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        mae = mean_absolute_error(y_test, test_predictions)
        mape = mean_absolute_percentage_error(y_test, test_predictions)

        st.sidebar.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.sidebar.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.sidebar.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")

        output_days = scaler.transform([[i + x_test[-1][0], (date.today() + timedelta(days=i)).weekday(), 
                                         (date.today() + timedelta(days=i)).month] for i in range(1, days_n)])
        future_dates = [date.today() + timedelta(days=i) for i in range(1, days_n)]
        predictions = best_svr.predict(output_days)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers', name='Predicted'))
        fig.update_layout(
            title=f"Predicted Close Price for Next {days_n - 1} Days",
            xaxis_title="Date",
            yaxis_title="Close Price",
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#121212',
            font_color='#e0e0e0',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#444'),
            title_x=0.5
        )
        return fig
    except Exception as e:
        st.sidebar.error(f"Error during prediction: {e}")
        raise

def calculate_technical_indicators(df):
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
        raise ValueError("Input must be a DataFrame with a 'Close' column")
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def plot_technical_indicator(df, indicator):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")

    fig = go.Figure()
      
    if indicator == "RSI":
        st.sidebar.info(f"📈 Generating RSI for {stock_code} from {start_date.strftime('%d %B %Y')} to {end_date.strftime('%d %B %Y')}")
        st.sidebar.info(f"Relative Strength Index: RSI is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.")
        st.sidebar.info(f"Calculation: The RSI is calculated using the following formula: RSI = 100 - [100 / (1 + RS)] Where RS = Average Gain / Average Loss")
        st.sidebar.info(f"Scale: RSI oscillates between 0 and 100.")
        st.sidebar.info(f"Overbought (RSI > 70): This might be a sell signal, indicating a potential price pullback.")
        st.sidebar.info(f"Oversold (RSI < 30): This might be a buy signal, indicating a potential price bounce.")        
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        title = "Relative Strength Index (RSI)"
        
    elif indicator == "MACD":
        st.sidebar.info(f"📈 Generating MACD for {stock_code} from {start_date.strftime('%d %B %Y')} to {end_date.strftime('%d %B %Y')}")
        st.sidebar.info("""Moving Average Convergence Divergence: MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a stock.""")
        st.sidebar.info(f"MACD Line = 12-day EMA - 26-day EMA")
        st.sidebar.info(f"Signal Line = 9-day EMA of MACD Line")
        st.sidebar.info(f"Calculation: MACD Line - Signal Line")
        st.sidebar.info(f"MACD above Signal Line: Bullish signal")
        st.sidebar.info(f"MACD below Signal Line: Bearish signal")
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line'))        
        fig.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal_Line'], name='MACD Histogram'))
        
        for i in range(1, len(df)):
            if (df['MACD'].iloc[i] > df['Signal_Line'].iloc[i] and 
                df['MACD'].iloc[i-1] <= df['Signal_Line'].iloc[i-1]):
                fig.add_annotation(x=df.index[i], y=df['MACD'].iloc[i],
                                text="Buy", showarrow=True, arrowhead=1)
            elif (df['MACD'].iloc[i] < df['Signal_Line'].iloc[i] and 
                df['MACD'].iloc[i-1] >= df['Signal_Line'].iloc[i-1]):
                fig.add_annotation(x=df.index[i], y=df['MACD'].iloc[i],
                                text="Sell", showarrow=True, arrowhead=1)

        title = "Moving Average Convergence Divergence (MACD)"
    else:
        raise ValueError("Invalid indicator specified")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#121212',
        font_color='#e0e0e0',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#444'),
        title_x=0.5
    )
    return fig

def display_metrics(ticker):
    info = ticker.info
    
    metrics = {
        "Market Cap": {
            "value": info.get('marketCap', 'N/A'),
            "format": lambda x: f"${int(x):,}" if isinstance(x, (int, float)) else "N/A",
            "delta": "Total market value of the company",
            "icon": "💰"
        },
        "52 Week High": {
            "value": info.get('fiftyTwoWeekHigh', 'N/A'),
            "format": lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else "N/A",
            "delta": "Highest price in last 52 weeks",
            "icon": "⬆️"
        },
        "52 Week Low": {
            "value": info.get('fiftyTwoWeekLow', 'N/A'),
            "format": lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else "N/A",
            "delta": "Lowest price in last 52 weeks",
            "icon": "⬇️"
        }
    }
    
    cols = st.columns(len(metrics))
    for col, (label, data) in zip(cols, metrics.items()):
        with col:
            st.metric(
                f"{data['icon']} {label}",
                data['format'](data['value']),
                data['delta'],
                help=f"Click for more info about {label}"
            )

def display_news(ticker):
    try:
        news = ticker.news
        if not news:
            st.info("No recent news available for this stock.")
            return

        st.markdown("""<div style='padding: 10px; background-color: #1e1e1e; border-radius: 10px;'><h3 style='color: #6200ea;'>📰 Latest News & Sentiment</h3></div>""", unsafe_allow_html=True)

        for article in news[:8]:
            positive_words = ['rise', 'gain', 'profit', 'up', 'high', 'strong']
            negative_words = ['fall', 'drop', 'loss', 'down', 'low', 'weak']
            
            title = article.get('title', '').lower()
            sentiment_score = sum(word in title for word in positive_words) - sum(word in title for word in negative_words)
            
            if sentiment_score > 0:
                sentiment_icon = "🟢"
                sentiment_text = "Positive"
            elif sentiment_score < 0:
                sentiment_icon = "🔴"
                sentiment_text = "Negative"
            else:
                sentiment_icon = "⚪"
                sentiment_text = "Neutral"

            with st.expander(f"{sentiment_icon} {article.get('title', 'No title')}"):
                publish_date = dt.fromtimestamp(article.get('providerPublishTime', 0))
                st.markdown(f"""<div style='padding: 10px; background-color: #2c2c2c; border-radius: 5px;'><p style='color: #e0e0e0;'><strong>Published:</strong> {publish_date.strftime('%Y-%m-%d %H:%M')}</p><p style='color: #e0e0e0;'><strong>Publisher:</strong> {article.get('publisher', 'Unknown')}</p><p style='color: #e0e0e0;'><strong>Sentiment:</strong> {sentiment_text}</p></div>""", unsafe_allow_html=True)
                
                if 'link' in article:
                    st.markdown(f"[Read full article]({article['link']})")
    except Exception as e:
        st.error(f"Unable to fetch news: {str(e)}")

display_market_overview()

st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #6200ea; font-size: 3em;'>Advanced Stock Prediction App</h1>
        <p style='color: #e0e0e0;'>Analyze, predict, and visualize stock market trends with AI</p>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "📈 Technical Indicators", "🔮 Prediction"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_code = st.text_input("Enter stock code", key="stock_code", placeholder="e.g., AAPL, GOOGL, MSFT")
        
        if st.button("Analyze", key="analyze_button"):
            if stock_code:
                st.sidebar.info(f"📊 Analyzing stock code {stock_code}")
                st.sidebar.info(f"📊 Fetching information of stock code {stock_code}")
                st.sidebar.info(f"📊 Fetching news related to stock code {stock_code}")
                load_animation()
                ticker = yf.Ticker(stock_code.upper())
                
                with st.expander("Company Information", expanded=True):
                    display_metrics(ticker)
                    info = ticker.info
                    st.write(info.get('longBusinessSummary', 'No description available.'))
            else:
                st.warning("Please enter a stock code.")
    
    with col2:
        if stock_code:
            display_news(yf.Ticker(stock_code))
        else:
            st.info("Enter a stock code")
with tab3:
    st.markdown("### 🔮 Stock Price Prediction")
    forecast_days = st.slider("Number of days to forecast", 1, 60, 25)
    
    if st.button("Forecast"):
        if stock_code:
            load_animation()
            try:
                fig = predict(stock_code, forecast_days + 1)
                st.plotly_chart(fig, use_container_width=True)
                st.success("Forecast generated successfully!")
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
        else:
            st.warning("Please enter a stock code first.")

with tab2:
    st.markdown("### 📈 Technical Indicators")
    
    start_date = st.date_input("Select start date", dt.now() - timedelta(days=30))
    end_date = st.date_input("Select end date", dt.now())
    
    indicator = st.selectbox("Select Technical Indicator", ["RSI", "MACD"])
    
    if st.button("Calculate"):
        if stock_code:
            load_animation()
            try:
                df = yf.download(stock_code, start=start_date, end=end_date)
                if df.empty:
                    st.warning("No data available for the selected date range.")
                else:
                    df = calculate_technical_indicators(df)
                    fig = plot_technical_indicator(df, indicator)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a stock code first.")

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