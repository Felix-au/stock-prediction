import streamlit as st
import yfinance as yf
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime as dt, timedelta
from src.helpers import is_nse_stock

def load_animation():
    """
    Displays a smooth progress loading bar on the Streamlit interface.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Loading... {i+1}%")
        time.sleep(0.01)
    status_text.empty()
    progress_bar.empty()

def display_market_overview():
    """
    Renders a live market overview section in the sidebar with ticker prices.
    """
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #6200ea;'>📈 QuantX</h1>
        </div>
    """, unsafe_allow_html=True)

    indices = {
        '^NSEI': 'Nifty 50',  
        '^BSESN': 'Sensex',
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^RUT': 'Russell 2000',
    }
    
    st.sidebar.subheader("Market Overview")
    
    end_date = dt.now()
    start_date = end_date - timedelta(days=7)
    
    for symbol, name in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                latest_data = data.iloc[-1]
                previous_close = data.iloc[-2]['Close'] if len(data) > 1 else latest_data['Open']
                
                current_price = latest_data['Close']
                change = ((current_price - previous_close) / previous_close) * 100
                
                color = 'green' if change >= 0 else 'red'
                arrow = '↑' if change >= 0 else '↓'
                
                currency_symbol = '₹' if symbol in ['^NSEI', '^BSESN'] else '$'
                
                st.sidebar.markdown(
                    f"{name}: {currency_symbol}{current_price:.2f} "
                    f"<span style='color:{color};'>{arrow} {abs(change):.2f}%</span>",
                    unsafe_allow_html=True
                )
            else:
                st.sidebar.write(f"{name}: Data not available")
        except Exception as e:
            st.sidebar.write(f"{name}: Error fetching data")
            st.error(f"Error fetching data for {name}: {str(e)}")

def plot_technical_indicator(df, indicator, stock_code, start_date=None, end_date=None):
    """
    Plots Relative Strength Index (RSI) or Moving Average Convergence Divergence (MACD).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")

    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()

    fig = go.Figure()
      
    if indicator == "RSI":
        st.sidebar.info(f"📈 Generating RSI for {stock_code} from {start_date.strftime('%d %B %Y')} to {end_date.strftime('%d %B %Y')}")
        st.sidebar.info("Relative Strength Index: RSI is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.")
        st.sidebar.info("Calculation: The RSI is calculated using the following formula: RSI = 100 - [100 / (1 + RS)] Where RS = Average Gain / Average Loss")
        st.sidebar.info("Scale: RSI oscillates between 0 and 100.")
        st.sidebar.info("Overbought (RSI > 70): This might be a sell signal, indicating a potential price pullback.")
        st.sidebar.info("Oversold (RSI < 30): This might be a buy signal, indicating a potential price bounce.")        
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        title = "Relative Strength Index (RSI)"
        
    elif indicator == "MACD":
        st.sidebar.info(f"📈 Generating MACD for {stock_code} from {start_date.strftime('%d %B %Y')} to {end_date.strftime('%d %B %Y')}")
        st.sidebar.info("Moving Average Convergence Divergence: MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a stock.")
        st.sidebar.info("MACD Line = 12-day EMA - 26-day EMA")
        st.sidebar.info("Signal Line = 9-day EMA of MACD Line")
        st.sidebar.info("Calculation: MACD Line - Signal Line")
        st.sidebar.info("MACD above Signal Line: Bullish signal")
        st.sidebar.info("MACD below Signal Line: Bearish signal")
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
        yaxis_title=f"Value ({'₹' if is_nse_stock(stock_code) else '$'})",
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#121212',
        font_color='#e0e0e0',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#444'),
        title_x=0.5
    )
    return fig

def display_metrics(ticker, stock_code):
    """
    Renders high-level metrics card such as Current Price, 52 Week High, and 52 Week Low.
    """
    info = ticker.info
    currency_symbol = '₹' if is_nse_stock(stock_code) else '$'
    
    metrics = {
        "Current Price": {
            "value": info.get('currentPrice', 'N/A'),
            "format": lambda x: f"{currency_symbol}{x:.2f}" if isinstance(x, (int, float)) else "N/A",
            "delta": "Current market price",
            "icon": "💵"  
        },
        "52 Week High": {
            "value": info.get('fiftyTwoWeekHigh', 'N/A'),
            "format": lambda x: f"{currency_symbol}{x:.2f}" if isinstance(x, (int, float)) else "N/A",
            "delta": "Highest price in last 52 weeks",
            "icon": "⬆️"
        },
        "52 Week Low": {
            "value": info.get('fiftyTwoWeekLow', 'N/A'),
            "format": lambda x: f"{currency_symbol}{x:.2f}" if isinstance(x, (int, float)) else "N/A",
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
            )

def mkt_cap(ticker, stock_code):
    """
    Displays the total market cap of the company.
    """
    info = ticker.info
    currency_symbol = '₹' if is_nse_stock(stock_code) else '$'
    
    metrics = {
        "Market Cap": {
        "value": info.get('marketCap', 'N/A'),
        "format": lambda x: f"{currency_symbol}{int(x):,}" if isinstance(x, (int, float)) else "N/A",
        "delta": "Total market value",
        "icon": "💰"
        },
    }
    
    cols = st.columns(len(metrics))
    for col, (label, data) in zip(cols, metrics.items()):
        with col:
            st.metric(
                f"{data['icon']} {label}",
                data['format'](data['value']),
                data['delta'],
            )

def display_news(ticker):
    """
    Loads latest news articles for the stock and runs simple dictionary-based sentiment scoring.
    """
    try:
        news = ticker.news
        if not news:
            st.info("No recent news available for this stock.")
            return

        st.markdown(
            """
            <div style='padding: 10px; background-color: #1e1e1e; border-radius: 10px;'>
                <h3 style='color: #6200ea;'>📰 Latest News & Sentiment</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        positive_words = [
            'rise', 'gain', 'profit', 'up', 'high', 'strong', 'surge', 'growth', 'increase',
            'improve', 'bullish', 'outperform', 'record high', 'optimistic', 'win', 'boost',
            'achieve', 'positive', 'successful', 'peak', 'strength', 'expand', 'advancement',
            'appreciation', 'support', 'benefit', 'recovery', 'exceed', 'upgrade', 'momentum',
            'solid', 'resilient', 'milestone', 'uptrend', 'stability', 'robust', 'strengthen',
            'innovation', 'accelerate', 'advantage', 'increasing', 'notable', 'gain', 'upside',
            'revive', 'turnaround', 'expansion', 'strategic', 'favorable', 'confidence', 'record'
        ]
        
        negative_words = [
            'fall', 'drop', 'loss', 'down', 'low', 'weak', 'decline', 'decrease', 'losses',
            'bearish', 'underperform', 'record low', 'pessimistic', 'fail', 'plummet', 'negative',
            'crisis', 'concern', 'cut', 'reduce', 'downgrade', 'pressure', 'risk', 'volatility',
            'recession', 'slump', 'default', 'uncertain', 'pullback', 'withdrawal', 'collapse',
            'headwind', 'struggle', 'disappoint', 'deteriorate', 'delay', 'regress', 'stagnant',
            'challenge', 'cutback', 'drop-off', 'fear', 'warning', 'bear market', 'turmoil',
            'unfavorable', 'doubt', 'fluctuate', 'shortfall', 'weaken', 'hurdle', 'loss-making',
            'pressure', 'reduce', 'retract', 'vulnerable', 'cutting', 'slowdown'
        ]

        for article in news[:8]:
            # Support both new nested 'content' structure and old flat structure
            if 'content' in article:
                content = article['content']
            else:
                content = article
            
            title = content.get('title', 'No title')
            title_lower = title.lower()
            sentiment_score = sum(word in title_lower for word in positive_words) - sum(word in title_lower for word in negative_words)
            
            if sentiment_score > 0:
                sentiment_icon = "🟢"
                sentiment_text = "Positive"
            elif sentiment_score < 0:
                sentiment_icon = "🔴"
                sentiment_text = "Negative"
            else:
                sentiment_icon = "⚪"
                sentiment_text = "Neutral"

            # Parse publish date
            pub_date_raw = content.get('pubDate') or content.get('providerPublishTime')
            if isinstance(pub_date_raw, str):
                try:
                    # ISO string '2026-05-23T21:05:00Z'
                    publish_date = dt.strptime(pub_date_raw.split('.')[0].replace('Z', ''), '%Y-%m-%dT%H:%M:%S')
                except Exception:
                    publish_date = dt.now()
            elif isinstance(pub_date_raw, (int, float)):
                publish_date = dt.fromtimestamp(pub_date_raw)
            else:
                publish_date = dt.now()

            # Parse publisher
            provider = content.get('provider')
            if isinstance(provider, dict):
                publisher = provider.get('displayName', 'Unknown')
            else:
                publisher = content.get('publisher', 'Unknown')

            # Parse link
            click_through = content.get('clickThroughUrl')
            if isinstance(click_through, dict):
                link = click_through.get('url')
            else:
                link = content.get('link')

            with st.expander(f"{sentiment_icon} {title}"):
                st.markdown(
                    f"""
                    <div style='padding: 10px; background-color: #2c2c2c; border-radius: 5px;'>
                        <p style='color: #e0e0e0;'><strong>Published:</strong> {publish_date.strftime('%Y-%m-%d %H:%M')}</p>
                        <p style='color: #e0e0e0;'><strong>Publisher:</strong> {publisher}</p>
                        <p style='color: #e0e0e0;'><strong>Sentiment:</strong> {sentiment_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if link:
                    st.markdown(f"[Read full article]({link})")
    except Exception as e:
        st.error(f"Unable to fetch news: {str(e)}")
