import streamlit as st
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import date, timedelta
import plotly.graph_objects as go
from src.helpers import is_nse_stock

def get_prepared_data(stock):
    """
    Downloads historical stock data, structures features, normalizes them,
    and returns train/test splits, scaler, and original DataFrame.
    """
    st.sidebar.info(f"Downloading stock data for {stock}")
    ticker = yf.Ticker(stock)
    df = ticker.history(period='1y')
    
    if df.empty:
        raise ValueError(f"No stock data available for {stock}.")

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
    
    return x_train, x_test, y_train, y_test, scaler, df

def predict(stock, days_n, algorithm="Linear Regression"):
    """
    Trains the chosen forecasting algorithm on historical data and predicts close price
    for the next days_n days. Returns a Plotly figure and performance metrics.
    """
    try:
        if algorithm == "Run All and Compare":
            x_train, x_test, y_train, y_test, scaler, df = get_prepared_data(stock)
            
            algorithms = [
                "Linear Regression",
                "Random Forest",
                "Gradient Boosting"
            ]
            
            metrics_dict = {}
            forecasts = {}
            future_dates = [date.today() + timedelta(days=i) for i in range(1, days_n)]
            
            fig = go.Figure()
            
            for alg in algorithms:
                if alg == "Linear Regression":
                    st.sidebar.info("Training Linear Regression...")
                    model = LinearRegression()
                    model.fit(x_train, y_train)
                elif alg == "Random Forest":
                    st.sidebar.info("Training Random Forest...")
                    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                    model.fit(x_train, y_train)
                elif alg == "Gradient Boosting":
                    st.sidebar.info("Training Gradient Boosting...")
                    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
                    model.fit(x_train, y_train)
                
                # Metrics
                test_predictions = model.predict(x_test)
                rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
                mae = mean_absolute_error(y_test, test_predictions)
                mape = mean_absolute_percentage_error(y_test, test_predictions)
                
                metrics_dict[alg] = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
                
                # Predictions
                output_days = scaler.transform([[i + x_test[-1][0], (date.today() + timedelta(days=i)).weekday(), 
                                                 (date.today() + timedelta(days=i)).month] for i in range(1, days_n)])
                predictions = model.predict(output_days)
                forecasts[alg] = predictions
                
                # Plot line
                fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers', name=alg))
                
            fig.update_layout(
                title=f"Forecast close price Comparison for {stock.upper()}",
                xaxis_title="Date",
                yaxis_title=f"Value ({'₹' if is_nse_stock(stock) else '$'})",
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#121212',
                font_color='#e0e0e0',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#444'),
                title_x=0.5
            )
            
            return fig, metrics_dict, (future_dates, forecasts)

        # Single algorithm path
        x_train, x_test, y_train, y_test, scaler, df = get_prepared_data(stock)
        
        # Select and train model
        if algorithm == "Linear Regression":
            st.sidebar.info("Training Baseline Linear Regression Model...")
            model = LinearRegression()
            model.fit(x_train, y_train)
            st.sidebar.info("Linear Regression model trained.")

        elif algorithm == "Random Forest":
            st.sidebar.info("Training Random Forest Model...")
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            model.fit(x_train, y_train)
            st.sidebar.info("Random Forest model trained.")

        elif algorithm == "Gradient Boosting":
            st.sidebar.info("Training Gradient Boosting Model...")
            model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            model.fit(x_train, y_train)
            st.sidebar.info("Gradient Boosting model trained.")

        else:
            raise ValueError(f"Unknown forecasting algorithm: {algorithm}")

        # Make predictions on test set for validation
        test_predictions = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        mae = mean_absolute_error(y_test, test_predictions)
        mape = mean_absolute_percentage_error(y_test, test_predictions)

        st.sidebar.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.sidebar.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.sidebar.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")

        # Forecast future values
        output_days = scaler.transform([[i + x_test[-1][0], (date.today() + timedelta(days=i)).weekday(), 
                                         (date.today() + timedelta(days=i)).month] for i in range(1, days_n)])
        future_dates = [date.today() + timedelta(days=i) for i in range(1, days_n)]
        predictions = model.predict(output_days)

        # Plot predicted results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers', name=f'{algorithm} Predicted'))
        fig.update_layout(
            title=f"Forecast close price for Next {days_n - 1} Days ({algorithm})",
            xaxis_title="Date",
            yaxis_title=f"Value ({'₹' if is_nse_stock(stock) else '$'})",
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#121212',
            font_color='#e0e0e0',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#444'),
            title_x=0.5
        )
        
        metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
        return fig, metrics, (future_dates, predictions)
    except Exception as e:
        st.sidebar.error(f"Error during prediction: {e}")
        raise
