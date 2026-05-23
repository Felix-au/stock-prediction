import streamlit as st
import numpy as np
import yfinance as yf
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import date, timedelta
import plotly.graph_objects as go
from src.helpers import is_nse_stock

def predict(stock, days_n):
    """
    Downloads stock data, trains an SVR model using optimized hyper-parameters,
    and returns a Plotly figure showing the forecasted stock prices.
    """
    try:
        st.sidebar.info(f"Downloading stock data for {stock}")
        ticker = yf.Ticker(stock)
        df = ticker.history(period='1y')
        
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
            yaxis_title=f"Value ({'₹' if is_nse_stock(stock) else '$'})",
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
