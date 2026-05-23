# 📈 Advanced Stock Prediction App

An elegant, high-fidelity AI-powered dashboard designed to analyze stock trends, compute technical indicators, and forecast stock prices using optimized Machine Learning.

This application has been successfully migrated to the modern **`uv`** package manager and fully modularized into clean, reusable Python components.

---

## 🚀 Quick Start (Running the App)

Because we are using the blazing-fast Rust-based **`uv`** package manager, getting the application running locally is extremely simple.

### Prerequisites

Ensure you have **Python 3.13+** and **`uv`** installed on your system.

### Running the App

Simply run the following command in the project root:

```bash
uv run streamlit run main.py
```

This command will:
1. Automatically construct and link a virtual environment (`.venv`) if one does not exist.
2. Speedily resolve and lock all required dependencies listed in `pyproject.toml`.
3. Launch the Streamlit dashboard on a local port (typically [http://localhost:8501](http://localhost:8501)).

---

## 🛠️ Project Architecture (Modular Structure)

The project has been refactored from a single monolithic file into a highly modular package directory structure, ensuring maximum code cleanliness, separation of concerns, and ease of testing:

```text
stock-prediction/
├── pyproject.toml       # Modern uv packaging & dependency manager
├── uv.lock              # Auto-generated lockfile for reproducible environments
├── main.py              # Main Entry Point coordinating the tabs & sidebar UI
└── src/                 # Modular Code Package
    ├── __init__.py      # Package boundaries & clean API exports
    ├── helpers.py       # Helper functions (e.g. currency and exchange formatting)
    ├── indicators.py    # Math & Data processing for SMA, RSI, and MACD
    ├── predictions.py   # Machine Learning model training & forecasting (SVR)
    └── components.py    # Custom UI cards, sidebar ticker indices, & Plotly charts
```

### Module Descriptions

1. **`main.py`**
   - The orchestrator of the web dashboard. Sets page configurations, injects custom dark-theme glassmorphism CSS, builds the 3-tab layout, and coordinates calls to backend components.
2. **`src/components.py`**
   - Renders modular UI widgets including the live Sidebar Market Indices ticker overview (`Nifty 50`, `Sensex`, `S&P 500`, etc.), company info summaries, live financial news cards with dictionary-based sentiment scoring, and custom interactive Plotly charts.
3. **`src/predictions.py`**
   - Houses the core AI logic. Downloads historic data, scales input vectors (`StandardScaler`), partitions the data, and runs a dual **Randomized Search** followed by a targeted **Grid Search** (`GridSearchCV`) to optimize an RBF **Support Vector Regression (SVR)** model for forecasting.
4. **`src/indicators.py`**
   - Computes advanced mathematical financial metrics like **Simple Moving Averages (SMA)**, **Relative Strength Index (RSI)**, and **Moving Average Convergence Divergence (MACD)**.
5. **`src/helpers.py`**
   - Utility functions, such as identifying Indian NSE stocks ending with `.NS` to dynamically adapt the currency notation between INR (₹) and USD ($).

---

## 📦 Dependencies

All packages are declared inside `pyproject.toml` and locked in `uv.lock`. High-level dependencies include:

* **`streamlit`** - Modern UI layout, sliders, buttons, inputs, and dark-theme configurations.
* **`yfinance`** - Real-time market prices, company metrics, historical charts, and live business updates.
* **`scikit-learn`** - Preprocessing, data scaling, hyperparameter tuning (`GridSearchCV`), and Support Vector Regressors.
* **`plotly`** - Custom interactive chart visualization with zooming, panning, and hover-triggered details.
* **`pandas` & `numpy`** - High-speed vector operations and data frames.

---

## 🔮 Machine Learning Prediction Details

The forecasting engine in `src/predictions.py` leverages a fine-tuned **Support Vector Regressor (SVR)**:
1. **Feature Engineering**: Features include day indexes, day of the week, and month number.
2. **Standardization**: Features are normalized using `StandardScaler` to optimize SVR convergence.
3. **Randomized Search**: Rapidly tests hyperparameter combinations across `C`, `epsilon`, and `gamma` grids.
4. **Grid Search**: Fine-tunes around the best randomized parameters to construct the final estimator.
5. **Validation Metrics**: Computes and displays performance evaluation indices:
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Mean Absolute Percentage Error (MAPE)
