# 📈 Stock Price Predictor App using LSTM

A Streamlit web app that predicts stock closing prices using a trained LSTM model. 
The app fetches 20 years of historical stock data from Yahoo Finance and shows technical indicators + LSTM predictions.

## 🚀 Features

- **Live Stock Data**: Fetches Open, High, Low, Close, Adj Close, Volume from Yahoo Finance
- **Moving Averages**: Plots MA 100, MA 200, MA 250 days
- **LSTM Prediction**: Predicts future closing prices using a pre-trained Keras LSTM model
- **Interactive**: Enter any stock ticker like `GOOG`, `AAPL`, `TSLA`, `RELIANCE.NS`
- **Performance Metric**: Shows RMSE between original and predicted values

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** - For web app interface
- **Keras + JAX** - For LSTM model inference
- **yFinance** - For stock data
- **Pandas, Numpy, Scikit-learn, Matplotlib** - For data processing and visualization

## 📦 Installation

1. **Clone/Download this project** to `Desktop/Tanvi ML Work`

2. **Install required libraries**
Open cmd and run:
```bash
pip install streamlit yfinance pandas numpy scikit-learn matplotlib keras jax jaxlib

Add the trained model file
Make sure Latest_stock_price_model.keras is in the same folder as web_stock_price_predictor.py

▶️ How to Run
Open Command Prompt
Go to project folder: cd Desktop\Tanvi ML Work
Set Keras backend to JAX: set KERAS_BACKEND=jax
Run the Streamlit app: python -m streamlit run web_stock_price_predictor.py
The app will open at http://localhost:8501

📊 How to Use
Enter a Stock Ticker in the input box.
Examples:
USE Stocks: GOOG, AAPL, MSFT, TSLA
Indian Stocks: RELIANCE.NS, TCS.NS, INFY.NS
Press Enter
The app will show: Stock Data Table for last 20 years
                   Moving Average charts
                   LSTM Prediction vs Original Price graph
                   RMSE Score

📁 Project Structure Tanvi ML Work/
│
├── web_stock_price_predictor.py    # Main Streamlit app
├── Latest_stock_price_model.keras  # Trained LSTM model
└── README.md                       # README file
