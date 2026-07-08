import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("📈 Stock Price Predictor App using LSTM")

# 1. User input
stock = st.text_input("Enter the Stock ID", "GOOG")

# 2. Get last 20 years data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# 3. Download data - FIX: Keep all columns
google_data = yf.download(stock, start, end)

if google_data.empty:
    st.error("Please enter a valid Stock Ticker. Ex: GOOG, AAPL, TSLA, RELIANCE.NS")
    st.stop()

# FIX: Flatten columns if yfinance returns multiindex
if isinstance(google_data.columns, pd.MultiIndex):
    google_data.columns = [col[0] for col in google_data.columns]

google_data.dropna(inplace=True)

# 4. Load trained model
try:
    model = load_model("Latest_stock_price_model.keras")
except:
    st.error("Model file 'Latest_stock_price_model.keras' not found. Put it in same folder.")
    st.stop()

st.subheader("Stock Data")
st.write(google_data) # This will now show Open High Low Close Adj Close Volume

# 5. Split data - We still need only Close for prediction
splitting_len = int(len(google_data)*0.7)
close_data = google_data['Close']
x_test = pd.DataFrame(close_data[splitting_len:])

# 6. plot_graph function
def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset, 'g')
    plt.title("Close Price vs Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(['MA', 'Original Close', 'MA2'])
    return fig

# 7. MA 250 days
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data, 0))

# 8. MA 200 days
st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data, 0))

# 9. MA 100 days
st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 0))

# 10. MA 100 and 250 together
st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# 11. Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# 12. Create sequences
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

# 13. Predictions
predictions = model.predict(x_data, verbose=0)

# 14. Inverse transform
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# 15. Create dataframe for plotting
plotting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index = google_data.index[splitting_len+100:]
)

st.subheader("Original values vs Predicted values")
st.write(plotting_data.tail())

# 16. Final combined plot
st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(google_data['Close'][:splitting_len+100], label="Data- not used")
plt.plot(plotting_data['original_test_data'], label="Original Test data")
plt.plot(plotting_data['predictions'], label="Predicted Test data")
plt.legend()
plt.title("Close Price vs Time")
plt.xlabel("Date")
plt.ylabel("Price")
st.pyplot(fig)

# 17. RMSE
rmse = np.sqrt(np.mean((inv_pre - inv_y_test)**2))
st.metric("RMSE", f"{rmse:.2f}")