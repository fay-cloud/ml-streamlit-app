import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta

# Function to fetch BTC USD history
def get_btc_usd_history_yfinance():
    btc = yf.Ticker("BTC-USD")
    hist = btc.history(period="max")
    return hist[['Close']].rename(columns={'Close': 'price'}).reset_index()

# Load the trained model
model = joblib.load("btc_usd_rf_model.pkl")

# Get the BTC price history
btc_df = get_btc_usd_history_yfinance()
df = btc_df.copy()

# Prepare the features (lag, pct_change, rolling mean, etc.)
df['timestamp'] = df['Date'].astype('int64') // 10**9
df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
df['lag_1'] = df['price'].shift(1)
df['lag_2'] = df['price'].shift(2)
df['pct_change'] = df['price'].pct_change()
df['rolling_mean_3'] = df['price'].rolling(window=3).mean()
df['rolling_std_3'] = df['price'].rolling(window=3).std()
df.dropna(inplace=True)

# Feature columns
features = ['lag_1', 'lag_2', 'pct_change', 'rolling_mean_3', 'rolling_std_3']

# Streamlit UI
st.title("📈 BTC/USD Price Direction Prediction")

st.markdown("This app predicts whether the BTC/USD price will go **UP (1)** or **DOWN (0)** on a future date based on the trained model.")

# Input: future date
st.subheader("🗓️ Enter the future date (YYYY-MM-DD):")
user_input_date = st.text_input("Future Date", datetime.today().strftime('%Y-%m-%d'))

if user_input_date:
    try:
        input_date = datetime.strptime(user_input_date, "%Y-%m-%d")
    except ValueError:
        st.error("Please enter the date in YYYY-MM-DD format.")

    # Ensure the date is in the future
    if input_date <= datetime.today():
        st.error("Please enter a date that is **tomorrow** or later.")
    else:
        # Get the latest data point (most recent date in the dataset)
        last_row = df.iloc[-1]

        # Create new data for the input date based on the last row
        prediction_data = {
            'lag_1': last_row['price'],  # last price as lag_1
            'lag_2': df.iloc[-2]['price'],  # second last price as lag_2
            'pct_change': (last_row['price'] - df.iloc[-2]['price']) / df.iloc[-2]['price'],  # pct_change from last two days
            'rolling_mean_3': df['price'].rolling(window=3).mean().iloc[-1],  # last 3-day rolling mean
            'rolling_std_3': df['price'].rolling(window=3).std().iloc[-1]  # last 3-day rolling std
        }

        # Convert to DataFrame for prediction
        prediction_data_df = pd.DataFrame([prediction_data])

        # Make prediction for the future date
        prediction = model.predict(prediction_data_df)[0]
        prob = model.predict_proba(prediction_data_df)[0][prediction]

        # Display the result
        st.subheader("🧠 Prediction Result:")
        if prediction == 1:
            st.success(f"📈 The model predicts the price will go **UP** with {prob:.2%} confidence.")
        else:
            st.error(f"📉 The model predicts the price will go **DOWN** with {prob:.2%} confidence.")
