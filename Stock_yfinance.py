#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Define available stock tickers for the dropdown
PREDEFINED_TICKERS = {
    "Coca-Cola (KO)": "KO",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN",
}

# Load stock data
def load_stock_data(ticker: str, start: str, end: str):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Add Golden Cross/Death Cross signals
def add_golden_death_cross_signals(stock_data: pd.DataFrame):
    stock_data['30-Day MA'] = stock_data['Close'].rolling(window=30).mean()
    stock_data['90-Day MA'] = stock_data['Close'].rolling(window=90).mean()
    stock_data['Signal'] = 0
    stock_data.loc[stock_data['30-Day MA'] > stock_data['90-Day MA'], 'Signal'] = 1  # Golden Cross (Buy)
    stock_data.loc[stock_data['30-Day MA'] < stock_data['90-Day MA'], 'Signal'] = -1  # Death Cross (Sell)
    stock_data = stock_data.dropna()
    return stock_data

# Train the model
def train_signal_model(stock_data: pd.DataFrame):
    X = stock_data[['Close', '30-Day MA', '90-Day MA']]
    y = stock_data['Signal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

# Plot stock signals
def plot_signals(stock_data: pd.DataFrame):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    plt.plot(stock_data.index, stock_data['30-Day MA'], label='30-Day MA', color='orange', linestyle='--')
    plt.plot(stock_data.index, stock_data['90-Day MA'], label='90-Day MA', color='green', linestyle='--')
    buy_signals = stock_data[stock_data['Signal'] == 1]
    sell_signals = stock_data[stock_data['Signal'] == -1]
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal (Golden Cross)', alpha=1)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal (Death Cross)', alpha=1)
    plt.title('Stock Price with Golden Cross/Death Cross Signals (30-day & 90-day MAs)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    st.pyplot(plt)  # Display the plot in Streamlit

# Main function for the Streamlit app
def main():
    st.title("Stock Analysis with Golden Cross/Death Cross")
    st.write("A machine learning application for analyzing stock trends using Golden Cross and Death Cross strategies.")

    # Dropdown for selecting a stock ticker
    ticker_name = st.selectbox("Select a Stock:", list(PREDEFINED_TICKERS.keys()))
    ticker = PREDEFINED_TICKERS[ticker_name]

    # Input fields for date range
    start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

    if start_date >= end_date:
        st.error("End date must be after start date.")
    else:
        # Load and process stock data
        stock_data = load_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        stock_data_with_signals = add_golden_death_cross_signals(stock_data)

        # Train the model and get predictions
        model, X_test, y_test, y_pred = train_signal_model(stock_data_with_signals)

        # Display classification report and confusion matrix
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        st.text(confusion_matrix(y_test, y_pred))

        # Plot the stock data with signals
        plot_signals(stock_data_with_signals)


# In[4]:





# In[5]:


# Run the app
if __name__ == "__main__":
    main()


# In[ ]:




