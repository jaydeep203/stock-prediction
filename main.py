import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st

def fetch_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
    data = data[['Close']]
    return data

def prepare_data(data):
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    X = data[['Close']]
    y = data['Target']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled, scaler

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict(X_train, X_test, y_train, y_test, scaler):
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    model = create_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    loss = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    y_test_actual = scaler.inverse_transform(y_test)
    y_pred_actual = scaler.inverse_transform(y_pred)
    comparison = pd.DataFrame({'Actual': y_test_actual.flatten(), 'Predicted': y_pred_actual.flatten()})
    return loss, comparison

st.title("Stock Price Prediction using LSTM")

if st.button('Run Prediction'):
    with st.spinner('Fetching data and running prediction...'):
        ticker = 'AAPL'
        data = fetch_data(ticker)
        X_scaled, y_scaled, scaler = prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        loss, comparison = train_and_predict(X_train, X_test, y_train, y_test, scaler)
        
        st.subheader('Model Evaluation')
        st.write(f'Test Loss: {loss:.4f}')
        
        st.subheader('Comparison of Actual and Predicted Prices')
        st.write(comparison.head())

        st.line_chart(comparison)
