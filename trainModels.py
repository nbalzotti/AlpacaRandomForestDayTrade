import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import backtrader as bt
import ta
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import requests
from copy import copy
import json
import random
import pytz

API_KEY = 'PKGP06EF93N8MC1QUB19'
API_SECRET = 'dpWbBgaV3bABmBnldpYsAdn0DgyWacesaKEM1D1q'
BASE_URL = 'https://data.alpaca.markets/v2/stocks'

symbols = ["AAPL", "GOOG", "TSLA"]
timeframe='1Hour'
lookback_period_days = 365
saved_dataset_size = 3000
models = {}
backup_folder = "model_backups"
models_file = 'models.pkl'

headers = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': API_SECRET}

def fetch_and_preprocess_data(symbol, start_date, end_date, timeframe=timeframe, limit=10000):
    tz = pytz.UTC
    start_date = start_date.replace(tzinfo=tz)
    end_date = end_date.replace(tzinfo=tz)
    
    params = {
        'timeframe': timeframe,
        'start': start_date.isoformat(),
        'end': end_date.isoformat(),
        'limit': limit
    }

    all_data = []
    while start_date < end_date:
        params['start'] = start_date.isoformat()
        url = f"{BASE_URL}/{symbol}/bars"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        raw_data = response.json()['bars']
        
        if len(raw_data) == 0:
            break

        all_data.extend(raw_data)
        
        # Update start_date to the timestamp of the last fetched candle + 1 time unit
        last_timestamp = pd.to_datetime(raw_data[-1]['t'])
        start_date = last_timestamp + pd.to_timedelta(f'1 {timeframe.lower()}')
    
    data = pd.DataFrame(all_data)
    data['t'] = pd.to_datetime(data['t'])
    data.set_index('t', inplace=True)

    # Preprocess the data (handle missing data, normalization, etc.)
    data.dropna(inplace=True)

    # Calculate technical indicators
    data['RSI'] = ta.momentum.RSIIndicator(data['c']).rsi()
    bb = ta.volatility.BollingerBands(data['c'])
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_middle'] = bb.bollinger_mavg()
    data['BB_lower'] = bb.bollinger_lband()
    macd = ta.trend.MACD(data['c'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()

    # Create feature matrix X and target vector y
    data.dropna(inplace=True)
    
    # The order MUST be maintained otherwise it doesnt know which features are which
    features = ['RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal']
    
    X = data[features].values
    data['gain_pct_10'] = (data['c'].shift(-10) - data['c']) / data['c'] * 100
    y = data['gain_pct_10']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return data, X, y

def train_model(X, y):
    # Initialize and train the Random Forest Classifier
    model = SGDRegressor(random_state=42)
    model.fit(X, y)

    return model

def backtest_model(symbol, model, data, buy_threshold=1, sell_threshold=1):
    cash = 100000
    position = 0
    buy_price = 0
    num_positive_trades = 0
    num_trades = 0

    for index, row in data.iterrows():
        X = row[['RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal']].values
        X = X.reshape(1, -1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        gain_prediction = model.predict(X)[0]

        if gain_prediction > buy_threshold and position == 0:
            num_shares = cash // row['c']
            if num_shares > 0:
                position = num_shares
                buy_price = row['c']
                cash -= num_shares * buy_price
            print(f"{symbol}: Buying at {row['c']} (gain_prediction: {gain_prediction})")

        elif gain_prediction < -sell_threshold and position > 0:
            cash += position * row['c']
            profit = position * (row['c'] - buy_price)
            if profit > 0:
                num_positive_trades += 1
            num_trades += 1
            position = 0
            print(f"{symbol}: Selling at {row['c']} (gain_prediction: {gain_prediction})")

    # Account for remaining position
    if position > 0:
        cash += position * data['c'].iloc[-1]
        profit = position * (data['c'].iloc[-1] - buy_price)
        if profit > 0:
            num_positive_trades += 1
        num_trades += 1

    final_value = cash
    initial_value = 100000
    net_profit = final_value - initial_value
    accuracy = num_positive_trades / num_trades if num_trades > 0 else 0

    return accuracy, net_profit

def refine_model(symbol, model,  old_model, backtest_data, X, y):
    # Refine the model with the new data
    model.partial_fit(X, y)

    # Backtest the refined model
    accuracy, net_profit = backtest_model(symbol, model, backtest_data)
    accuracy_old, net_profit_old = backtest_model(symbol, old_model, backtest_data)

    # Check if the refined model meets your performance criteria
    if net_profit < net_profit_old:
        return old_model, accuracy_old, net_profit_old
    else:
        return model, accuracy, net_profit

def save_models(models, filename):
    with open(filename, "wb") as f:
        pickle.dump(models, f)

def load_models(filename):
    with open(filename, "rb") as f:
        models = pickle.load(f)
    return models

def backup_models(models, backup_folder):
    timestamp = int(time.time())
    backup_filename = os.path.join(backup_folder, f"models_backup_{timestamp}.pkl")
    save_models(models, backup_filename)
    
def get_and_save_datasets():
    data_folder = "data"
    
    for symbol in symbols:
        symbol_data_folder = os.path.join(data_folder, symbol)
        os.makedirs(symbol_data_folder, exist_ok=True)

        pickle_file = os.path.join(symbol_data_folder, f"{symbol}_data.pkl")

        if os.path.exists(pickle_file):
            print(f"Data for {symbol} already exists. Skipping...")
            continue
        
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback_period_days)

        data, X, y = fetch_and_preprocess_data(symbol, start_date, end_date)

        data_length = len(data)
        chunk_size = int(data_length / saved_dataset_size)
        
        if chunk_size < 1:
            print(f"Warning: Chunk amount for {symbol} is less than 1. Consider increasing saved_dataset_size.")
            continue

        data_chunks = []
        
        for i in range(saved_dataset_size):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size

            chunk_data = data.iloc[start_index:end_index]
            chunk_X = X[start_index:end_index]
            chunk_y = y.iloc[start_index:end_index]
            
            start_timestamp = chunk_data.index.min().isoformat()
            end_timestamp = chunk_data.index.max().isoformat()

            data_chunk = {
                'raw_data': chunk_data,
                'X': chunk_X.tolist(),
                'y': chunk_y.tolist(),
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp,
                'is_used': False
            }

            data_chunks.append(data_chunk)

        random.shuffle(data_chunks)
        
        pickle_file = os.path.join(symbol_data_folder, f"{symbol}_data.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(data_chunks, f)

def get_next_saved_dataset(symbol):
    data_folder = "data"
    symbol_data_folder = os.path.join(data_folder, symbol)
    pickle_file = os.path.join(symbol_data_folder, f"{symbol}_data.pkl")

    if not os.path.exists(pickle_file):
        print(f"No saved data found for {symbol}. Please run get_and_save_datasets() first.")
        return None, None, None

    with open(pickle_file, "rb") as f:
        data_chunks = pickle.load(f)

    for data_chunk in data_chunks:
        if not data_chunk["is_used"]:
            data_chunk["is_used"] = True
            with open(pickle_file, "wb") as f:
                pickle.dump(data_chunks, f)
            return data_chunk["raw_data"], np.array(data_chunk["X"]), np.array(data_chunk["y"])

    print(f"All saved data for {symbol} has been used. Please run get_and_save_datasets() again.")
    return None, None, None

def main():
    
    # Check if models.pkl exists and load it, otherwise create a new dictionary
    if os.path.exists(models_file):
        with open(models_file, 'rb') as f:
            models = pickle.load(f)
    else:
        models = {}
        
    #refresh data
    get_and_save_datasets()
    
    for symbol in symbols:
        if symbol not in models:
            # Grab new data
            data, X, y = get_next_saved_dataset(symbol)
            
            #if no unused data is availible, skip over this symbol.
            if(data is None):
                continue
            model = train_model(X, y)

            # Save the new model and its accuracy in the dictionary
            models[symbol] = {'model': model}
            print(f"Model created for {symbol}")

    # Save the models to a file using pickle
    save_models(models, "models.pkl")

    # Refining and backtesting models (loop or schedule to run periodically)
    training_time = 0
    while True:
        start_time = time.time()

        for symbol in symbols:
            
            # Fetch and preprocess new data
            data, X, y = get_next_saved_dataset(symbol)
            backtest_data, _, _, = get_next_saved_dataset(symbol)
            #if no unused data is availible for the first instance, skip over this symbol.
            if(data is None):
                if(backtest_data is None):
                    continue
                #if the first query for data came through but the second did not, then split the data in half, one half for fitting and the other for backtesting.
                data_len = len(data)
                half_len = data_len // 2
                backtest_data = data.iloc[half_len:]
                data = data.iloc[:half_len]
                X = X[:half_len]
                y = y[:half_len]

            # Load the existing model for the symbol
            model = models[symbol]['model']

            # Refine and backtest the model with the new data
            currentModel = copy(model)
            model, accuracy, net_profit = refine_model(symbol, model, currentModel, backtest_data, X , y)
            # Save the refined model
            models[symbol] = {'model': model, 'accuracy': accuracy}
            print(f"Model refined for {symbol} with an accuracy of {accuracy:.4f} and net profit of {net_profit:.2f}")
            
        # Save the refined models to a file using pickle
        save_models(models, "models.pkl")

        # Update training time and create a backup if necessary
        end_time = time.time()
        training_time += end_time - start_time

        if training_time >= 60 * 60:
            # Create a backup of the current models
            backup_models(models, backup_folder)

            # Reset training time
            training_time = 0

        # Add a mechanism to stop the refining and backtesting loop when desired

if __name__ == "__main__":
    main()