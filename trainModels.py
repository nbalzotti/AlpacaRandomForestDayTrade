import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from backtrader import Cerebro, Strategy, Broker
import backtrader as bt
import ta
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import requests

API_KEY = 'your_alpaca_api_key'
API_SECRET = 'your_alpaca_secret_key'
BASE_URL = 'https://data.alpaca.markets/v2/stocks'

symbols = ["AAPL", "GOOG", "TSLA"]
lookback_period = 365 * 2
models = {}
backup_folder = "model_backups"
models_file = 'models.pkl'

headers = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': API_SECRET}

def fetch_and_preprocess_data(symbol, lookback_period=365*2):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_period)
    
    params = {
        'timeframe': '1Day',
        'start': start_date.isoformat(),
        'end': end_date.isoformat(),
        'limit': 1000
    }
    
    url = f"{BASE_URL}/{symbol}/bars"
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    raw_data = response.json()['bars']
    
    data = pd.DataFrame(raw_data)
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
    
    #The order MUST be maintained otherwise it doesnt know which features are which
    features = ['RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal']
    
    X = data[features].values
    data['gain_pct_10'] = (data['c'].shift(-10) - data['c']) / data['c'] * 100
    y = data['gain_pct_10']

    return X, y

def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def backtest_model(symbol, model, X_full, y_full):
    # Define a custom strategy class using the provided model
    class ModelBasedStrategy(bt.Strategy):
        params = (
            ('printlog', False),
        )

        def __init__(self):
            self.data_ready = False
            self.order = None
            self.buyprice = None
            self.buycomm = None
            self.num_positive_trades = 0
            self.num_trades = 0

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                return

            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log('BUY EXECUTED, Price: {:.2f}, Cost: {:.2f}, Comm {:.2f}'.format(
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm
                    ))
                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm
                elif order.issell():
                    profit = order.executed.price - self.buyprice
                    if profit > 0:
                        self.num_positive_trades += 1
                    self.num_trades += 1
                    self.log('SELL EXECUTED, Price: {:.2f}, Profit: {:.2f}, Comm {:.2f}'.format(
                        order.executed.price,
                        profit,
                        order.executed.comm
                    ))

            self.order = None

        def log(self, txt, dt=None, doprint=False):
            if self.params.printlog or doprint:
                dt = dt or self.datas[0].datetime.date(0)
                print(f'{dt.isoformat()}, {txt}')

        def next(self):
            if not self.data_ready:
                self.data_ready = len(self) > len(X_full)
                return

            # Replace with your feature extraction logic
            features = # ...
            
            position = self.getposition(self.data).size
            gain_prediction = self.model.predict(features.reshape(1, -1))[0]
            if gain_prediction > self.buy_threshold and position == 0:
                self.buy()
            elif gain_prediction < -self.sell_threshold and position != 0:
                self.sell()

    # Set up the backtesting environment
    cerebro = bt.Cerebro()
    cerebro.addstrategy(ModelBasedStrategy)
    
    # Create a data feed for the symbol using the historical data (you can use any compatible data feed)
    data_feed = bt.feeds.PandasData(dataname=X_full)  
    cerebro.adddata(data_feed)

    # Run the backtest
    cerebro.broker.setcash(100000.0)
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()

    # Calculate the accuracy and performance metrics
    strategy = results[0]
    accuracy = strategy.num_positive_trades / strategy.num_trades if strategy.num_trades > 0 else 0
    net_profit = final_value - initial_value

    return accuracy, net_profit

def refine_model(symbol, model, X_old, y_old, X_new, y_new):
    # Refine the model with the new data
    model.fit(X_new, y_new)

    # Combine the old and new data for backtesting
    X_full = pd.concat([X_old, X_new], axis=0)
    y_full = pd.concat([y_old, y_new], axis=0)

    # Backtest the refined model
    accuracy, performance_metrics = backtest_model(symbol, model, X_full, y_full)

    # Check if the refined model meets your performance criteria
    min_accuracy = 0.6
    min_profit_factor = 1.5
    if accuracy >= min_accuracy and performance_metrics['profit_factor'] >= min_profit_factor:
        # Save the refined model
        models[symbol] = model
        save_models(models)
    else:
        # Discard the refined model and keep using the old one
        pass

    return accuracy, performance_metrics

def save_models(models, filename):
    with open(filename, "wb") as f:
        pickle.dump(models, f)

def load_models(filename):
    with open(filename, "rb") as f:
        models = pickle.load(f)
    return models

def backup_models(models, backup_folder, interval=5*60*60):
    timestamp = int(time.time())
    backup_filename = os.path.join(backup_folder, f"models_backup_{timestamp}.pkl")
    save_models(models, backup_filename)

def main():
    
    # Check if models.pkl exists and load it, otherwise create a new dictionary
    if os.path.exists(models_file):
        with open(models_file, 'rb') as f:
            models = pickle.load(f)
    else:
        models = {}
    
    for symbol in symbols:
        if symbol not in models:
            # Fetch and preprocess data, then train the model
            X, y = fetch_and_preprocess_data(symbol)
            model, accuracy = train_model(X, y)

            # Save the new model and its accuracy in the dictionary
            models[symbol] = {'model': model, 'accuracy': accuracy}
            

    # Save the models to a file using pickle
    save_models(models, "models.pkl")

    # Refining and backtesting models (loop or schedule to run periodically)
    training_time = 0
    while True:
        start_time = time.time()

        for symbol in symbols:
            # Fetch and preprocess new data for each symbol
            X_old, y_old = fetch_and_preprocess_data(symbol, lookback_period)
            X_new, y_new = fetch_and_preprocess_data(symbol, lookback_period, new_data=True)

            # Load the existing model for the symbol
            model = models[symbol]

            # Refine and backtest the model with the new data
            refine_model(symbol, model, X_old, y_old, X_new, y_new)

            # Save the refined model
            models[symbol] = model

        # Save the refined models to a file using pickle
        save_models(models, "models.pkl")

        # Update training time and create a backup if necessary
        end_time = time.time()
        training_time += end_time - start_time

        if training_time >= 5 * 60 * 60:
            # Create a backup of the current models
            backup_models(models, backup_folder)

            # Reset training time
            training_time = 0

        # Add a mechanism to stop the refining and backtesting loop when desired

if __name__ == "__main__":
    main()