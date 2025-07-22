# mt5_data_fetcher.py
import MetaTrader5 as mt5
import pandas as pd
import argparse
from datetime import datetime, timedelta

def fetch_mt5_data(symbol, timeframe, start_date, end_date):
    if not mt5.initialize():
        print("MT5 initialization failed")
        return None
        
    # Convert timeframe to MT5 constant
    tf_mapping = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }
    
    mt5_timeframe = tf_mapping.get(timeframe)
    if mt5_timeframe is None:
        print(f"Unsupported timeframe: {timeframe}")
        return None
        
    # Convert dates to UTC timestamps
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())
    
    # Fetch data
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_ts, end_ts)
    mt5.shutdown()
    
    if rates is None:
        print(f"No data returned for {symbol} {timeframe}")
        return None
        
    # Create DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'time': 'datetime',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Add necessary columns for Backtrader
    df['openinterest'] = 0
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume', 'openinterest']]

def save_data_for_backtrader(symbol, htf_tf, ltf_tf, start_date, end_date):
    # Fetch HTF data
    htf_df = fetch_mt5_data(symbol, htf_tf, start_date, end_date)
    if htf_df is None:
        return
        
    # Fetch LTF data
    ltf_df = fetch_mt5_data(symbol, ltf_tf, start_date, end_date)
    if ltf_df is None:
        return
        
    # Save to CSV files
    htf_filename = f"{symbol.replace(' ', '_')}_{htf_tf}_{start_date.date()}_{end_date.date()}.csv"
    ltf_filename = f"{symbol.replace(' ', '_')}_{ltf_tf}_{start_date.date()}_{end_date.date()}.csv"
    
    htf_df.to_csv(htf_filename)
    ltf_df.to_csv(ltf_filename)
    
    print(f"Data saved to:\nHTF: {htf_filename}\nLTF: {ltf_filename}")
    
    return htf_filename, ltf_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch MT5 data for Backtrader')
    parser.add_argument('--symbol', type=str, default="Volatility 75 Index", help='Symbol to fetch')
    parser.add_argument('--htf', type=str, default='M5', help='Higher timeframe (M1, M5, M15, H1, H4, D1)')
    parser.add_argument('--ltf', type=str, default='M1', help='Lower timeframe (M1, M5, M15, H1, H4, D1)')
    parser.add_argument('--days', type=int, default=5, help='Number of days to fetch')
    
    args = parser.parse_args()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Fetch and save data
    save_data_for_backtrader(
        symbol=args.symbol,
        htf_tf=args.htf,
        ltf_tf=args.ltf,
        start_date=start_date,
        end_date=end_date
    )