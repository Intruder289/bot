import pandas as pd
import numpy as np
import logging
import time
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from hierarchical_extremes import HierarchicalExtremes

SYMBOL = "Volatility 75 Index"
TIMEFRAME = mt5.TIMEFRAME_M15
ATR_LOOKBACK = 168
LEVELS = 3
ENTRY_THRESHOLD = 100
STOP_LOSS = 100.0
TAKE_PROFIT_MULTIPLIER = 1.7
LOT_SIZE = 0.011
POINT_VALUE = 1.9
INITIAL_BALANCE = 100.0
MIN_BARS_SINCE_LEVEL = 4
WARMUP_BARS = ATR_LOOKBACK * 2
MAX_RETRIES = 3


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("livetrading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LiveTrader')

class LiveTrade:
    def __init__(self, trade_id, position_id, entry_time, entry_price, direction, level_price, sl, tp, lot_size):
        self.trade_id = trade_id
        self.position_id = position_id
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.level_price = level_price
        self.sl = sl
        self.tp = tp
        self.lot_size = lot_size
        self.exit_time = None
        self.exit_price = None
        self.status = 'open'
        self.profit = 0.0

    def calculate_profit(self, exit_price):
        if self.direction == 'buy':
            points = (exit_price - self.entry_price)
        else:
            points = (self.entry_price - exit_price)
        self.profit = points * self.lot_size * POINT_VALUE
        return self.profit

def initialize_mt5():
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        raise ConnectionError("Could not connect to MT5 Terminal")
    
    logger.info("MT5 initialized successfully")
    return mt5.account_info().balance

def get_historical_data(symbol, timeframe, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None:
        logger.error("Failed to get historical data")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
    return df[['time', 'open', 'high', 'low', 'close']]

def get_current_bar(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.warning("No tick data available for symbol")
        time.sleep(1)
        return None
        
    return {
        'time': datetime.now(),
        'open': tick.bid,
        'high': tick.bid,
        'low': tick.bid,
        'close': tick.bid
    }

def update_current_bar(current_bar, tick):
    current_bar['high'] = max(current_bar['high'], tick.bid)
    current_bar['low'] = min(current_bar['low'], tick.bid)
    current_bar['close'] = tick.bid
    return current_bar

def check_signal(he, historical_levels, last_bar, open_trades):
    if open_trades:
        return None
        
    current_price = last_bar['close']
    direction = None
    level_price = None
    
    # Check BUY signal (retest of historical low)
    for price, bar_idx, level_time in historical_levels['lows']:
        level_zone_top = price + ENTRY_THRESHOLD
        level_zone_bottom = price - ENTRY_THRESHOLD
        
        if level_zone_bottom <= last_bar['low'] <= level_zone_top:
            direction = 'buy'
            level_price = price
            break
    
    # Check SELL signal (retest of historical high)
    if direction is None:
        for price, bar_idx, level_time in historical_levels['highs']:
            level_zone_top = price + ENTRY_THRESHOLD
            level_zone_bottom = price - ENTRY_THRESHOLD
            
            if level_zone_bottom <= last_bar['high'] <= level_zone_top:
                direction = 'sell'
                level_price = price
                break
    
    return direction, level_price

def place_trade(symbol, direction, level_price):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
        
    entry_price = tick.ask if direction == 'buy' else tick.bid
    sl_price = entry_price - STOP_LOSS if direction == 'buy' else entry_price + STOP_LOSS
    tp_price = entry_price + (STOP_LOSS * TAKE_PROFIT_MULTIPLIER) if direction == 'buy' \
               else entry_price - (STOP_LOSS * TAKE_PROFIT_MULTIPLIER)
    
    logger.info(f"Calculated SL: {sl_price:.2f}, TP: {tp_price:.2f}, Entry: {entry_price:.2f}")
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20,
        "magic": 2025,
        "comment": f"Retest {level_price}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Trade failed: {result.comment}")
        return None
        
    return result.order, result.position_id, entry_price, sl_price, tp_price

def check_trade_closed(position_id):
    positions = mt5.positions_get(ticket=position_id)
    return positions is None or len(positions) == 0

def get_position_result(position_id):
    deals = mt5.history_deals_get(position=position_id)
    if deals is None or len(deals) < 2:
        return None, None, None
        
    close_deal = None
    for deal in deals:
        if deal.entry == 1:  
            close_deal = deal
            break
            
    if close_deal is None:
        return None, None, None
        
    exit_price = close_deal.price
    exit_time = datetime.fromtimestamp(close_deal.time)
    status = 'sl' if close_deal.reason == mt5.DEAL_REASON_SL else 'tp' if close_deal.reason == mt5.DEAL_REASON_TP else 'other'
    return exit_price, exit_time, status

def run_live_trading():

    account_balance = initialize_mt5()
    logger.info(f"Starting balance: ${account_balance:.2f}")
    

    hist_data = get_historical_data(SYMBOL, TIMEFRAME, WARMUP_BARS)
    if hist_data is None:
        mt5.shutdown()
        return
        

    he = HierarchicalExtremes(levels=LEVELS, atr_lookback=ATR_LOOKBACK)
    historical_levels = {'highs': [], 'lows': []}
    bars = [row for _, row in hist_data.iterrows()]
    

    logger.info(f"Warming up model with {WARMUP_BARS} bars...")
    for i, bar in enumerate(bars):
        he.update(i, hist_data.index, hist_data['high'].values, 
                 hist_data['low'].values, hist_data['close'].values)
        update_historical_levels(he, historical_levels, i, bar['time'])
    logger.info(f"Model warmed up. Historical levels: {len(historical_levels['highs'])} highs, {len(historical_levels['lows'])} lows")
    

    open_trades = []
    closed_trades = []
    trade_id = 1
    current_bar = None
    last_bar_time = None
    

    logger.info("Starting live trading...")
    try:
        while True:
            
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None:
                time.sleep(1)
                continue
                
            current_time = datetime.now()
            
            
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None:
                logger.warning("No tick data available for symbol")
                time.sleep(1)
                continue

            
            if current_bar is None or current_time >= last_bar_time + timedelta(minutes=15):
                logger.info(f"Initializing new bar at {current_time}")

            
            if current_bar is None or current_time >= last_bar_time + timedelta(minutes=15):
                
                if current_bar is not None:
                    bars.append(current_bar)
                    last_bar = bars[-1]
                    
                   
                    idx = len(bars) - 1
                    index_arr = np.arange(len(bars))
                    high_arr = np.array([b['high'] for b in bars])
                    low_arr = np.array([b['low'] for b in bars])
                    close_arr = np.array([b['close'] for b in bars])
                    he.update(idx, index_arr, high_arr, low_arr, close_arr)
                    

                    update_historical_levels(he, historical_levels, idx, last_bar['time'])
                    

                    if not open_trades:
                        signal = check_signal(he, historical_levels, last_bar, open_trades)

                        
                        if signal[0] is not None:
                            direction, level_price = signal
                            logger.info(f"Signal detected: {direction.upper()} at level {level_price:.2f}")
                            trade_result = place_trade(SYMBOL, direction, level_price)
                            if trade_result:
                                order_id, position_id, entry_price, sl, tp = trade_result
                                new_trade = LiveTrade(
                                    trade_id, position_id, current_time, 
                                    entry_price, direction, level_price, sl, tp, LOT_SIZE
                                )
                                open_trades.append(new_trade)
                                logger.info(f"{direction.upper()} Entry (ID:{trade_id}) at {entry_price:.2f}")
                                trade_id += 1
                
                # Start new bar
                current_bar = get_current_bar(SYMBOL)
                last_bar_time = current_time.replace(second=0, microsecond=0)
            else:
                # Update current bar
                if current_bar:
                    current_bar = update_current_bar(current_bar, tick)
            
            # Check trade exits
            for trade in open_trades[:]:
                # Debug trade closure
                if check_trade_closed(trade.position_id):
                    exit_price, exit_time, status = get_position_result(trade.position_id)
                    if exit_price:
                        profit = trade.calculate_profit(exit_price)
                        account_balance += profit
                        trade.exit_price = exit_price
                        trade.exit_time = exit_time
                        trade.status = status
                        closed_trades.append(trade)
                        open_trades.remove(trade)
                        logger.info(f"Trade {trade.trade_id} closed: {status.upper()}, Profit: ${profit:.2f}")
                    else:
                        logger.warning(f"Couldn't verify close for trade {trade.trade_id}")
            
            time.sleep(0.1)  
            
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    finally:
       
        for trade in open_trades:
            mt5.Close(SYMBOL, ticket=trade.position_id)
            logger.info(f"Closed trade {trade.trade_id} on shutdown")
        
        mt5.shutdown()
        logger.info("MT5 connection closed")
        return closed_trades, account_balance

def update_historical_levels(he, historical_levels, bar_idx, bar_time):
    # Track new highs
    current_high = he.get_level_high_price(2, 0)
    if not np.isnan(current_high):
        logger.info(f"New high detected: {current_high:.2f} at {bar_time}")
        if not historical_levels['highs'] or current_high > max(h[0] for h in historical_levels['highs']):
            historical_levels['highs'].append((current_high, bar_idx, bar_time))
    
    # Track new lows
    current_low = he.get_level_low_price(2, 0)
    if not np.isnan(current_low):
        logger.info(f"New low detected: {current_low:.2f} at {bar_time}")
        if not historical_levels['lows'] or current_low < min(l[0] for l in historical_levels['lows']):
            historical_levels['lows'].append((current_low, bar_idx, bar_time))

if __name__ == "__main__":
    logger.info("Starting live trading strategy")
    trades, balance = run_live_trading()
    logger.info(f"Final balance: ${balance:.2f}")
    logger.info(f"Total trades: {len(trades)}")