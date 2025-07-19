import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from hierarchical_extremes import HierarchicalExtremes

# Backtest Configuration
CSV_PATH = "Volatility_75_Index_M15_2025-06-19_2025-07-19.csv"
ATR_LOOKBACK = 168
LEVELS = 4
ENTRY_THRESHOLD = 100.0
STOP_LOSS = 100.0
TAKE_PROFIT_MULTIPLIER = 1.5
LOT_SIZE = 0.011
POINT_VALUE = 1.9
INITIAL_BALANCE = 100.0
MIN_BARS_SINCE_LEVEL = 4  # Minimum bars since level was formed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtester')

class Trade:
    def __init__(self, trade_id, entry_time, entry_price, direction, level_price, sl, tp, lot_size):
        self.trade_id = trade_id
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction  # 'buy' or 'sell'
        self.level_price = level_price  # The level price being retested
        self.sl = sl
        self.tp = tp
        self.lot_size = lot_size
        self.exit_time = None
        self.exit_price = None
        self.status = 'open'
        self.profit = 0.0

    def check_exit(self, current_bar):
        high = current_bar['high']
        low = current_bar['low']
        
        if self.direction == 'buy':
            if high >= self.tp:
                self.exit_price = self.tp
                self.status = 'tp'
                return True
            if low <= self.sl:
                self.exit_price = self.sl
                self.status = 'sl'
                return True
        elif self.direction == 'sell':
            if low <= self.tp:
                self.exit_price = self.tp
                self.status = 'tp'
                return True
            if high >= self.sl:
                self.exit_price = self.sl
                self.status = 'sl'
                return True
        return False

    def calculate_profit(self):
        if self.direction == 'buy':
            points = (self.exit_price - self.entry_price)
        else:
            points = (self.entry_price - self.exit_price)
        self.profit = points * self.lot_size * POINT_VALUE
        return self.profit

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    column_map = {'datetime': 'time', 'openinterest': 'open_interest'}
    df.rename(columns=column_map, inplace=True)
    
    if 'time' not in df.columns:
        raise ValueError("CSV missing 'time' column")
    

    column_map = {
        'datetime' : 'time',
        'Open' : 'open',
        'High': 'high',
        'Low' : 'low',
        'Close': 'close'
    }

    df.rename(columns=column_map, inplace = True)

    df['time'] = pd.to_datetime(df['time'])
    required_cols = ['time', 'open', 'high', 'low', 'close']
    
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing columns: {missing}")
    
    return df.sort_values('time').reset_index(drop=True)

def run_backtest(df):
    he = HierarchicalExtremes(levels=LEVELS, atr_lookback=ATR_LOOKBACK)
    open_trades = []
    closed_trades = []
    account_balance = INITIAL_BALANCE
    trade_id = 1
    warmup_bars = ATR_LOOKBACK * 2
    logger.info(f"Warming up model with {warmup_bars} bars...")
    
    # Store historical levels with their formation time
    historical_levels = {'highs': [], 'lows': []}
    
    # Preload data into hierarchical extremes
    for i in range(warmup_bars):
        if i < len(df):
            he.update(i, df.index, df['high'].values, 
                      df['low'].values, df['close'].values)
    
    for i, row in df.iterrows():
        if i < warmup_bars:
            continue
            
        if i % 500 == 0:
            logger.info(f"Processing bar {i+1}/{len(df)} - {row['time']}")
        
        # Update model
        he.update(i, df.index, df['high'].values, 
                  df['low'].values, df['close'].values)
        
        # Track newly formed levels
        current_l3_high = he.get_level_high_price(2, 0)
        current_l3_low = he.get_level_low_price(2, 0)
        
        # Record new highs
        if not np.isnan(current_l3_high):
            # Only record if it's a new high
            if not historical_levels['highs'] or current_l3_high > max([h[0] for h in historical_levels['highs']]):
                historical_levels['highs'].append((current_l3_high, i, row['time']))
        
        # Record new lows
        if not np.isnan(current_l3_low):
            # Only record if it's a new low
            if not historical_levels['lows'] or current_l3_low < min([l[0] for l in historical_levels['lows']]):
                historical_levels['lows'].append((current_l3_low, i, row['time']))
        
        # Check for retests of historical levels
        no_open_trades = len(open_trades) == 0
        current_price = row['close']
        
        # Check BUY signal (retest of historical low)
        for level_price, level_bar_idx, level_time in historical_levels['lows']:
            # Skip if level was formed recently
            if (i - level_bar_idx) < MIN_BARS_SINCE_LEVEL:
                continue
                
            # Check if price is retesting the level
            level_zone_top = level_price + ENTRY_THRESHOLD
            level_zone_bottom = level_price - ENTRY_THRESHOLD
            
            if level_zone_bottom <= row['low'] <= level_zone_top and no_open_trades:
                entry_price = current_price
                sl_price = entry_price - STOP_LOSS
                tp_price = entry_price + (STOP_LOSS * TAKE_PROFIT_MULTIPLIER)
                
                new_trade = Trade(
                    trade_id=trade_id,
                    entry_time=row['time'],
                    entry_price=entry_price,
                    direction='buy',
                    level_price=level_price,
                    sl=sl_price,
                    tp=tp_price,
                    lot_size=LOT_SIZE
                )
                open_trades.append(new_trade)
                trade_id += 1
                logger.info(f"BUY Entry (ID:{new_trade.trade_id}) at retest of {level_price:.2f} "
                            f"(formed at {level_time}) | Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
                break  # Only take one trade per bar
        
        # Check SELL signal (retest of historical high)
        for level_price, level_bar_idx, level_time in historical_levels['highs']:
            # Skip if level was formed recently
            if (i - level_bar_idx) < MIN_BARS_SINCE_LEVEL:
                continue
                
            # Check if price is retesting the level
            level_zone_top = level_price + ENTRY_THRESHOLD
            level_zone_bottom = level_price - ENTRY_THRESHOLD
            
            if level_zone_bottom <= row['high'] <= level_zone_top and no_open_trades:
                entry_price = current_price
                sl_price = entry_price + STOP_LOSS
                tp_price = entry_price - (STOP_LOSS * TAKE_PROFIT_MULTIPLIER)
                
                new_trade = Trade(
                    trade_id=trade_id,
                    entry_time=row['time'],
                    entry_price=entry_price,
                    direction='sell',
                    level_price=level_price,
                    sl=sl_price,
                    tp=tp_price,
                    lot_size=LOT_SIZE
                )
                open_trades.append(new_trade)
                trade_id += 1
                logger.info(f"SELL Entry (ID:{new_trade.trade_id}) at retest of {level_price:.2f} "
                            f"(formed at {level_time}) | Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
                break  # Only take one trade per bar
        
        # Check for exits
        for trade in open_trades[:]:
            if trade.check_exit(row):
                trade.exit_time = row['time']
                profit = trade.calculate_profit()
                account_balance += profit
                closed_trades.append(trade)
                open_trades.remove(trade)
                
                logger.info(
                    f"Trade ID:{trade.trade_id} Closed ({trade.status.upper()}) | "
                    f"Entry: {trade.entry_price:.2f} | Exit: {trade.exit_price:.2f} | "
                    f"Profit: {profit:.2f} | Balance: {account_balance:.2f}"
                )
    
    # Close any remaining trades
    for trade in open_trades:
        trade.exit_time = df.iloc[-1]['time']
        trade.exit_price = df.iloc[-1]['close']
        trade.status = 'force_close'
        profit = trade.calculate_profit()
        account_balance += profit
        closed_trades.append(trade)
        logger.info(f"Force closed Trade ID:{trade.trade_id} | Profit: {profit:.2f}")
    
    return closed_trades, account_balance, historical_levels

def plot_trades(df, trades, historical_levels):
    """Plot price chart with trade entries and exits"""
    plt.figure(figsize=(16, 10))
    
    # Plot price
    plt.plot(df['time'], df['close'], label='Price', alpha=0.7, linewidth=1.5)
    
    # Plot historical levels
    for level_price, _, level_time in historical_levels['highs']:
        plt.axhline(y=level_price, color='red', linestyle='--', alpha=0.5)
        plt.text(df['time'].iloc[0], level_price, f'High: {level_price:.2f}', 
                 verticalalignment='bottom', horizontalalignment='left', color='red')
    
    for level_price, _, level_time in historical_levels['lows']:
        plt.axhline(y=level_price, color='green', linestyle='--', alpha=0.5)
        plt.text(df['time'].iloc[0], level_price, f'Low: {level_price:.2f}', 
                 verticalalignment='top', horizontalalignment='left', color='green')
    
    # Plot trades
    for trade in trades:
        # Entry point
        entry_color = 'darkgreen' if trade.direction == 'buy' else 'darkred'
        plt.scatter(trade.entry_time, trade.entry_price, 
                    color=entry_color, s=120, 
                    marker='^' if trade.direction == 'buy' else 'v',
                    zorder=5,
                    label=f"{trade.direction.capitalize()} Entry (ID:{trade.trade_id})")
        
        # Exit point
        exit_color = 'blue' if trade.status == 'tp' else 'black'
        plt.scatter(trade.exit_time, trade.exit_price, 
                    color=exit_color, s=120, 
                    marker='o', zorder=5,
                    label=f"Exit {trade.status.upper()} (ID:{trade.trade_id})")
        
        # Trade line
        plt.plot([trade.entry_time, trade.exit_time], 
                 [trade.entry_price, trade.exit_price], 
                 color='purple' if trade.profit > 0 else 'gray', 
                 linestyle='-', alpha=0.7)
        
        # Level price
        plt.axhline(y=trade.level_price, color='orange', linestyle='-', alpha=0.3)
    
    plt.title('Retest Trading Strategy - Trade Placements')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', label='Historical Highs'),
        Line2D([0], [0], color='green', linestyle='--', label='Historical Lows'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='darkgreen', markersize=10, label='Buy Entry'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='darkred', markersize=10, label='Sell Entry'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='TP Exit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='SL Exit'),
        Line2D([0], [0], color='purple', label='Profitable Trade'),
        Line2D([0], [0], color='gray', label='Losing Trade')
    ]
    
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    plt.savefig('retest_trades_visualization.png', dpi=300)
    plt.show()

def generate_report(closed_trades, final_balance, historical_levels):
    if not closed_trades:
        print("No trades executed")
        return
    
    # Detailed trade report
    print("\n===== DETAILED TRADE REPORT =====")
    print(f"{'ID':<5} {'Dir':<6} {'Level':>12} {'Entry Time':<20} {'Entry':>10} "
          f"{'Exit Time':<20} {'Exit':>10} {'Status':<6} {'Profit':>10} {'Level Age':>10}")
    print("-" * 115)
    
    for trade in closed_trades:
        level_age = (trade.entry_time - trade.level_time).total_seconds()/3600 if hasattr(trade, 'level_time') else 0
        print(f"{trade.trade_id:<5} {trade.direction:<6} {trade.level_price:>12.2f} "
              f"{trade.entry_time.strftime('%Y-%m-%d %H:%M:%S'):<20} {trade.entry_price:>10.2f} "
              f"{trade.exit_time.strftime('%Y-%m-%d %H:%M:%S'):<20} {trade.exit_price:>10.2f} "
              f"{trade.status:<6} {trade.profit:>10.2f} {level_age:>10.1f}h")
    
    # Summary metrics
    profits = [t.profit for t in closed_trades]
    winning_trades = [t for t in closed_trades if t.profit > 0]
    losing_trades = [t for t in closed_trades if t.profit <= 0]
    
    total_profit = sum(profits)
    win_rate = len(winning_trades) / len(closed_trades) * 100
    avg_win = np.mean([t.profit for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.profit for t in losing_trades]) if losing_trades else 0
    profit_factor = abs(avg_win/avg_loss) if avg_loss != 0 else float('inf')
    
    print("\n===== PERFORMANCE SUMMARY =====")
    print(f"Initial Balance:    {INITIAL_BALANCE:.2f}")
    print(f"Final Balance:      {final_balance:.2f}")
    print(f"Net Profit:         {total_profit:.2f} ({total_profit/INITIAL_BALANCE*100:.2f}%)")
    print(f"Total Trades:       {len(closed_trades)}")
    print(f"Win Rate:           {win_rate:.2f}%")
    print(f"Avg. Win:           {avg_win:.2f}")
    print(f"Avg. Loss:          {avg_loss:.2f}")
    print(f"Profit Factor:      {profit_factor:.2f}" if avg_loss != 0 else "Profit Factor:      INF")
    print(f"Largest Win:        {max(profits):.2f}" if profits else "Largest Win:        0.00")
    print(f"Largest Loss:       {min(profits):.2f}" if profits else "Largest Loss:       0.00")
    print(f"\nHistorical Levels Tracked: {len(historical_levels['highs'])} highs, {len(historical_levels['lows'])} lows")
    
    # Calculate cumulative balance
    cumulative_balance = INITIAL_BALANCE
    trade_data = []
    for trade in closed_trades:
        cumulative_balance += trade.profit
        trade_data.append({
            'ID': trade.trade_id,
            'Entry Time': trade.entry_time,
            'Exit Time': trade.exit_time,
            'Direction': trade.direction,
            'Level Price': trade.level_price,
            'Entry Price': trade.entry_price,
            'Exit Price': trade.exit_price,
            'SL': trade.sl,
            'TP': trade.tp,
            'Status': trade.status,
            'Profit': trade.profit,
            'Duration (min)': (trade.exit_time - trade.entry_time).total_seconds() / 60,
            'Cumulative Balance': int(cumulative_balance) 
        })

    trade_df = pd.DataFrame(trade_data)
    trade_df.to_csv("retest_trade_report.csv", index=False)
    logger.info("Detailed trade report saved to retest_trade_report.csv")

if __name__ == "__main__":
    logger.info("Starting backtest with RETEST strategy...")
    
    try:
        data = load_data(CSV_PATH)
        logger.info(f"Loaded {len(data)} historical bars")
        
        trades, balance, historical_levels = run_backtest(data)
        
        if trades:
            logger.info("Generating trade report and visualization...")
            generate_report(trades, balance, historical_levels)
            # plot_trades(data, trades, historical_levels)
        else:
            logger.info("No trades executed during backtest")
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise