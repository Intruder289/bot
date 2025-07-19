# local_extreme.py
import pandas as pd
from dataclasses import dataclass

@dataclass
class LocalExtreme:
    ext_type: int    # 1 For high, -1 for low
    index: int       # bar index
    price: float
    timestamp: pd.Timestamp

    conf_index: int  # bar index that its confirmed
    conf_price: float
    conf_timestamp: pd.Timestamp  


def extremes_sanity_checks(ext_df):
    if len(ext_df) < 2:
        return

    # Always alternating
    assert len(ext_df[ext_df['ext_type'] == ext_df['ext_type'].shift()]) == 0

    # extreme index is always increasing
    assert ext_df['index'].diff().min() >= 0

    ext_df['last'] = ext_df['price'].shift()

    # All highs are greater than prior low
    high_exts = ext_df[ext_df['ext_type'] == 1]
    assert len(high_exts[high_exts['price'] <= high_exts['last']]) == 0

    # All lows are less than prior high
    low_exts = ext_df[ext_df['ext_type'] == -1]
    assert len(low_exts[low_exts['price'] >= low_exts['last']]) == 0