# hierarchicak_extremes.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import copy

from atr_directional_change import ATRDirectionalChange 
from local_extreme import LocalExtreme, extremes_sanity_checks

class HierarchicalExtremes:
    def __init__(self, levels: int, atr_lookback: int):
        self._base_dc = ATRDirectionalChange(atr_lookback)
        self._levels = levels
        
        self.extremes = []
        for x in range(levels):
            self.extremes.append([])

    @staticmethod
    def _comparison(x, y, ext_type: int):
        if ext_type == 1:
            return x > y
        else:
            return x < y 
    
    def _new_ext(
        self, level, conf_i, conf_price, conf_time, ext_type
    ):
        '''
        This function called when a new extreme (top or bottom) is confirmed
        at the given level.

        It will check for / confirm an extreme at the next level (level + 1) 
        '''
        if level >= self._levels - 1:
            return
        
        ext_i = len(self.extremes[level]) - 1
        new_ext = self.extremes[level][ext_i]
        assert new_ext.ext_type == ext_type

        # There is not at least 2 prior extremes of the same type
        if ext_i < 4: 
            return

        # Previous extreme is potential a next level extreme
        prev_ext = self.extremes[level][ext_i - 2]
        assert prev_ext.ext_type == ext_type
        if not self._comparison(prev_ext.price, new_ext.price, ext_type):
            return

        # Find the previous extreme on the next level (level + 1) 
        prev_next_lvl = None
        if len(self.extremes[level + 1]) > 0:
            prev_next_lvl = self.extremes[level + 1][-1]
            if prev_next_lvl.ext_type != ext_type:
                if not self._comparison(prev_ext.price, prev_next_lvl.price, ext_type):
                    return 
        
        # Find ext before the previous
        # Loop to deal with equal priced highs
        for prior_i in range(ext_i - 4, -1, -2): 
            prior = self.extremes[level][prior_i]
            assert prior.ext_type == ext_type

            # This invalidates the potential next level extreme
            if self._comparison(prior.price, prev_ext.price, ext_type):
                return

            if prev_next_lvl is not None and prior.index <= prev_next_lvl.index:
                break
            
            # Move back extreme point on equal price
            elif prior.price == prev_ext.price:
                prev_ext = prior
            elif self._comparison(prior.price, prev_ext.price, -ext_type):
                break

        # new extreme of next level 
        new_ext = copy.copy(prev_ext)

        # Update confirmations to current
        new_ext.conf_index = conf_i
        new_ext.conf_price = conf_price
        new_ext.conf_timestamp = conf_time

        # Prior extreme at next level is of the same type.
        # Find an intermediate extreme to upgrade to ensure alternating
        # The lowest low between two highs
        # Or highest high between two lows
        if prev_next_lvl is not None and prev_next_lvl.ext_type == ext_type:
            upgrade_point = None
            for j in range(ext_i - 1, -1, -2):
                prior = self.extremes[level][j]
                assert prior.ext_type == -ext_type

                # Only look between two high's indexs
                if prior.index >= new_ext.index:
                    continue
                if prior.index <= prev_next_lvl.index:
                    break
                if upgrade_point is None or not self._comparison(prior.price, upgrade_point.price, ext_type):
                    upgrade_point = prior

            assert upgrade_point is not None
            upgraded = copy.copy(upgrade_point)
            upgraded.conf_index = conf_i
            upgraded.conf_price = conf_price
            upgraded.conf_timestamp = conf_time
            self.extremes[level + 1].append(upgraded)
            self._new_ext(level + 1, conf_i, conf_price, conf_time, -ext_type)

        self.extremes[level + 1].append(new_ext)
        self._new_ext(level + 1, conf_i, conf_price, conf_time, ext_type)

    def update(
        self, i: int, time_index: pd.DatetimeIndex, 
        high: np.array, low: np.array, close: np.array
    ):
        prev_len = len(self._base_dc.extremes)
        self._base_dc.update(i, time_index, high, low, close)

        new_dc_point = len(self._base_dc.extremes) > prev_len
        if not new_dc_point:
            return
        
        new_ext = self._base_dc.extremes[-1]
        self.extremes[0].append(new_ext)
        
        self._new_ext(0, i, close[i], time_index[i], new_ext.ext_type)
    
    def _get_level_extreme(self, level, ext_type: int, lag=0) -> LocalExtreme:
        lvl_len = len(self.extremes[level])
        if lvl_len == 0:
            return None
        last_ext = self.extremes[level][-1]

        offset = 0
        if last_ext.ext_type != ext_type:
            offset = 1
        
        l2 = lag * 2
        if l2 + offset >= len(self.extremes[level]):
            return None
        return self.extremes[level][-(l2 + offset + 1)]
    
    def get_level_high(self, level: int, lag: int = 0):
        return self._get_level_extreme(level, 1, lag)
    
    def get_level_low(self, level: int, lag: int = 0):
        return self._get_level_extreme(level, -1, lag)
    
    def get_level_high_price(self, level: int, lag: int = 0) -> float:
        lvl = self._get_level_extreme(level, 1, lag)
        if lvl is None:
            return np.nan
        else:
            return lvl.price
    
    def get_level_low_price(self, level: int, lag: int = 0) -> float:
        lvl = self._get_level_extreme(level, -1, lag)
        if lvl is None:
            return np.nan
        else:
            return lvl.price


if __name__ == '__main__':
    df = pd.read_parquet('BTCUSD.pq')
    he = HierarchicalExtremes(levels=5, atr_lookback=24 * 60)

    h = df['high'].to_numpy()
    l = df['low'].to_numpy()
    c = df['close'].to_numpy()
    
    lvl3_low = np.full(c.shape[0], np.nan) 
    for i in range(len(h)):
        he.update(i, df.index, h, l, c)
        
        lvl3_low[i] = he.get_level_low_price(3)
    
    df['lvl3_low'] = lvl3_low
    df['close'].plot()
    df['lvl3_low'].plot()
    plt.show()
     