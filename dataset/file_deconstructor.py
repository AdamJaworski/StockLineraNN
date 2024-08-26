import numpy as np
from utlities.utilities import get_rsi_for_data

class File:
    open_list: list
    close_list: list
    high_list: list
    low_list: list
    rsi_list: list
    candle_list: list
    size: int
    reference_price: float
    candle_raw: list
    def __init__(self, file_path):
        self.open_list   = []
        self.close_list  = []
        self.high_list   = []
        self.low_list    = []
        self.rsi_list    = []
        self.candle_list = []
        self.candle_raw = []
        csv_data = np.loadtxt(file_path, delimiter=',', dtype=str)

        self.index_map = index_map = {'open': None, 'close': None, 'high': None, 'low': None}
        for index, tag in enumerate(csv_data[0]):
            tag_lower = str(tag).lower()
            for key in index_map:
                if key in tag_lower and not index_map[key]:
                    index_map[key] = index


        rsi_data = []
        for candle in csv_data[1:15]:
            rsi_data.append(float(candle[index_map['close']]))

        self.reference_price = float(csv_data[14][index_map['close']])
        for index, candle in enumerate(csv_data[15:]):
            rsi_data.append(float(candle[index_map['close']]))
            rsi_data.pop(0)
            self.open_list.append(   (float(candle[index_map['open']])  - float(csv_data[14 + index][index_map['close']]))   / float(csv_data[14 + index][index_map['close']]))
            self.close_list.append(  (float(candle[index_map['close']]) - float(csv_data[14 + index][index_map['close']]))   / float(csv_data[14 + index][index_map['close']]))
            self.high_list.append(   (float(candle[index_map['high']])  - float(csv_data[14 + index][index_map['close']]))   / float(csv_data[14 + index][index_map['close']]))
            self.low_list.append(    (float(candle[index_map['low']])   - float(csv_data[14 + index][index_map['close']]))   / float(csv_data[14 + index][index_map['close']]))
            self.rsi_list.append(    float(get_rsi_for_data(rsi_data))    / 100)
            self.candle_list.append([self.open_list[index], self.high_list[index], self.low_list[index], self.close_list[index],self.rsi_list[index]])
            self.candle_raw.append([float(candle[index_map['open']]), float(candle[index_map['high']]), float(candle[index_map['low']]), float(candle[index_map['close']]),  self.rsi_list[index] * 100])


        self.size = len(self.candle_list)

    def get_size(self):
        return self.size

    def get_candles(self):
        return self.candle_list

    def get_open(self, index):
        return self.open_list[index]

    def get_close(self, index):
        return self.close_list[index]

    def get_high(self, index):
        return self.high_list[index]

    def get_low(self, index):
        return self.low_list[index]

    def get_rsi(self, index):
        return self.rsi_list[index]


if __name__ == "__main__":
    file = File(r'..\Data\DataTraining\GPW_DLY_ALE, 1D.csv')
    print(file.get_candles())