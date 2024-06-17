import pandas as pd
import numpy as np
import random
import math
import collections
from enum import Enum

from torch.utils.data import IterableDataset, DataLoader

from misc.time_feature import time_features


class DataType(Enum):
    CN_STOCK_5MIN = 1
    CN_STOCK_DAY = 2
    US_STOCK_5MIN = 3
    US_STOCK_DAY = 4
    UNKOWN = 5


random.seed(2024)
np.random.seed(2024)


class StockDataset(IterableDataset):
    ChannelNames = ["open", "close", "high", "low", "vwap", "vol", "amount"]
    def __init__(self,
                 files_list: list,
                 maxseq_len: int,
                 random_split_ratio: float,
                 use_time_feature: bool,
                 data_type: DataType,
                 random_mask_ratio=0,
                 cache_size=10):
        super().__init__()
        random.shuffle(files_list)
        self.files_list = files_list
        self.maxseq_len = maxseq_len
        self.random_split_ratio = random_split_ratio
        self.random_mask_ratio = random_mask_ratio
        self.use_time_feature = use_time_feature
        self.data_type = data_type
        self.file_idx = 0
        self.cache_size = cache_size
        self.data_cache = collections.deque()  # Using deque for efficient pops

    def load_and_cache_file(self):
        while len(self.data_cache) < self.cache_size and self.file_idx < len(self.files_list):
            file_path = self.files_list[self.file_idx]
            df = pd.read_csv(file_path)
            df.sort_values(by='time', inplace=True, ignore_index=True)
            real_length = len(df)
            pad_length = max(real_length, self.maxseq_len *
                             math.ceil(real_length / self.maxseq_len))
            df = df.reindex(range(pad_length), method='pad')
            data_df = df[StockDataset.ChannelNames]
            begin_idx = list(range(0, real_length, self.maxseq_len))
            time_df = None
            if self.use_time_feature:
                if self.data_type in (DataType.CN_STOCK_DAY, DataType.US_STOCK_DAY):
                    time_df = pd.to_datetime(df['time'], format='%Y%m%d').dt
                    time_df = time_features(time_df, 'd')
                elif self.data_type in (DataType.CN_STOCK_5MIN, DataType.US_STOCK_5MIN):
                    time_df = pd.to_datetime(df['time']).dt
                    time_df = time_features(time_df, 'min')
                else:
                    raise Exception("can not support time type")

            extra_num = int(len(begin_idx) * self.random_split_ratio)
            extra_samples = np.random.randint(
                0, min(pad_length - self.maxseq_len, real_length), size=extra_num)
            begin_idx.extend(extra_samples)

            for start in begin_idx:
                sub_seq = data_df.iloc[start:start + self.maxseq_len]
                sub_time_df = None
                if time_df is not None:
                    sub_time_df = time_df[start:start + self.maxseq_len, :]
                if self.random_mask_ratio > 0:
                    num_rows_to_mask = int(
                        np.ceil(len(sub_seq) * self.random_mask_ratio))
                    # 随机选择要掩码的行索引
                    rows_to_mask = np.random.choice(
                        sub_seq.index, size=num_rows_to_mask, replace=False)
                    # 对选中的行应用掩码
                    sub_seq.loc[rows_to_mask] = 0
                sub_length = min(real_length - start, self.maxseq_len)
                if sub_time_df is None:
                    self.data_cache.append([sub_seq.to_numpy(), sub_length])
                else:
                    self.data_cache.append(
                        [sub_seq.to_numpy(), sub_length, sub_time_df])
            self.file_idx += 1
        random.shuffle(self.data_cache)

    def reset(self):
        self.file_idx = 0
        random.shuffle(self.files_list)
        self.data_cache.clear()

    def __iter__(self):
        while self.file_idx < len(self.files_list):
            if not self.data_cache:
                self.load_and_cache_file()
                if not self.data_cache:
                    break
            while self.data_cache:
                yield self.data_cache.popleft()  # Efficient pop from left


if __name__ == "__main__":
    files_list = ["data/process/day/AAPL.csv"]
    dataset = StockDataset(files_list,
                           maxseq_len=100,
                           random_split_ratio=0.1,
                           use_time_feature=True,
                           data_type=DataType.US_STOCK_DAY,
                           cache_size=10)
    loader = DataLoader(dataset, batch_size=4)
    for seq, mask, a in loader:
        print(seq, mask, a)
