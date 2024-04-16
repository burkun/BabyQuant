import random
import pandas as pd
from enum import Enum
from torch.utils.data import DataLoader, Dataset

class DataType(Enum):
    Day = 1
    Minute = 2

class FileReader(object):
    def __init__(self, fpath, seq_type, seq_length):
        self.fpath = fpath
        self.seq_type = seq_type
        self.seq_length = seq_length
        self.stride = self.seq_length // 2

    def load(self):
        self.fout = pd.read_csv(self.fpath)
    
    def seq_size(self):
        pass

class StockDataset(Dataset):
    def __init__(self, files_list, seq_length):
        random.seed(2024)
        random.shuffle(files_list)
        self.file_list = files_list

    def __read_data__(self):

 
