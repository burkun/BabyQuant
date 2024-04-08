import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class PriceAjust(object):
    @staticmethod
    def simple_backword_adjust_once(df):
        if 'backward_A' not in df.columns:
            # no price adjust factor, do nothing
            return 
        df["backward_A"] = df["backward_A"].ffill()
        df["backward_B"] = df["backward_B"].ffill()
        df.fillna({'backward_A': 1}, inplace=True)
        df.fillna({'backward_B': 0}, inplace=True)
        cols = ['open', 'high', 'low', 'close']
        for col in cols:
            df[col] = df[col] * df['backward_A']  + df['backward_B']
        base_change_rate = df.loc[0, 'change_rate']
        df['change_rate'] = (df['close'] / df['close'].shift(1) - 1) * 100
        df.loc[0, 'change_rate'] = base_change_rate
        df['vol'] = df['vol'] / df['backward_A']
        return df
    
    @staticmethod
    def full_backward_adjust(df):
        """
        累积后复权，kline中出现间断点，其中对价格信息，
        price = price / factor, volume = volume * factor

        参数:
        - df (pd.DataFrame): [["CODE": "backward_A", "backward_B"]]

        返回值:
        - 复权后的dataframe

        使用示例:
        >>> df = full_backward_adjust(df)
        """
        def cum_backward_factor(df):
            index = df['backward_A'][df['backward_A'].notna()].index
            values_a = df['backward_A'][df['backward_A'].notna()].values
            values_b = df['backward_B'][df['backward_B'].notna()].values
            cum_b = 0.0
            cum_a = 1.0
            a = []
            b = []
            for idx in range(len(index)):
                if idx == 0:
                    cum_b = values_b[idx]
                else:
                    cum_b = cum_b * values_a[idx] + values_b[idx]
                cum_a = cum_a * values_a[idx]
                b.append(cum_b)
                a.append(cum_a)
            df.loc[index, 'backward_A'] = a
            df.loc[index, 'backward_B'] = b
            return df
        # Check if the necessary columns exist
        if 'backward_A' not in df.columns or 'backward_B' not in df.columns:
            return df  # Return the original DataFrame if the columns are missing
        df = cum_backward_factor(df)
        # Set missing values for backward_A to 1 and backward_B to 0
        df['backward_A'] = df['backward_A'].ffill()
        df['backward_B'] = df['backward_B'].ffill()
        df.fillna({"backward_A": 1}, inplace=True)
        df.fillna({"backward_B": 0}, inplace=True)
        # Adjust the price columns using the backward adjustment factors
        cols = ['open', 'high', 'low', 'close']
        for col in cols:
            df[col] = df[col] * df['backward_A'] + df['backward_B']
        # Adjust the volume using the backward adjustment factor for backward_A
        df['vol'] = df['vol'] / df['backward_A']
        return df

class StandNorm(object):

    def __init__(self, prefix):
        self.param = {}
        self.prefix = prefix
    
    def fit_transform(self, pd, column_name):
        pd_column = pd[column_name]
        self.param[column_name] = [
            pd_column.mean(),
            pd_column.std()
        ]
        pd_column_new = (pd_column - self.param[column_name][0]) / self.param[column_name][1]
        return pd_column_new
    
    def save(self, code, out_dir):
        out_file = open("{}/{}{}.norm".format(out_dir, self.prefix, code), "w")
        for fname in self.param:
            out_file.write("{}\t{}\t{}\t{}\n".format(code, 
                                                     fname, 
                                                     self.param[fname][0], 
                                                     self.param[fname][1]))
        out_file.close()
        

class SeqExtractor(object):
    DAY = "D_"
    MINUTE = "M_"

    COLUMNS = [
        "code",
        "time",
        "open",
        "close",
        "high",
        "low",
        "vol",
        "amount"
    ]
    @staticmethod
    def extract_one(file_path, out_dir, seq_type):
        if not os.path.exists(file_path):
            print("input path {} not exists".format(file_path))
            return file_path
        assert seq_type in (SeqExtractor.DAY, SeqExtractor.MINUTE)
        code = file_path.rsplit("/", 1)[-1].split(".csv")[0]
        sn = StandNorm("N_" + seq_type)
        df = pd.read_csv(file_path)
        if seq_type == SeqExtractor.DAY:
            df = PriceAjust.full_backward_adjust(df)
        df = df.fillna(0.0)
        for column in ("open", "close", "high", "low", "vol", "amount"):
            df[column] = sn.fit_transform(df, column)
        df = df[SeqExtractor.COLUMNS]
        df.to_csv(os.path.join(out_dir, code + ".csv"), index=False)
        sn.save(code, out_dir)
        return file_path
    
    @staticmethod
    def extract_all(day_dir, out_dir, seq_type):
        os.makedirs(out_dir, exist_ok=True)
        files = glob.glob(os.path.join(day_dir, "*.csv"))
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_list = list()
            for f in files:
                f = executor.submit(SeqExtractor.extract_one, f, out_dir, seq_type)
                future_list.append(f)
        for f in tqdm(future_list):
            f.result()
        SeqExtractor.merge_norm(out_dir, seq_type)

    @staticmethod
    def merge_norm(out_dir, seq_type):
        files = glob.glob(os.path.join(out_dir, "N_" + seq_type + "*"))
        fout = open(os.path.join(out_dir, "Normal.txt"), "w")
        for file_path in files:
            with open(file_path) as fin:
                for line in fin:
                    fout.write(line)
            os.remove(file_path)
        fout.close()

"""
对单只股票数据进行norm
"""
if __name__ == "__main__":
    day_dir = "data/raw/day"
    out_dir = "data/process/day"
    SeqExtractor.extract_all(day_dir, out_dir, SeqExtractor.DAY)
    min_dir = "data/raw/5min"
    out_dir = "data/process/5min"
    SeqExtractor.extract_all(min_dir, out_dir, SeqExtractor.MINUTE)
