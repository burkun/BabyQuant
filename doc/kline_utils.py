
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


def simple_forword_adjust_once(df):
  raise ValueError('Invoke a not implemented function')


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