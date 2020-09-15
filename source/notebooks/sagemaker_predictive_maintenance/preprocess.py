import numpy as np
import pandas as pd

columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14','s15', 's16', 's17', 's18', 's19', 's20', 's21']

def preprocess_data(data_location, columns=columns):
    # normalize sensor readings
    train_df = []
    eps = 0.000001  # for floating point issues during normalization
    for i in range(1, 5):
        df = pd.read_csv('{}/train_FD{:03d}.txt'.format(data_location ,i), delimiter=' ', header=None)
        df.drop(df.columns[[26, 27]], axis=1, inplace=True)
        df.columns = columns
        df[columns[2:]] = (df[columns[2:]] - df[columns[2:]].min() + eps) / (
                    df[columns[2:]].max() - df[columns[2:]].min() + eps)
        train_df.append(df)

    test_df = []

    # compute RUL (remaining useful life)
    for i, df in enumerate(train_df):
        rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        df = df.merge(rul, on=['id'], how='left')
        df['RUL'] = df['max'] - df['cycle']
        df.drop('max', axis=1, inplace=True)
        train_df[i] = df

    for i in range(1, 5):
        # Load time series
        df = pd.read_csv('{}/test_FD{:03d}.txt'.format(data_location, i), delimiter=' ', header=None)
        df.drop(df.columns[[26, 27]], axis=1, inplace=True)

        # Load the RUL values
        df_rul = pd.read_csv('{}/RUL_FD{:03d}.txt'.format(data_location, i), delimiter=' ', header=None)
        df_rul.drop(df_rul.columns[1], axis=1, inplace=True)
        df_rul.index += 1

        # Merge RUL and timeseries and compute RUL per timestamp
        df = df.merge(df_rul, left_on=df.columns[0], right_index=True, how='left')
        df.columns = columns + ['RUL_end']
        rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        df = df.merge(rul, on=['id'], how='left')  # We get the number of cycles per series
        df['RUL'] = df['max'] + df['RUL_end'] - df[
            'cycle']  # The RUL is the number of cycles per series + RUL - how many cycles have already ran
        df.drop(['max', 'RUL_end'], axis=1, inplace=True)

        # Normalize
        df[columns[2:]] = (df[columns[2:]] - df[columns[2:]].min() + eps) / (
                    df[columns[2:]].max() - df[columns[2:]].min() + eps)
        test_df.append(df)

    return train_df, test_df, columns