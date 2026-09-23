import json
import os

import numpy as np
import pandas as pd

# Current path
current_dir = os.path.dirname(os.path.abspath(__file__))

# target path
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

# Hyperparameters
dataset_name = 'Solar'
csv_path = base_dir + f'/datasets/raw_data/{dataset_name}/{dataset_name}.csv'
txt_path = base_dir + f'/datasets/raw_data/{dataset_name}/solar_AL.txt'
graph_file_path = None
output_dir = base_dir + f'/datasets/{dataset_name}'
target_channel = [0]
add_time_of_day = True
add_day_of_week = True
add_day_of_month = True
add_day_of_year = True
steps_per_day = 144  # 10-minute solar power
frequency = 1440 // steps_per_day
domain = 'electricity'
timestamps_desc = ['time of day', 'day of week', 'day of month', 'day of year']
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    'norm_each_channel': True,
    'rescale': False,
    'metrics': ['MAE', 'MSE'],
    'null_val': np.nan
}

def load_and_preprocess_data():
    '''Load Solar-Energy (137 Alabama PV plants, 10-min, 2006).'''

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df_index = pd.to_datetime(df['date'].values, format='%Y-%m-%d %H:%M:%S').to_numpy()
        df = df[df.columns[1:]]
        df.index = df_index
    else:
        df = pd.read_csv(txt_path, header=None)
        df_index = pd.date_range('2006-01-01 00:00:00', periods=len(df), freq='10min')
        df.index = df_index
        df.columns = [str(i) for i in range(df.shape[1])]
        # Keep a CSV copy in the same format as Weather / Traffic.
        out = df.copy()
        out.insert(0, 'date', df_index.strftime('%Y-%m-%d %H:%M:%S'))
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        out.to_csv(csv_path, index=False)
        print(f'Wrote {csv_path}')
    print(f'Raw time series shape: {df.shape}')
    return df

def add_temporal_features(df):
    '''Add time of day and day of week as features to the data.'''
    l = df.shape[0]
    timestamps = []

    if add_time_of_day:
        tod = [i % steps_per_day / steps_per_day for i in range(l)]
        tod = np.array(tod)
        timestamps.append(tod)

    if add_day_of_week:
        dow = df.index.dayofweek / 7
        timestamps.append(dow.values)

    if add_day_of_month:
        dom = (df.index.day - 1) / 31
        timestamps.append(dom.values)

    if add_day_of_year:
        doy = (df.index.dayofyear - 1) / 366
        timestamps.append(doy.values)

    timestamps = np.stack(timestamps, axis=-1)
    return timestamps


def split_and_save_data(data, timestamps):
    '''Save the preprocessed data to a binary file.'''
    train_ratio, val_ratio, _ = regular_settings['train_val_test_ratio']
    train_len = int(data.shape[0] * train_ratio)
    val_len = int(data.shape[0] * val_ratio)

    train_data = data[:train_len].astype(np.float32)
    val_data = data[train_len : train_len + val_len].astype(np.float32)
    test_data = data[train_len + val_len :].astype(np.float32)
    train_timestamps = timestamps[:train_len].astype(np.float32)
    val_timestamps = timestamps[train_len : train_len + val_len].astype(np.float32)
    test_timestamps = timestamps[train_len + val_len :].astype(np.float32)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"train_data shape: {train_data.shape}")
    np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
    print(f"val_data shape: {val_data.shape}")
    np.save(os.path.join(output_dir, 'val_data.npy'), val_data)
    print(f"test_data shape: {test_data.shape}")
    np.save(os.path.join(output_dir, 'test_data.npy'), test_data)
    print(f"train_timestamps shape: {train_timestamps.shape}")
    np.save(os.path.join(output_dir, 'train_timestamps.npy'), train_timestamps)
    print(f"val_timestamps shape: {val_timestamps.shape}")
    np.save(os.path.join(output_dir, 'val_timestamps.npy'), val_timestamps)
    print(f"test_timestamps shape: {test_timestamps.shape}")
    np.save(os.path.join(output_dir, 'test_timestamps.npy'), test_timestamps)
    print(f'Data saved to {output_dir}')

def save_description(data, timestamps):
    '''Save a description of the dataset to a JSON file.'''
    description = {
        'name': dataset_name,
        'domain': domain,
        'frequency (minutes)': frequency,
        'shape': data.shape,
        'timestamps_shape': timestamps.shape,
        'timestamps_description': timestamps_desc,
        'num_time_steps': data.shape[0],
        'num_vars': data.shape[1],
        'has_graph': graph_file_path is not None,
        'regular_settings': regular_settings,
    }
    description_path = os.path.join(output_dir, 'meta.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f'Description saved to {description_path}')
    print(description)
    print('\n')

def main():
    print(f"---------- Generating {dataset_name} data ----------")

    df = load_and_preprocess_data()
    timestamps = add_temporal_features(df)
    split_and_save_data(df.values, timestamps)
    save_description(df.values, timestamps)

if __name__ == '__main__':
    main()
