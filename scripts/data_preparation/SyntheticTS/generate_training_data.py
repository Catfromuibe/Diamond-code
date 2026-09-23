"""
Generate synthetic time series dataset (Physically-defined Continuous Signal + Layered Noise).
Components: Trend + Quasi-Periodic (Drift) + Transient (Chirp + AM).
Noise: Gaussian + Heavy-tail + Missing.
"""
import json
import os
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

# Constants
dataset_name = 'SyntheticTS'
seed = 42
frequency = 15                  
steps_per_day = 96              
train_val_test_ratio = [0.6, 0.2, 0.2]

def generate_smooth_trend(length: int, seed: int) -> np.ndarray:
    """
    Generate a non-linear global trend.
    """
    np.random.seed(seed)
    # Random walk
    rw = np.cumsum(np.random.normal(0, 0.1, size=length))
    # Apply Gaussian smoothing to make it look like a natural trend
    trend = gaussian_filter1d(rw, sigma=length/50)
    # Normalize
    trend = (trend - trend.mean()) / (trend.std() + 1e-6)
    return trend

def generate_volatile_envelope(length: int, base_level: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    regimes = np.zeros(length)
    cursor = 0
    
    while cursor < length:
        # random duration: keep the state for 50 to 500 time steps
        duration = np.random.randint(50, 500)
        end = min(cursor + duration, length)
        
        # state switch logic remains the same
        if np.random.random() < 0.4: # 40% probability of high noise
            val = 1.0
        else:
            val = 0.1
            
        regimes[cursor:end] = val
        cursor = end

    # smooth processing
    envelope = gaussian_filter1d(regimes, sigma=50)
    return (envelope * base_level)[:, np.newaxis]

def generate_continuous_signal(total_length: int, num_features: int, seed: int) -> np.ndarray:
    """
    [MODIFIED] Compose signal using strict 4 components:
    Trend + Periodic (with Drift) + Chirp + AM.
    """
    np.random.seed(seed)
    t = np.arange(total_length)
    data = np.zeros((total_length, num_features))

    for feat in range(num_features):
        feat_seed = seed + feat * 100
        
        # 1. Component: Trend (Background)
        trend = generate_smooth_trend(total_length, feat_seed)
        
        # 2. Component: Quasi-Periodic (Background with Drift)
        # Real systems have slight frequency drift (Phase Noise)
        periodic = np.zeros(total_length)
        num_freqs = np.random.randint(2, 4)
        for i in range(num_freqs):
            period = np.random.uniform(50, 500) 
            # Phase drift: random walk on phase
            phase_drift = np.cumsum(np.random.normal(0, 0.05, size=total_length))
            
            # Mix standard sine and harmonics
            if i % 2 == 0:
                component = np.sin(2 * np.pi * t / period + phase_drift)
            else:
                component = 0.5 * np.sin(2 * np.pi * t / (period/2) + phase_drift)
            periodic += component
            
        periodic = (periodic - periodic.mean()) / (periodic.std() + 1e-6)

        # 3. Component: Transient Events (Chirp & AM only)
        # use a cursor to ensure events don't overlap randomly
        transient = np.zeros(total_length)
        cursor = 0
        min_gap = 400   # Minimum quiet time between events (Refractory Period)
        max_gap = 1500  # Maximum quiet time
        
        while cursor < total_length - 200:
            # Move cursor forward by a random gap (Anti-clustering)
            gap = np.random.randint(min_gap, max_gap)
            cursor += gap
            
            if cursor >= total_length - 100:
                break

            # Event parameters
            duration = np.random.randint(60, 180)
            end = min(cursor + duration, total_length)
            local_len = end - cursor
            t_local = np.linspace(0, 1, local_len)
            
            # Hanning window for smooth entry/exit
            window = np.hanning(local_len)
            
            # [STRICTLY CHIRP OR AM]
            ev_type = np.random.choice(['chirp', 'am'], p=[0.5, 0.5])
            
            if ev_type == 'chirp':
                # Chirp (Freq sweep)
                f0 = np.random.uniform(5, 10) 
                f1 = np.random.uniform(1, 2)
                sig_frag = signal.chirp(t_local, f0=f0, f1=f1, t1=1, method='linear')
            
            elif ev_type == 'am':
                # AM (Beating)
                c_freq = np.random.uniform(10, 20)
                m_freq = np.random.uniform(2, 5)
                carrier = np.sin(2 * np.pi * c_freq * t_local)
                modulator = 0.5 * (1 + np.sin(2 * np.pi * m_freq * t_local))
                sig_frag = carrier * modulator
            
            # Inject event
            transient[cursor:end] += sig_frag * window * np.random.uniform(1.5, 2.5)
            
            # Advance cursor past the event
            cursor += duration

        # === Combine Components ===
        # When transient is active, it slightly suppresses background (saturation effect)
        transient_active = (np.abs(transient) > 0.01).astype(float)
        transient_smooth = gaussian_filter1d(transient_active, sigma=5) # Smooth transition
        
        background = 0.5 * trend + 0.5 * periodic
        combined = background * (1 - 0.3 * transient_smooth) + 0.5 * transient
        
        # === Normalize Clean Signal ===
        combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-6)
        data[:, feat] = combined

    return data

def generate_stable_envelope(length: int, base_level: float) -> np.ndarray:
    """Generate a stable noise intensity envelope (0.8x to 1.2x)."""
    # Smooth random walk
    rw = np.cumsum(np.random.normal(0, 0.01, size=length))
    rw = (rw - rw.min()) / (rw.max() - rw.min() + 1e-6)
    # Map to [0.8, 1.2] * base_level
    envelope = base_level * (0.8 + 0.4 * rw)
    return envelope[:, np.newaxis]

def add_layered_noise(data: np.ndarray, base_level: float, seed: int) -> np.ndarray:
    """
    Apply Layered Noise (Gaussian + Heavy-tail + Missing).
    If base_level is 0.0, return clean data without any noise.
    """
    # If noise level is 0, return clean data directly
    if base_level == 0.0:
        return data.copy()
    
    np.random.seed(seed)
    length, feats = data.shape
    envelope = generate_volatile_envelope(length, base_level, seed)
    
    final_data = data.copy()
    
    # 1. Base Layer: Gaussian Noise (Always present)
    gaussian = np.random.normal(0, 1, size=data.shape) * envelope
    
    # 2. Impulsive Layer: Heavy-tail (Student-t)
    spike_mask = np.random.random(data.shape) < 0.05
    heavy_noise = np.zeros_like(data)
    raw_spikes = np.random.standard_t(4, size=spike_mask.sum()) 
    raw_spikes = np.clip(raw_spikes, -4, 4) 
    heavy_noise[spike_mask] = raw_spikes * envelope[spike_mask] * 1.5 
    
    # 3. Apply Additive Noise
    final_data += gaussian * 0.8
    final_data += heavy_noise * 0.5
    
    # 4. Missing Data Layer (Dropouts)
    num_drops = int(length / 1000)
    for _ in range(num_drops):
        start = np.random.randint(0, length - 5)
        dur = np.random.randint(1, 5) 
        end = min(start + dur, length)
        final_data[start:end] = 0.0 
    
    return final_data

def add_temporal_features(df):
    """Add temporal features."""
    l = df.shape[0]
    timestamps = []
    timestamps.append(np.array([i % steps_per_day / steps_per_day for i in range(l)]))
    timestamps.append(df.index.dayofweek / 7)
    timestamps.append((df.index.day - 1) / 31)
    timestamps.append((df.index.dayofyear - 1) / 366)
    return np.stack(timestamps, axis=-1).astype(np.float32)

def split_and_save_data(data_noisy, data_clean, output_dir_local, timestamps=None):
    train_ratio, val_ratio, _ = train_val_test_ratio
    train_len = int(data_noisy.shape[0] * train_ratio)
    val_len = int(data_noisy.shape[0] * val_ratio)
    
    splits = {
        'train': (0, train_len),
        'val': (train_len, train_len + val_len),
        'test': (train_len + val_len, data_noisy.shape[0])
    }
    
    os.makedirs(output_dir_local, exist_ok=True)
    print(f"\n[IO] Saving data splits to {output_dir_local}...")
    
    for name, (start, end) in splits.items():
        np.save(os.path.join(output_dir_local, f'{name}_data.npy'), data_noisy[start:end].astype(np.float32))
        np.save(os.path.join(output_dir_local, f'{name}_data_clean.npy'), data_clean[start:end].astype(np.float32))
        print(f"  - {name}: {end-start} steps | Saved X and GT")
        if timestamps is not None:
            np.save(os.path.join(output_dir_local, f'{name}_timestamps.npy'), timestamps[start:end].astype(np.float32))

def save_description(data_shape, dataset_name_local, output_dir_local, num_chunks_local, chunk_size_local, base_noise_level_local):
    description = {
        'name': dataset_name_local,
        'description': 'Physically-defined Continuous Synthetic Dataset',
        'components': ['Smooth Trend', 'Quasi-Periodic (Drift)', 'Transient (Chirp/AM)'],
        'noise': ['Gaussian', 'Clipped Heavy-tail', 'Random Dropouts'],
        'shape': data_shape,
        'config': {
            'base_noise_level': base_noise_level_local
        }
    }
    with open(os.path.join(output_dir_local, 'meta.json'), 'w') as f:
        json.dump(description, f, indent=2)

def generate_single_dataset(noise_level_val, num_chunks_local, chunk_size_local, num_features_local, output_suffix_val="", base_dir_local=None):
    if base_dir_local is None:
        base_dir_local = base_dir
    
    dataset_name_with_suffix = dataset_name + output_suffix_val
    output_dir_local = base_dir_local + f'/datasets/{dataset_name_with_suffix}'
    raw_data_dir_local = base_dir_local + f'/datasets/raw_data/{dataset_name_with_suffix}'
    
    total_len = num_chunks_local * chunk_size_local
    
    print(f"---------- Generating {dataset_name_with_suffix} ----------")
    print(f"Total Steps: {total_len}, Base Noise: {noise_level_val}")

    # 1. Clean Signal: Defined components blended continuously
    data_clean = generate_continuous_signal(total_len, num_features_local, seed=seed)
    
    # 2. Layered Noise: Defined noise types overlaid
    data_noisy = add_layered_noise(data_clean, base_level=noise_level_val, seed=seed+1)
    
    dates = pd.date_range(start='2020-01-01', periods=total_len, freq=f'{frequency}min')
    
    os.makedirs(raw_data_dir_local, exist_ok=True)
    df_noisy = pd.DataFrame(data_noisy, columns=[f'feat_{i}' for i in range(num_features_local)])
    df_noisy.insert(0, 'date', dates)
    df_noisy.to_csv(os.path.join(raw_data_dir_local, f'{dataset_name}.csv'), index=False)
    
    df_clean = pd.DataFrame(data_clean, columns=[f'feat_{i}' for i in range(num_features_local)])
    df_clean.insert(0, 'date', dates)
    df_clean.to_csv(os.path.join(raw_data_dir_local, f'{dataset_name}_clean.csv'), index=False)
    print(f"[IO] CSVs saved to {raw_data_dir_local}")

    df_for_ts = pd.DataFrame(data_noisy)
    df_for_ts.index = dates
    timestamps = add_temporal_features(df_for_ts)
    
    split_and_save_data(data_noisy, data_clean, output_dir_local, timestamps)
    save_description(data_noisy.shape, dataset_name_with_suffix, output_dir_local, num_chunks_local, chunk_size_local, noise_level_val)
    print(f"\n---------- Generation Completed! ----------\n")

def main():
    base_noise_level = 0.3
    num_chunks = 100
    chunk_size = 336
    num_features = 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_level", type=float, default=base_noise_level)
    parser.add_argument("--num_chunks", type=int, default=num_chunks)
    parser.add_argument("--chunk_size", type=int, default=chunk_size)
    parser.add_argument("--num_features", type=int, default=num_features)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--generate_all", action="store_true")
    
    args = parser.parse_args()
    
    if args.generate_all:
        noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        for i, noise_level_val in enumerate(noise_levels, 1):
            if noise_level_val == 0.0:
                output_suffix = "_noise0.0"  # or "_clean" if you prefer
            else:
                output_suffix = f"_noise{noise_level_val:.1f}"
            generate_single_dataset(noise_level_val, args.num_chunks, args.chunk_size, args.num_features, output_suffix)
        return 0
    
    output_suffix = args.output_suffix if args.output_suffix else ""
    generate_single_dataset(args.noise_level, args.num_chunks, args.chunk_size, args.num_features, output_suffix)
    return 0

if __name__ == '__main__':
    main()