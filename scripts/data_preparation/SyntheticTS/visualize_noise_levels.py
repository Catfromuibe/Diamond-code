#!/usr/bin/env python
"""
Visualize synthetic datasets with different noise levels and analyze statistics.
Merges signal visualization (Time/Freq) with statistical metric analysis (SNR/SFM/Error).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import io

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust base_dir to point to the project root
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..'))) 
# Fallback if relative path fails
if not os.path.exists(os.path.join(base_dir, 'datasets')):
     base_dir = os.getcwd()

raw_data_dir = os.path.join(base_dir, 'datasets/raw_data')

# Noise levels to visualize
noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
dataset_name = 'SyntheticTS'

# Visualization settings
sample_size = 5000  # Number of points to visualize (for performance)
start_idx = 0       # Start index for visualization

def load_data(noise_level):
    """Load clean and noisy data for a given noise level."""
    suffix = f"_noise{noise_level:.1f}"
    dataset_dir = os.path.join(raw_data_dir, f"{dataset_name}{suffix}")
    
    clean_path = os.path.join(dataset_dir, f"{dataset_name}_clean.csv")
    noisy_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
    
    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        print(f"Warning: Files not found for noise_level={noise_level} at {dataset_dir}")
        return None, None
    
    clean_df = pd.read_csv(clean_path)
    noisy_df = pd.read_csv(noisy_path)
    
    # Parse date column if it exists
    if 'date' in clean_df.columns:
        clean_df['date'] = pd.to_datetime(clean_df['date'])
    if 'date' in noisy_df.columns:
        noisy_df['date'] = pd.to_datetime(noisy_df['date'])
    
    return clean_df, noisy_df

def compute_fft_spectrum(signal_data, dt=1.0):
    """Compute FFT spectrum and return frequencies and amplitudes."""
    fft_vals = np.fft.rfft(signal_data)
    amplitudes = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(len(signal_data), d=dt)
    return freqs, amplitudes

def compute_sfm(signal_data):
    """
    Compute Spectral Flatness Measure (SFM).
    SFM = Geometric Mean(P) / Arithmetic Mean(P)
    """
    fft_vals = np.fft.rfft(signal_data)
    power = np.abs(fft_vals) ** 2
    power = power + 1e-10  # Avoid zeros
    
    geometric_mean = np.exp(np.mean(np.log(power)))
    arithmetic_mean = np.mean(power)
    sfm = geometric_mean / arithmetic_mean
    return sfm

def create_comparison_plot():
    """Create a comprehensive comparison plot with time and frequency domains."""
    fig, axes = plt.subplots(len(noise_levels), 4, figsize=(32, 4 * len(noise_levels)))
    
    if len(noise_levels) == 1:
        axes = axes.reshape(1, -1)
    
    pct = 99
    
    for idx, noise_level in enumerate(noise_levels):
        clean_df, noisy_df = load_data(noise_level)
        
        if clean_df is None or noisy_df is None:
            continue
        
        total_len = len(clean_df)
        end_idx = min(start_idx + sample_size, total_len)
        
        clean_sample = clean_df.iloc[start_idx:end_idx]
        noisy_sample = noisy_df.iloc[start_idx:end_idx]
        
        clean_values = clean_sample['feat_0'].values
        noisy_values = noisy_sample['feat_0'].values
        noise_component = noisy_values - clean_values
        
        x_data = clean_sample['date'] if 'date' in clean_sample.columns else np.arange(len(clean_sample))

        # 1. Clean Signal
        axes[idx, 0].plot(x_data, clean_values, linewidth=1.5, alpha=0.8, color='#2E86AB')
        axes[idx, 0].set_title(f'Clean Signal (noise={noise_level})', fontsize=12, fontweight='bold')
        axes[idx, 0].set_ylabel('Value')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # 2. Noisy Signal
        axes[idx, 1].plot(x_data, noisy_values, linewidth=1.5, alpha=0.8, color='#A23B72')
        axes[idx, 1].set_title(f'Noisy Signal (noise={noise_level})', fontsize=12, fontweight='bold')
        axes[idx, 1].grid(True, alpha=0.3)
        
        # 3. Frequency Domain
        dt = 1.0
        freqs_clean, amp_clean = compute_fft_spectrum(clean_values, dt=dt)
        freqs_noisy, amp_noisy = compute_fft_spectrum(noisy_values, dt=dt)
        freqs_noise, amp_noise = compute_fft_spectrum(noise_component, dt=dt)
        
        thresh_signal = np.percentile(amp_clean, pct)
        thresh_noise = np.percentile(amp_noise, pct)
        
        axes[idx, 2].semilogy(freqs_clean, amp_clean, label='Clean', lw=2.2, alpha=0.95, color='#2E86AB')
        axes[idx, 2].semilogy(freqs_noisy, amp_noisy, label='Noisy', lw=1.8, alpha=0.55, color='#A23B72')
        axes[idx, 2].axhline(thresh_signal, color='#2ECC71', linestyle='--', label=f'Sig {pct}%', alpha=0.8)
        axes[idx, 2].set_title(f'Freq Domain (noise={noise_level})', fontsize=12, fontweight='bold')
        if idx == 0: axes[idx, 2].legend(loc='upper right')
        axes[idx, 2].grid(True, alpha=0.35)
        
        # 4. Reconstruction Analysis
        fft_noisy_val = np.fft.rfft(noisy_values)
        mask_noisy = amp_noisy > thresh_noise
        fft_noisy_filt = fft_noisy_val * mask_noisy
        recon_noisy = np.fft.irfft(fft_noisy_filt, n=len(noisy_values))
        
        axes[idx, 3].plot(x_data, clean_values, color='#34495E', alpha=0.35, lw=3.5, label='GT Clean')
        axes[idx, 3].plot(x_data, recon_noisy, color='#E74C3C', alpha=0.75, lw=2.2, label='Filtered')
        axes[idx, 3].set_title(f'Reconstruction ({pct}th pct)', fontsize=12, fontweight='bold')
        if idx == 0: axes[idx, 3].legend(loc='upper right')
        axes[idx, 3].grid(True, alpha=0.35)
    
    plt.tight_layout()
    return fig

def create_overlay_plot():
    """Create an overlay plot showing clean and noisy together."""
    fig, axes = plt.subplots(len(noise_levels), 1, figsize=(16, 4 * len(noise_levels)))
    
    if len(noise_levels) == 1: axes = [axes]
    
    for idx, noise_level in enumerate(noise_levels):
        clean_df, noisy_df = load_data(noise_level)
        if clean_df is None: continue
        
        end_idx = min(start_idx + sample_size, len(clean_df))
        clean_sample = clean_df.iloc[start_idx:end_idx]
        noisy_sample = noisy_df.iloc[start_idx:end_idx]
        x_data = clean_sample['date'] if 'date' in clean_sample.columns else np.arange(len(clean_sample))

        axes[idx].plot(x_data, clean_sample['feat_0'], label='Clean', lw=2, alpha=0.7, color='#2E86AB')
        axes[idx].plot(x_data, noisy_sample['feat_0'], label='Noisy', lw=1.5, alpha=0.6, color='#A23B72')
        axes[idx].set_title(f'Noise Level = {noise_level}', fontsize=12, fontweight='bold')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_statistics_table():
    """Compute statistics for all noise levels."""
    stats_data = []
    
    for noise_level in noise_levels:
        clean_df, noisy_df = load_data(noise_level)
        if clean_df is None: continue
        
        clean_vals = clean_df['feat_0'].values
        noisy_vals = noisy_df['feat_0'].values
        
        stats_data.append({
            'Noise Level': noise_level,
            'Data Points': len(clean_vals),
            'Clean Mean': np.mean(clean_vals),
            'Clean Std': np.std(clean_vals),
            'Clean SFM': compute_sfm(clean_vals),
            'Noisy Mean': np.mean(noisy_vals),
            'Noisy Std': np.std(noisy_vals),
            'Noisy SFM': compute_sfm(noisy_vals),
            'MSE': np.mean((clean_vals - noisy_vals) ** 2),
            'MAE': np.mean(np.abs(clean_vals - noisy_vals)),
            'SNR (dB)': 10 * np.log10(np.var(clean_vals) / (np.var(noisy_vals - clean_vals) + 1e-10))
        })
    
    return pd.DataFrame(stats_data)

def create_stats_analysis_plot(df):
    """Create visualization for statistics metrics (SNR, SFM, Error)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. SNR
    axes[0].plot(df['Noise Level'], df['SNR (dB)'], marker='o', color='green', linewidth=2)
    axes[0].set_title('Signal-to-Noise Ratio (SNR)')
    axes[0].set_xlabel('Noise Level')
    axes[0].set_ylabel('SNR (dB)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_xlim(0, 1.0)

    # 2. SFM Comparison
    width = 0.05
    axes[1].bar(df['Noise Level'] - width/2, df['Clean SFM'], width, label='Clean SFM', color='blue', alpha=0.7)
    axes[1].bar(df['Noise Level'] + width/2, df['Noisy SFM'], width, label='Noisy SFM', color='red', alpha=0.7)
    axes[1].set_title('Spectral Flatness Measure (SFM)')
    axes[1].set_xlabel('Noise Level')
    axes[1].set_ylabel('SFM Score')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # 3. Error Metrics
    axes[2].plot(df['Noise Level'], df['MSE'], marker='s', label='MSE', color='orange', linewidth=2)
    axes[2].plot(df['Noise Level'], df['MAE'], marker='^', label='MAE', color='purple', linewidth=2)
    axes[2].set_title('Error Metrics (MSE/MAE)')
    axes[2].set_xlabel('Noise Level')
    axes[2].set_ylabel('Error Value')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def main():
    print("=" * 60)
    print("Visualizing SyntheticTS Datasets & Analyzing Statistics")
    print("=" * 60)
    
    # Check dataset existence
    available = []
    for noise_level in noise_levels:
        suffix = f"_noise{noise_level:.1f}"
        if os.path.exists(os.path.join(raw_data_dir, f"{dataset_name}{suffix}")):
            available.append(noise_level)
    
    if not available:
        print(f"Error: No datasets found in {raw_data_dir}")
        return
    
    output_dir = os.path.join(base_dir, 'scripts/data_preparation/SyntheticTS/visualizations')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput Directory: {output_dir}")
    
    # 1. Comparison Plot
    print("\n[1/4] Creating signal comparison plot...")
    fig1 = create_comparison_plot()
    fig1.savefig(os.path.join(output_dir, 'clean_vs_noisy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Overlay Plot
    print("[2/4] Creating signal overlay plot...")
    fig2 = create_overlay_plot()
    fig2.savefig(os.path.join(output_dir, 'clean_vs_noisy_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Compute Statistics
    print("[3/4] Computing statistics...")
    stats_df = create_statistics_table()
    stats_path = os.path.join(output_dir, 'statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved statistics to: {stats_path}")
    print("\nStatistics Summary:")
    print("-" * 80)
    print(stats_df.to_string(index=False))
    print("-" * 80)
    
    # 4. Stats Analysis Plot
    print("\n[4/4] Creating statistics analysis plot...")
    fig3 = create_stats_analysis_plot(stats_df)
    fig3.savefig(os.path.join(output_dir, 'noise_stats_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    print("\n" + "=" * 60)
    print("All visualizations completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()