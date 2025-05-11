import logging
import os
from pathlib import Path
from joblib import Parallel, delayed
import pycatch22 as catch22
import scipy.signal
import pandas as pd
import numpy as np
from fooof import FOOOF

# def setup_logging(config):
#     """Set up logging configuration"""
#     # Create results directory if it doesn't exist
#     results_dir = Path(config['paths']['results'])
#     results_dir.mkdir(parents=True, exist_ok=True)
    
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(os.path.join(results_dir, 'pipeline.log')),
#             logging.StreamHandler()
#         ],
#         force=True
#     )

# def validate_paths(config):
    # """Validate existence of required paths"""
    # required_paths = ['base_data', 'results']
    # for path_key in required_paths:
    #     path = Path(config['paths'][path_key])
    #     if not path.exists() and path_key != 'results':
    #         raise FileNotFoundError(f"Required path {path_key} ({path}) does not exist")
    #     elif path_key == 'results':
    #         path.mkdir(parents=True, exist_ok=True)

def in_parallel(func, data, verbose=True):
    threads = os.cpu_count()

    if verbose:
        print(f"Processing {len(data)} items in parallel using {threads} threads")

    return Parallel(n_jobs=threads)(delayed(func)(item) for item in data)
    
def catch22_single_series(series):
        return catch22.catch22_all(series, catch24=True)

def compute_psd(data, fs):
    return scipy.signal.welch(data, fs=fs, window='hamming', nperseg=fs * 2, noverlap=fs, scaling='density')

def compute_psd_all_channels_parallel(data, fs):
    channel_groups = np.array_split(data.columns, os.cpu_count())
    def process_group(channels):
        freqs, psd_matrix = scipy.signal.welch(data[channels], fs, window='hamming', nperseg=fs*2, noverlap=fs, scaling='density', axis=0)
        return pd.DataFrame(psd_matrix, index = freqs, columns=channels)
    
    results = Parallel(n_jobs=-1)(delayed(process_group)(group) for group in channel_groups)

    return pd.concat(results, axis=1)

def fooof_single_series(args):
    freqs, psd_vals = args
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1, aperiodic_mode='knee')
    fm.fit(freqs, psd_vals, (1,40))
    return {
        'aperiodic_offset': fm.aperiodic_params_[0],
        'aperiodic_exponent': fm.aperiodic_params_[1],
        'r_squared': fm.r_squared_,
        'error': fm.error_,
        'num_peaks': fm.n_peaks_
    }

def compute_entropy_single_series(series):
    hist, bins = np.histogram(series, bins='auto', density=True)
    probabilities = hist / np.sum(hist)
    probabilities = probabilities[probabilities > 0]
    H = -np.sum(probabilities * np.log2(probabilities))
    return H
