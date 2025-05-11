#%%
import os
import pandas as pd
import numpy as np
import h5py
from process_ieeg import IEEGClipProcessor
from IPython import embed
from pathlib import Path
from features_univariate_utils import in_parallel, catch22_single_series, compute_psd_all_channels_parallel, fooof_single_series, compute_entropy_single_series
from fooof import FOOOF
import time

#%%
class UnivariateFeatures(IEEGClipProcessor):
    def __init__(self, subject_id: str):
        super().__init__()
        # self.project_root = Path(__file__).parent
        self.subject_id = subject_id
        # print(f"Script's project root: {self.project_root}")

        # Get base input and output directories from environment variables
        # Default to ./data/input and ./data/output if not set (for local non-Docker runs)
        input_base_dir = Path(os.environ.get('INPUT_DIR', 'data/input'))
        output_base_dir = Path(os.environ.get('OUTPUT_DIR', 'data/output'))

        # Construct full input path
        self.ieeg_processed  = input_base_dir.joinpath('interictal_ieeg_processed.h5')
        print(f"Attempting to load iEEG data from: {self.ieeg_processed}")

        # Construct full output path and ensure it exists
        self.output_dir = output_base_dir.joinpath('univariate_features', self.subject_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory set to: {self.output_dir}")

        self.ieeg_processed_bipolar, self.coordinates, self.sampling_frequency = self.load_ieeg()
        self.psd_df = compute_psd_all_channels_parallel(self.ieeg_processed_bipolar, self.sampling_frequency)

    def load_ieeg(self):
        if not self.ieeg_processed.exists():
            raise FileNotFoundError(f"No iEEG processed file found for subject {self.subject_id}")
        
        with h5py.File(self.ieeg_processed, 'r') as f:
                coordinates_data = f['/bipolar_montage/coordinates']
                ieeg_data = f['/bipolar_montage/ieeg']
                ieeg = pd.DataFrame(ieeg_data, columns=ieeg_data.attrs['channels_labels'])
                sampling_frequency = ieeg_data.attrs['sampling_rate']
                coordinates = pd.concat([
                    pd.DataFrame(coordinates_data, columns=['x', 'y', 'z'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['original_labels'], columns=['orig_labels'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['roi'], columns=['roi'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['roiNum'], columns=['roiNum'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['spared'], columns=['spared'], index=coordinates_data.attrs['labels'])
                ], axis=1)
        
        return ieeg, coordinates, sampling_frequency
        
    def catch22_features(self):
        channels = self.ieeg_processed_bipolar.columns
        time_series_list = [self.ieeg_processed_bipolar[channel].values for channel in channels]

        results = in_parallel(catch22_single_series, time_series_list, verbose=True)

        feature_names = results[0]['names']
        feature_values = np.array([result['values'] for result in results])
        results_df = pd.DataFrame(feature_values, index=channels, columns=feature_names)

        return results_df

    def fooof_features(self):
        channels = self.ieeg_processed_bipolar.columns
        freqs = self.psd_df.index.values

        inputs = [(freqs, self.psd_df[channel].values) for channel in channels]

        results = in_parallel(fooof_single_series, inputs, verbose=True)

        return pd.DataFrame(results, index=channels)

    def bandpower_features(self):
        bands = {
            'delta': [1, 4],
            'theta': [4, 8], 
            'alpha': [8, 13],
            'beta': [13, 30],
            'gamma': [30, 100]
        }
        
        results = {}
        freqs = self.psd_df.index.values
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            results[band_name] = self.psd_df.loc[mask].mean()
            
        return pd.DataFrame(results)

    def entropy_features(self):
        def compute_entropy_for_channel(channel):
            data = self.ieeg_processed_bipolar[channel].values
            result = compute_entropy_single_series(data)
            return (channel, result)
        
        results = in_parallel(compute_entropy_for_channel, self.ieeg_processed_bipolar.columns, verbose=True)
        # Convert list of tuples to a dictionary, then to a Pandas Series
        entropy_dict = dict(results)
        return pd.Series(entropy_dict, name='entropy')


#%%

if __name__ == "__main__":
    subject_id = "sub-RID0031"
    features_calculator = UnivariateFeatures(subject_id)

    # Define output paths
    output_dir = features_calculator.output_dir

    # Catch22 Features
    print("Calculating Catch22 Features...")
    catch22_df = features_calculator.catch22_features()
    catch22_output_path = output_dir.joinpath(f"{subject_id}_catch22_features.csv")
    catch22_df.to_csv(catch22_output_path)
    print(f"Catch22 Features saved to: {catch22_output_path}")
    
    # FOOOF Features
    print("\nCalculating FOOOF Features...")
    fooof_df = features_calculator.fooof_features()
    fooof_output_path = output_dir.joinpath(f"{subject_id}_fooof_features.csv")
    fooof_df.to_csv(fooof_output_path)
    print(f"FOOOF Features saved to: {fooof_output_path}")
    
    # Bandpower Features
    print("\nCalculating Bandpower Features...")
    bandpower_df = features_calculator.bandpower_features()
    bandpower_output_path = output_dir.joinpath(f"{subject_id}_bandpower_features.csv")
    bandpower_df.to_csv(bandpower_output_path)
    print(f"Bandpower Features saved to: {bandpower_output_path}")

    # Entropy Features
    print("\nCalculating Entropy Features...")
    entropy_series = features_calculator.entropy_features()
    entropy_output_path = output_dir.joinpath(f"{subject_id}_entropy_features.csv")
    entropy_series.to_csv(entropy_output_path)
    print(f"Entropy Features saved to: {entropy_output_path}")

    print("\nAll features calculated and saved.")

# %%
