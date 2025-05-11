#%%
import re
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial import distance
from typing import Union, List, Tuple
from scipy import signal
from pathlib import Path
from dotenv import load_dotenv
import os
from IPython import embed

#%%
class IEEGTools:
    def __init__(self):
        """
        Initialize the IEEGTools class
        """
        # Parameters for bad channel detection
        self.tile = 99
        self.mult = 10
        self.num_above = 1
        self.abs_thresh = 5e3
        self.percent_60_hz = 0.7
        self.mult_std = 10

    def clean_labels(self, channel_li):
        '''
        This function cleans a list of channels and returns the new channels
        '''
        new_channels = []
        keep_channels = np.ones(len(channel_li), dtype=bool)
        for i in channel_li:
            # standardizes channel names
            M = re.match(r"(\D+)(\d+)", i)

            # account for channels that don't have number e.g. "EKG", "Cz"
            if M is None:
                M = re.match(r"(\D+)", i)
                lead = M.group(1).replace("EEG", "").strip()
                contact = 0
            else:
                lead = M.group(1).replace("EEG", "").strip()
                contact = int(M.group(2))
     
            new_channels.append(f"{lead}{contact:02d}")

        return new_channels
    
    def automatic_bipolar_montage(self, ieeg_interictal: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a bipolar montage from the ieeg_interictal dataframe.
        
        For example, if we have channels:
        LA01, LA02, LA03, LB01, LB02, LB03
        
        It will create bipolar pairs:
        LA01-LA02, LA02-LA03
        LB01-LB02, LB02-LB03
        """
        # Get array of channel names
        channels = np.array(ieeg_interictal.columns)
        nchan = len(channels)
        dfBipolar = None
        
        # Loop through each channel except the last one
        for ch in range(nchan - 1):
            ch1 = channels[ch]
            
            # Parse the channel name into lead and contact number
            # Example: "LA01" -> lead="LA", contact=1
            M = re.match(r"(\D+)(\d+)", ch1)
            if M is None:
                # Handle special channels like "EKG" or "Cz"
                M = re.match(r"(\D+)", ch1)
                lead = M.group(1)
                contact = 0
            else:
                lead = M.group(1)      # e.g., "LA"
                contact = int(M.group(2))  # e.g., 1

            # Create the name of the next sequential contact
            # e.g., if ch1="LA01", then ch2="LA02"
            ch2 = lead + f"{(contact + 1):02d}"

            # If the next contact exists in our channel list
            if ch2 in channels:
                ch2Ind = np.where(channels == ch2)[0][0]
                # Create bipolar channel by subtracting ch2 from ch1
                bipolar = pd.Series(
                    (ieeg_interictal.iloc[:, ch] - ieeg_interictal.iloc[:, ch2Ind]),
                    name=ch1
                )
                
                # Add to our results DataFrame
                if dfBipolar is None:  # First bipolar pair
                    dfBipolar = pd.DataFrame(bipolar)
                else:  # Add additional bipolar pairs
                    dfBipolar = pd.concat([dfBipolar, pd.DataFrame(bipolar)], axis=1)
        
        # Return results or empty DataFrame if no bipolar pairs were found
        if dfBipolar is not None:
            return dfBipolar
        else:
            return pd.DataFrame()
            
    def filter_ieeg(self, ieeg_interictal: pd.DataFrame, sampling_rate: int, 
                    low_freq: float = 0.5, high_freq: float = 80, notch_freq: float = 60) -> pd.DataFrame:
        '''
        Apply filter the iEEG data:
        1. Applies a bandpass filter between 0.5Hz and 80Hz
        2. Applies a notch filter at 60Hz to remove power line noise
        
        Args:
            ieeg_interictal (pd.DataFrame): iEEG data with channel names as columns
            sampling_rate (int): Sampling rate of the data in Hz
            low_freq (float): Low cut-off frequency in Hz (default: 0.5)
            high_freq (float): High cut-off frequency in Hz (default: 80)
            notch_freq (float): Notch filter frequency in Hz (default: 60)
            
        Returns:
            pd.DataFrame: Filtered bipolar montage data
        '''
        
        # Convert DataFrame to numpy array for filtering (channels are columns)
        data = ieeg_interictal.values
        
        # Define filter parameters
        nyquist = sampling_rate / 2  # Nyquist frequency
        low_freq = low_freq / nyquist     # Low cut-off frequency (normalized)
        high_freq = high_freq / nyquist     # High cut-off frequency (normalized)
        notch_freq = notch_freq / nyquist    # Notch filter frequency (normalized)
        
        # Design bandpass filter - 4th order Butterworth filter
        b_bandpass, a_bandpass = signal.butter(4, [low_freq, high_freq], btype='bandpass')
        
        # Apply bandpass filter
        filtered_data = signal.filtfilt(b_bandpass, a_bandpass, data, axis=0)
        
        # Design notch filter - quality factor Q=30 (narrower is higher Q)
        b_notch, a_notch = signal.iirnotch(notch_freq, Q=30, fs=sampling_rate)
        
        # Apply notch filter
        filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data, axis=0)
        
        # Convert back to DataFrame with original column names
        ieeg_filtered = pd.DataFrame(filtered_data, columns=ieeg_interictal.columns)
        
        return ieeg_filtered
        
    def prctile(self, x: np.ndarray, p: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Calculate the percentile of the data, adjusting for sample size.
        
        Args:
            x (np.ndarray): Input data array
            p (float or array-like): Percentile(s) to compute
            
        Returns:
            np.ndarray: Computed percentile values
        """
        p = np.asarray(p, dtype=float)
        n = len(x)
        p = (p - 50) * n / (n - 1) + 50
        p = np.clip(p, 0, 100)
        return np.percentile(x, p)
        
    def identify_bad_channels(self, data: np.ndarray, fs: float) -> Tuple[np.ndarray, dict]:
        """
        Identify bad channels based on various criteria.

        Args:
            data (np.ndarray): Matrix representing iEEG data, of shape samples X channels
            fs (float): Sampling frequency of the iEEG data

        Returns:
            Tuple[np.ndarray, dict]: Boolean array of bad channels and details dictionary
        """
        nchs = data.shape[1]
        bad = []
        high_ch = []
        nan_ch = []
        zero_ch = []
        high_var_ch = []
        noisy_ch = []

        # Calculate statistics for all channels
        all_std = np.nanstd(data, axis=0)
        all_bl = np.nanmedian(data, axis=0)

        for ich in range(nchs):
            eeg = data[:, ich]

            # Check for NaNs
            if np.sum(np.isnan(eeg)) > 0.5 * len(eeg):
                bad.append(ich)
                nan_ch.append(ich)
                continue

            # Check for zeros
            if np.sum(eeg == 0) > 0.5 * len(eeg):
                bad.append(ich)
                zero_ch.append(ich)
                continue

            # Check for high amplitude
            if np.sum(np.abs(eeg - all_bl[ich]) > self.abs_thresh) > 10:
                bad.append(ich)
                high_ch.append(ich)
                continue

            # Check for high variance
            pct = self.prctile(eeg, [100 - self.tile, self.tile])
            thresh = [
                all_bl[ich] - self.mult * (all_bl[ich] - pct[0]),
                all_bl[ich] + self.mult * (pct[1] - all_bl[ich]),
            ]
            sum_outside = np.sum(eeg > thresh[1]) + np.sum(eeg < thresh[0])
            if sum_outside >= self.num_above:
                bad.append(ich)
                high_var_ch.append(ich)
                continue

            # Check for 60 Hz noise
            Y = np.fft.fft(eeg - np.nanmean(eeg))
            P = np.abs(Y) ** 2
            freqs = np.linspace(0, fs, len(P) + 1)[:-1]

            # Take first half of the spectrum
            P = P[: int(np.ceil(len(P) / 2))]
            freqs = freqs[: int(np.ceil(len(freqs) / 2))]

            # Calculate power in 60 Hz band
            P_60Hz = np.sum(P[(freqs > 58) & (freqs < 62)]) / np.sum(P)
            if P_60Hz > self.percent_60_hz:
                bad.append(ich)
                noisy_ch.append(ich)
                continue

        # Check for channels with high standard deviation
        median_std = np.nanmedian(all_std)
        higher_std = np.where(all_std > self.mult_std * median_std)[0]
        bad_std = [i for i in higher_std if i not in bad]
        bad.extend(bad_std)

        # Create binary mask
        bad_bin = np.zeros(nchs, dtype=bool)
        bad_bin[bad] = True

        details = {
            "noisy": noisy_ch,
            "nans": nan_ch,
            "zeros": zero_ch,
            "var": high_var_ch,
            "higher_std": bad_std,
            "high_voltage": high_ch
        }

        return bad_bin, details

    def channels_in_mask(self, ieeg_coords, subject_id, plot=False):
        '''
        This function returns the channels that are in the mask
        
        Parameters:
        -----------
        ieeg_coords : pandas DataFrame
            DataFrame containing electrode coordinates and ROI information
        mask_path : str
            Path to the mask file
        plot : bool, optional
            Whether to create a 3D visualization (default: False)
        '''
        # get mask path
        load_dotenv()
        BIDS_PATH = Path(os.getenv('BIDS_PATH'))
                
        mask_path = BIDS_PATH.joinpath(subject_id, 'derivatives', 'post_to_pre', 'surgerySeg_in_preT1.nii.gz')
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found for subject {subject_id}")
        mask = nib.load(mask_path)
        hdr = mask.header
        affine = mask.affine
        mask_data = np.asarray(mask.dataobj)
        # get coordinates of all non-zero voxels in the mask
        mask_coords = np.column_stack(np.where(mask_data != 0))
        mask_coords_mm = apply_affine(affine, mask_coords)
        mask_coords_mm = pd.DataFrame(mask_coords_mm, columns=['x', 'y', 'z'])

        dist = distance.cdist(ieeg_coords.loc[:, ['x','y','z']], mask_coords_mm, 'euclidean')
        dist = np.sum(dist < 5, axis=1) == 0
        ieeg_coords['spared'] = dist

        if plot:
            # plot 3d scatter of mask_coords_mm
            fig = px.scatter_3d(mask_coords_mm, x='x', y='y', z='z', opacity=0.5)
            # add ieeg_coords to the plot and color by binary in_mask
            fig.add_trace(go.Scatter3d(x=ieeg_coords['x'], 
                                    y=ieeg_coords['y'], 
                                    z=ieeg_coords['z'], 
                                    mode='markers',
                                    marker=dict(size=5, 
                                              color=ieeg_coords['spared'].astype(int),
                                              colorscale=[[0, 'red'], [1, 'blue']])))  # Custom binary colorscale
            fig.show()        
        return ieeg_coords

if __name__ == "__main__":
    pass

# %%