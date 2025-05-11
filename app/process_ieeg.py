#%%
import pandas as pd
import h5py
import re
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from typing import Union, List, Tuple
import mne
from IPython import embed
from scipy import signal
from process_ieeg_utils import IEEGTools

#%%
class IEEGClipProcessor(IEEGTools):
    def __init__(self):
        super().__init__()
        self.project_root = Path(__file__).parent.parent

    def find_subject_files(self, subject_id: str) -> Tuple[Path, Path]:
        """Find H5 and electrode reconstruction files for a subject.
        
        Args:
            subject_id (str): Subject ID to find files for
            
        Returns:
            Tuple[Path, Path]: Paths to iEEG file and electrode reconstruction file
        """
        try:
            ieeg_file_path = next(self.project_root.joinpath('data', 'source', 'BIDS').rglob(f'{subject_id}/**/interictal_ieeg*.h5'))
            ieeg_recon_path = next(self.project_root.joinpath('data', 'source', 'BIDS').rglob(f'{subject_id}/**/*electrodes2ROI.csv'))
            ieeg_recon_mni_path = next(self.project_root.joinpath('data', 'source', 'BIDS').rglob(f'{subject_id}/**/*electrodes2ROI_mni152_corrected.csv'))
            self.ieeg_file_path = ieeg_file_path
            self.ieeg_recon_path = ieeg_recon_path
            self.ieeg_recon_mni_path = ieeg_recon_mni_path
            return ieeg_file_path, ieeg_recon_path, ieeg_recon_mni_path
        except StopIteration:
            raise FileNotFoundError(f"No iEEG clips found for subject {subject_id}")
    
    def load_ieeg_clips(self, ieeg_file_path: Path) -> Tuple[pd.DataFrame, float]:
        """Load all iEEG clips from an H5 file into a single DataFrame.
        
        Args:
            ieeg_file_path (Path): Path to H5 file containing iEEG clips
            
        Returns:
            Tuple[pd.DataFrame, float]: DataFrame with all clips and sampling rate
        """
        ieeg = pd.DataFrame()
        sampling_rate = None
        
        with h5py.File(ieeg_file_path, 'r') as f:
            all_clips = list(f.keys())
            for clip_id in all_clips:
                clip = f[clip_id]
                sampling_rate = clip.attrs.get('sampling_rate')
                ieeg_clip = pd.DataFrame(clip, columns=clip.attrs.get('channels_labels'))
                ieeg = pd.concat([ieeg, ieeg_clip], axis=0)
        
        return ieeg.reset_index(drop=True), sampling_rate
    
    def prepare_electrodes_and_ieeg(self, ieeg_data: pd.DataFrame, electrodes_file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean labels and find common channels between iEEG and electrode reconstruction.
        
        Args:
            ieeg_data (pd.DataFrame): DataFrame containing iEEG data
            electrodes_file_path (Path): Path to electrode reconstruction file
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Filtered iEEG data and electrode reconstruction
        """
        # Load electrodes data
        electrodes2ROI = pd.read_csv(electrodes_file_path).set_index('labels')
        
        # Clean labels
        ieeg_data.columns = self.clean_labels(ieeg_data.columns)
        electrodes2ROI['clean_labels'] = self.clean_labels(electrodes2ROI.index)
        
        # Find common channels
        keep_channels = list(set(ieeg_data.columns) & set(electrodes2ROI['clean_labels']))
        
        if not keep_channels:
            raise ValueError(f"No common channels found between ieeg_data and electrodes2ROI")
        
        # Filter to common channels
        electrodes2ROI = electrodes2ROI[electrodes2ROI['clean_labels'].isin(keep_channels)]
        electrodes2ROI = electrodes2ROI.reset_index().set_index('clean_labels')
        ieeg_data = ieeg_data.loc[:, keep_channels]
        
        # Reorder electrodes to match ieeg_data
        electrodes2ROI = electrodes2ROI.loc[ieeg_data.columns]
        
        return ieeg_data, electrodes2ROI
    
    def remove_bad_channels(self, ieeg_data: pd.DataFrame, electrodes2ROI: pd.DataFrame, 
                          sampling_rate: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Identify and remove bad channels.
        
        Args:
            ieeg_data (pd.DataFrame): DataFrame containing iEEG data
            electrodes2ROI (pd.DataFrame): DataFrame with electrode information
            sampling_rate (float): Sampling rate of the iEEG data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: iEEG and electrodes with bad channels removed
        """
        # Identify bad channels
        bad_channels, details = self.identify_bad_channels(ieeg_data.values, sampling_rate)
        
        # Remove bad channels
        good_channels = ~bad_channels
        ieeg_data = ieeg_data.iloc[:, good_channels]
        electrodes2ROI = electrodes2ROI.iloc[good_channels]
        
        return ieeg_data, electrodes2ROI
    
    def process_ieeg_signal(self, ieeg_data: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
        """Apply bipolar montage and filter the iEEG data.
        
        Args:
            ieeg_data (pd.DataFrame): DataFrame containing iEEG data
            sampling_rate (float): Sampling rate of the iEEG data
            
        Returns:
            pd.DataFrame: Processed iEEG data
        """
        ieeg_bipolar = self.automatic_bipolar_montage(ieeg_data)
        ieeg_filtered = self.filter_ieeg(ieeg_interictal=ieeg_bipolar, sampling_rate=sampling_rate)
        
        return ieeg_filtered
    
    def finalize_electrodes(self, electrodes2ROI: pd.DataFrame, ieeg_filtered: pd.DataFrame, 
                           subject_id: str) -> pd.DataFrame:
        """Finalize electrode data, removing channels outside brain.
        
        Args:
            electrodes2ROI (pd.DataFrame): DataFrame with electrode information
            ieeg_filtered (pd.DataFrame): Filtered iEEG data
            subject_id (str): Subject ID
            
        Returns:
            pd.DataFrame: Finalized electrode information
        """
        # Remove channels not in ieeg_filtered
        electrodes2ROI = electrodes2ROI[electrodes2ROI.index.isin(ieeg_filtered.columns)]
        
        # Remove channels outside brain but keep white-matter
        electrodes2ROI = electrodes2ROI[electrodes2ROI['roi'] != 'outside-brain']
        
        # Select and rename columns
        electrodes2ROI = electrodes2ROI.filter(['labels','mm_x', 'mm_y', 'mm_z', 'roi', 'roiNum'])\
                                      .rename(columns={'mm_x': 'x', 'mm_y': 'y', 'mm_z': 'z'})
        
        # Apply mask
        electrodes2ROI = self.channels_in_mask(ieeg_coords=electrodes2ROI, subject_id=subject_id)
        
        return electrodes2ROI
    
    def plot_eeg_data(self, ieeg_data: pd.DataFrame, sampling_rate: float) -> None:
        """Plot the EEG data using MNE.
        
        Args:
            ieeg_data (pd.DataFrame): DataFrame containing iEEG data
            sampling_rate (float): Sampling rate of the iEEG data
        """
        # Create MNE info object
        labels = list(ieeg_data.columns)
        info = mne.create_info(ch_names=labels, sfreq=sampling_rate, ch_types=['eeg'] * len(labels))
        
        # Make ieeg_data to mne raw object
        ieeg_mne = mne.io.RawArray(ieeg_data.values.T, info)
        
        # Plot with interactive settings
        fig = ieeg_mne.plot(
            scalings='auto',
            n_channels=len(labels),
            title='EEG Recording\n'
                  '(Use +/- keys to scale, = to reset)\n'
                  '(Click & drag to select area, arrow keys to navigate)',
            show=True,
            block=False,
            duration=10,
            start=0
        )

    def process_raw_ieeg(self, subject_id: str, plotEEG: bool = False, saveEEG: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process iEEG data for a subject from raw to filtered data with electrode information.
        
        Args:
            subject_id (str): Subject ID to load data for
            plotEEG (bool, optional): Whether to plot the EEG data. Defaults to False.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Filtered iEEG data and electrode information
        """
        # Step 1: Find the files
        ieeg_file_path, ieeg_recon_path = self.find_subject_files(subject_id)
        
        # Step 2: Load the iEEG clips
        ieeg_data, sampling_rate = self.load_ieeg_clips(ieeg_file_path)
        
        # Step 3: Prepare electrodes and iEEG data
        ieeg_data, electrodes2ROI = self.prepare_electrodes_and_ieeg(ieeg_data, ieeg_recon_path)
        
        # Step 4: Remove bad channels
        ieeg_data, electrodes2ROI = self.remove_bad_channels(ieeg_data, electrodes2ROI, sampling_rate)
        
        # Step 5: Process the iEEG signal (bipolar montage and filtering)
        ieeg_filtered = self.process_ieeg_signal(ieeg_data, sampling_rate)
        
        # Step 6: Finalize the electrodes data
        electrodes2ROI = self.finalize_electrodes(electrodes2ROI, ieeg_filtered, subject_id)
        
        # Step 7: Final alignment of iEEG and electrodes
        ieeg_filtered = ieeg_filtered.loc[:, electrodes2ROI.index]

        # Step 8: Sort data by channel labels and columns
        electrodes2ROI = electrodes2ROI.sort_index()
        ieeg_filtered = ieeg_filtered.sort_index(axis=1)
        
        # Verify alignment
        if not np.array_equal(electrodes2ROI.index, ieeg_filtered.columns):
            raise ValueError(f"Electrodes2ROI and ieeg_filtered do not have the same channels for subject {subject_id}")
        
        # Optional: Plot the EEG data
        if plotEEG:
            self.plot_eeg_data(ieeg_filtered, sampling_rate)

        if saveEEG:
            self.save_ieeg_processed(ieeg_filtered, sampling_rate, electrodes2ROI, subject_id)

        return ieeg_filtered, electrodes2ROI
    
    def save_ieeg_processed(self, ieeg_filtered: pd.DataFrame, sampling_rate: float, electrodes2ROI: pd.DataFrame, subject_id: str) -> None:
        """Save the processed iEEG data and electrode information to a CSV file.
        
        Args:
            ieeg_filtered (pd.DataFrame): Filtered iEEG data
            sampling_rate (float): Sampling rate of the data
            electrodes2ROI (pd.DataFrame): Electrode information
            subject_id (str): Subject ID
        """
        # Replace 'source' with 'derivatives' in the path
        file_path_parts = list(self.ieeg_file_path.parts)
        source_index = file_path_parts.index('source')
        file_path_parts[source_index] = 'derivatives'
        
        # Create the derivatives directory based on the original path structure
        destination_path = Path(*file_path_parts[:-1])  # Remove the file name and its parent directory
        destination_path.mkdir(parents=True, exist_ok=True)
        h5_file_path = destination_path / 'interictal_ieeg_processed.h5'
        
        # Check if file exists and handle accordingly
        if h5_file_path.exists():
            print(f"File already exists at {h5_file_path}. Will overwrite.")
        
        # Calculate optimal chunk size for ieeg data (time × channels)
        # Assuming most access will be by time segments
        n_samples, n_channels = ieeg_filtered.shape
        chunk_size = (min(10000, n_samples), min(n_channels, 32))
        
        try:
            with h5py.File(h5_file_path, 'w') as f:
                # Create a group for this subject
                subj_group = f.create_group('bipolar_montage')
                
                # Save iEEG data as float32 to save space
                ieeg_h5 = subj_group.create_dataset('ieeg', 
                                          data=ieeg_filtered.values.astype(np.float32),
                                          dtype='float32', 
                                          compression='gzip',
                                          compression_opts=4,  # Balance between speed and compression
                                          chunks=chunk_size)   # Optimize for time-series access
                
                # Add metadata as attributes
                ieeg_h5.attrs['sampling_rate'] = sampling_rate
                ieeg_h5.attrs['channels_labels'] = ieeg_filtered.columns.tolist()
                ieeg_h5.attrs['shape'] = ieeg_filtered.shape
                ieeg_h5.attrs['raw_data_file'] = self.ieeg_file_path.name
                ieeg_h5.attrs['subject_id'] = subject_id
                
                # Save electrode data
                coords_data = electrodes2ROI[['x', 'y', 'z']].values.astype(np.float32)
                native_coord_mm = subj_group.create_dataset('coordinates', 
                                                    data=coords_data,
                                                    dtype='float32', 
                                                    compression='gzip')
                
                # Add electrode metadata
                native_coord_mm.attrs['labels'] = electrodes2ROI.index.tolist()
                native_coord_mm.attrs['original_labels'] = electrodes2ROI['labels'].tolist()
                native_coord_mm.attrs['roi'] = electrodes2ROI['roi'].tolist()
                native_coord_mm.attrs['roiNum'] = electrodes2ROI['roiNum'].tolist()
                native_coord_mm.attrs['spared'] = electrodes2ROI['spared'].tolist()
                
            print(f"Successfully saved processed iEEG data for {subject_id} to {h5_file_path}")
        except Exception as e:
            print(f"Error saving data for {subject_id}: {str(e)}")

# Define the function outside the if __name__ == "__main__" block
def process_subject(subject_id):
    try:
        print(f"Processing {subject_id}...")
        ieeg = IEEGClipProcessor()
        ieeg_filtered, electrodes2ROI = ieeg.process_raw_ieeg(subject_id, saveEEG=True)
        print(f"Completed processing {subject_id}")
        return subject_id, True
    except Exception as e:
        print(f"Error processing {subject_id}: {str(e)}")
        return subject_id, False

if __name__ == "__main__":

    # subjects_to_find = [
    #     'sub-RID0031', 'sub-RID0032', 'sub-RID0033', 'sub-RID0050', 'sub-RID0051',
    #     'sub-RID0064', 'sub-RID0089', 'sub-RID0101', 'sub-RID0117', 'sub-RID0143',
    #     'sub-RID0167', 'sub-RID0175', 'sub-RID0179', 'sub-RID0190', 'sub-RID0193',
    #     'sub-RID0222', 'sub-RID0238', 'sub-RID0267', 'sub-RID0301', 'sub-RID0320',
    #     'sub-RID0322', 'sub-RID0332', 'sub-RID0381', 'sub-RID0405', 'sub-RID0412',
    #     'sub-RID0424', 'sub-RID0508', 'sub-RID0562', 'sub-RID0589', 'sub-RID0595',
    #     'sub-RID0621', 'sub-RID0658', 'sub-RID0675', 'sub-RID0679', 'sub-RID0700',
    #     'sub-RID0785', 'sub-RID0796', 'sub-RID0852', 'sub-RID0883', 'sub-RID0893',
    #     'sub-RID0941', 'sub-RID0967'
    # ]
    
    # Single subject test - uncomment to test one subject first
    process_subject('sub-RID0031')
    
    # Run parallel processing
    # print(f"Starting parallel processing for {len(subjects_to_find)} subjects")
    # with Pool() as pool:
    #     results = pool.map(process_subject, subjects_to_find)

# %%
