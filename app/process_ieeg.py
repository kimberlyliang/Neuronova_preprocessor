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
from dotenv import load_dotenv
import os
import logging
import time
from datetime import datetime

#%%
class IEEGClipProcessor(IEEGTools):
    def __init__(self):
        super().__init__()
        self.project_root = Path(__file__).parent.parent
        self.input_dir = Path(os.getenv('INPUT_DIR', '/data/input'))
        print(f"Initialized with input directory: {self.input_dir}")

    def find_subject_files(self, subject_id: str) -> List[Tuple[Path, Path, Path]]:
        """Find all iEEG files and their corresponding recon files for a subject.
        
        Args:
            subject_id (str): Subject ID to find files for
            
        Returns:
            List[Tuple[Path, Path, Path]]: List of tuples containing (ieeg_file_path, ieeg_recon_path, ieeg_recon_mni_path)
        """
        print(f"Recursively searching for files in: {self.input_dir / subject_id}")
        subject_dir = self.input_dir / subject_id
        
        # Find all iEEG files recursively
        ieeg_files = list(subject_dir.rglob('interictal_ieeg_*.h5'))
        if not ieeg_files:
            raise FileNotFoundError(f"No iEEG files found for subject {subject_id}")
            
        # Find recon files recursively
        try:
            recon_file = next(subject_dir.rglob('*electrodes2ROI.csv'))
            print(f"Found recon file: {recon_file}")
        except StopIteration:
            raise FileNotFoundError(f"No electrode reconstruction file found for subject {subject_id}")
            
        try:
            recon_mni_file = next(subject_dir.rglob('*electrodes2ROI_mni152_corrected.csv'))
            print(f"Found MNI recon file: {recon_mni_file}")
        except StopIteration:
            raise FileNotFoundError(f"No MNI electrode reconstruction file found for subject {subject_id}")
            
        # Return list of tuples, one for each iEEG file
        return [(ieeg_file, recon_file, recon_mni_file) for ieeg_file in ieeg_files]
    
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

    def process_raw_ieeg(self, subject_id: str, plotEEG: bool = False, saveEEG: bool = False, ieeg_files: List[Tuple[Path, Path, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process iEEG data for a subject from raw to filtered data with electrode information.
        
        Args:
            subject_id (str): Subject ID to load data for
            plotEEG (bool, optional): Whether to plot the EEG data. Defaults to False.
            saveEEG (bool, optional): Whether to save the processed data. Defaults to False.
            ieeg_files (List[Tuple[Path, Path, Path]], optional): List of iEEG files to process. Defaults to None.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Filtered iEEG data and electrode information
        """
        print("\n=== Starting process_raw_ieeg ===")
        # Step 1: Find the files
        print("Step 1: Finding files...")
        if ieeg_files is None:
            ieeg_files = self.find_subject_files(subject_id)
        print(f"Found {len(ieeg_files)} iEEG files, processing first one:")
        ieeg_file_path, ieeg_recon_path, ieeg_recon_mni_path = ieeg_files[0]
        print(f"  iEEG: {ieeg_file_path}")
        print(f"  Recon: {ieeg_recon_path}")
        print(f"  MNI Recon: {ieeg_recon_mni_path}")
        
        # Step 2: Load the iEEG clips
        print("\nStep 2: Loading iEEG clips...")
        ieeg_data, sampling_rate = self.load_ieeg_clips(ieeg_file_path)
        print(f"Loaded iEEG data shape: {ieeg_data.shape}, sampling rate: {sampling_rate}")
        
        # Step 3: Prepare electrodes and iEEG data
        print("\nStep 3: Preparing electrodes and iEEG data...")
        ieeg_data, electrodes2ROI = self.prepare_electrodes_and_ieeg(ieeg_data, ieeg_recon_path)
        print(f"Prepared data shapes - iEEG: {ieeg_data.shape}, electrodes: {electrodes2ROI.shape}")
        
        # Step 4: Remove bad channels
        print("\nStep 4: Removing bad channels...")
        ieeg_data, electrodes2ROI = self.remove_bad_channels(ieeg_data, electrodes2ROI, sampling_rate)
        print(f"After removing bad channels - iEEG: {ieeg_data.shape}, electrodes: {electrodes2ROI.shape}")
        
        # Step 5: Process the iEEG signal (bipolar montage and filtering)
        print("\nStep 5: Processing iEEG signal...")
        ieeg_filtered = self.process_ieeg_signal(ieeg_data, sampling_rate)
        print(f"Processed iEEG shape: {ieeg_filtered.shape}")
        
        # Step 6: Finalize the electrodes data
        print("\nStep 6: Finalizing electrodes data...")
        electrodes2ROI = self.finalize_electrodes(electrodes2ROI, ieeg_filtered, subject_id)
        print(f"Finalized electrodes shape: {electrodes2ROI.shape}")
        
        # Step 7: Final alignment of iEEG and electrodes
        print("\nStep 7: Final alignment...")
        ieeg_filtered = ieeg_filtered.loc[:, electrodes2ROI.index]
        print(f"Aligned iEEG shape: {ieeg_filtered.shape}")

        # Step 8: Sort data by channel labels and columns
        print("\nStep 8: Sorting data...")
        electrodes2ROI = electrodes2ROI.sort_index()
        ieeg_filtered = ieeg_filtered.sort_index(axis=1)
        
        # Verify alignment
        if not np.array_equal(electrodes2ROI.index, ieeg_filtered.columns):
            raise ValueError(f"Electrodes2ROI and ieeg_filtered do not have the same channels for subject {subject_id}")
        
        # Optional: Plot the EEG data
        if plotEEG:
            self.plot_eeg_data(ieeg_filtered, sampling_rate)

        if saveEEG:
            print("\nSaving processed data...")
            self.save_ieeg_processed(ieeg_filtered, sampling_rate, electrodes2ROI, subject_id, ieeg_file_path)

        return ieeg_filtered, electrodes2ROI
    
    def save_ieeg_processed(self, ieeg_filtered: pd.DataFrame, sampling_rate: float, electrodes2ROI: pd.DataFrame, subject_id: str, ieeg_file_path: Path) -> None:
        """Save the processed iEEG data and electrode information to a CSV file.
        
        Args:
            ieeg_filtered (pd.DataFrame): Filtered iEEG data
            sampling_rate (float): Sampling rate of the data
            electrodes2ROI (pd.DataFrame): Electrode information
            subject_id (str): Subject ID
            ieeg_file_path (Path): Path to the original iEEG file
        """
        print("\n=== Starting save_ieeg_processed ===")
        print(f"Input parameters:")
        print(f"  ieeg_filtered shape: {ieeg_filtered.shape}")
        print(f"  sampling_rate: {sampling_rate}")
        print(f"  electrodes2ROI shape: {electrodes2ROI.shape}")
        print(f"  subject_id: {subject_id}")
        print(f"  ieeg_file_path: {ieeg_file_path}")
        
        # Get output path from environment variable, fallback to data/output
        output_base = Path(os.getenv('OUTPUT_DIR', 'data/output'))
        print(f"Using output base path: {output_base}")
        
        # Create the same directory structure as input
        destination_path = output_base / subject_id / 'derivatives' / 'processed'
        print(f"Creating output directory: {destination_path}")
        destination_path.mkdir(parents=True, exist_ok=True)
        
        h5_file_path = destination_path / f"{ieeg_file_path.stem}_processed.h5"
        print(f"Will save to: {h5_file_path}")
        
        # Check if file exists and handle accordingly
        if h5_file_path.exists():
            print(f"File already exists at {h5_file_path}. Will overwrite.")
        
        # Calculate optimal chunk size for ieeg data (time × channels)
        n_samples, n_channels = ieeg_filtered.shape
        chunk_size = (min(10000, n_samples), min(n_channels, 32))
        print(f"Using chunk size: {chunk_size}")
        
        try:
            print("Opening H5 file for writing...")
            with h5py.File(h5_file_path, 'w') as f:
                print("Creating bipolar_montage group...")
                subj_group = f.create_group('bipolar_montage')
                
                print("Saving iEEG data...")
                ieeg_h5 = subj_group.create_dataset('ieeg', 
                                          data=ieeg_filtered.values.astype(np.float32),
                                          dtype='float32', 
                                          compression='gzip',
                                          compression_opts=4,
                                          chunks=chunk_size)
                
                print("Adding iEEG metadata...")
                ieeg_h5.attrs['sampling_rate'] = sampling_rate
                ieeg_h5.attrs['channels_labels'] = ieeg_filtered.columns.tolist()
                ieeg_h5.attrs['shape'] = ieeg_filtered.shape
                ieeg_h5.attrs['raw_data_file'] = str(ieeg_file_path.name)
                ieeg_h5.attrs['subject_id'] = subject_id
                
                print("Saving electrode data...")
                coords_data = electrodes2ROI[['x', 'y', 'z']].values.astype(np.float32)
                native_coord_mm = subj_group.create_dataset('coordinates', 
                                                    data=coords_data,
                                                    dtype='float32', 
                                                    compression='gzip')
                
                print("Adding electrode metadata...")
                native_coord_mm.attrs['labels'] = electrodes2ROI.index.tolist()
                native_coord_mm.attrs['original_labels'] = electrodes2ROI['labels'].tolist()
                native_coord_mm.attrs['roi'] = electrodes2ROI['roi'].tolist()
                native_coord_mm.attrs['roiNum'] = electrodes2ROI['roiNum'].tolist()
                native_coord_mm.attrs['spared'] = electrodes2ROI['spared'].tolist()
                
            print(f"Successfully saved processed iEEG data for {subject_id} to {h5_file_path}")
        except Exception as e:
            print(f"Error saving data for {subject_id}: {str(e)}")
            raise
        print("=== Completed save_ieeg_processed ===\n")

# Define the function outside the if __name__ == "__main__" block
def process_subject(subject_id, h5_file_path=None):
    try:
        print(f"Processing {subject_id}...")
        ieeg = IEEGClipProcessor()
        if h5_file_path:
            # If we already have the H5 file path, find the recon files in the subject directory
            subject_dir = h5_file_path.parent.parent  # Go up two levels to get subject directory
            try:
                recon_file = next(subject_dir.rglob('*electrodes2ROI.csv'))
                recon_mni_file = next(subject_dir.rglob('*electrodes2ROI_mni152_corrected.csv'))
                ieeg_files = [(h5_file_path, recon_file, recon_mni_file)]
            except StopIteration as e:
                raise FileNotFoundError(f"Could not find recon files for subject {subject_id}: {str(e)}")
        else:
            # Otherwise, find all files for this subject
            ieeg_files = ieeg.find_subject_files(subject_id)
        print(f"Found files for {subject_id}")
        ieeg_filtered, electrodes2ROI = ieeg.process_raw_ieeg(subject_id, saveEEG=True, ieeg_files=ieeg_files)
        print(f"Completed processing {subject_id}")
        return subject_id, True
    except Exception as e:
        print(f"Error processing {subject_id}: {str(e)}")
        return subject_id, False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting iEEG processing...")
    start_time = time.time()
    
    # Get base input directory from environment variable or default
    input_base_dir = Path(os.environ.get('INPUT_DIR', 'data/input'))
    output_base_dir = Path(os.environ.get('OUTPUT_DIR', 'data/output'))
    logger.info(f"Input directory: {input_base_dir}")
    logger.info(f"Output directory: {output_base_dir}")
    
    try:
        # Find all H5 files in the input directory
        possible_filenames = [
            'interictal_ieeg_processed.h5',
            'interictal_ieeg_wake_processed.h5',
            'interictal_ieeg_day*.h5',  # For files like interictal_ieeg_day9.h5
            'interictal_ieeg_*.h5'      # Catch-all for any other interictal files
        ]
        h5_files = []
        
        # Search for all H5 files globally in input directory
        for filename in possible_filenames:
            found_files = list(input_base_dir.rglob(filename))
            if found_files:
                logger.info(f"Found {len(found_files)} files matching {filename}")
                h5_files.extend(found_files)
        
        # Remove duplicates (in case a file matches multiple patterns)
        h5_files = list(set(h5_files))
        
        if not h5_files:
            raise FileNotFoundError(f"No H5 files found in {input_base_dir} or its subdirectories")
        
        logger.info(f"Found {len(h5_files)} unique H5 files to process")
        
        # Process each H5 file
        results = []
        for h5_file in h5_files:
            logger.info(f"\nProcessing file: {h5_file}")
            try:
                # Extract subject ID from the file path
                subject_id = h5_file.parent.parent.name  # Go up two levels to get subject ID
                logger.info(f"Processing subject: {subject_id}")
                
                # Process the subject with the known H5 file path
                result = process_subject(subject_id, h5_file_path=h5_file)
                results.append(result)
                
                logger.info(f"Completed processing for {subject_id}")
            except Exception as e:
                logger.error(f"Error processing {h5_file}: {str(e)}", exc_info=True)
                logger.info("Continuing with next file...")
                results.append((subject_id, False))
                continue
        
        # Print summary
        end_time = time.time()
        logger.info(f"\nProcessing Summary (Total time: {end_time - start_time:.2f} seconds):")
        
        successful = [s for s, success in results if success]
        failed = [s for s, success in results if not success]
        
        logger.info(f"\nSuccessfully processed ({len(successful)}):")
        for subject in successful:
            logger.info(f"{subject}")
            
        if failed:
            logger.info(f"\nFailed to process ({len(failed)}):")
            for subject in failed:
                logger.info(f"{subject}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        raise

# %%
