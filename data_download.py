"""
ZTF Light Curve Data Downloader

This module provides multi-threaded parallel downloading of ZTF light curve
data from the NASA/IPAC Infrared Science Archive (IRSA).

Usage:
    Configure the File_Path (input catalog) and File_Path2 (output directory),
    then run: python data_download.py

Author: [Your Name]
Date: 2023
"""

import csv
import threading
import time
import wget
import pandas as pd
import astropy.coordinates as coord
import os
import math
import numpy as np

# Configuration
File_Path2 = "output"  # Output directory for downloaded light curves
File_Path = "catalog"  # Input catalog file (CSV format with 'ra' and 'dec' columns)

# Number of download threads
N_THREADS = 20


def download_worker(start_idx, end_idx, thread_id):
    """
    Worker function for downloading ZTF light curves.
    
    Args:
        start_idx: Starting index in the catalog
        end_idx: Ending index in the catalog
        thread_id: Thread identifier for logging
    """
    for i in range(start_idx, end_idx):
        coord_str = ccoord_str[i].replace(" ", "")
        try:
            output_file = f'./{File_Path2}/ZTFJ{coord_str}.csv'
            if not os.path.exists(output_file):
                wget.download(
                    f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?"
                    f"POS=CIRCLE+{tap_results.ra.iloc[i]}+{tap_results.dec.iloc[i]}"
                    f"+0.00083&COLLECTION=&FORMAT=csv",
                    out=output_file
                )
            print(f'ZTFJ{coord_str} - Thread {thread_id}')
        except Exception as e:
            # Log failed downloads but continue processing
            pass
        continue


def create_download_threads():
    """
    Create and start multiple download threads for parallel data acquisition.
    
    Returns:
        List of Thread objects
    """
    total_sources = len(cc.ra)
    chunk_size = math.ceil(total_sources / N_THREADS)
    
    threads = []
    for i in range(N_THREADS):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_sources)
        
        t = threading.Thread(
            target=download_worker,
            args=(start_idx, end_idx, i + 1)
        )
        threads.append(t)
    
    return threads


def main():
    """
    Main execution function: loads catalog and initiates parallel downloads.
    """
    global tap_results, cc, ccoord_str
    
    # Load source catalog
    tap_results = pd.read_csv(f'{File_Path}.csv')
    
    # Create SkyCoord objects from RA/Dec
    cc = coord.SkyCoord(
        tap_results.ra.iloc,
        tap_results.dec.iloc,
        unit='deg',
        frame='icrs'
    )
    
    # Generate coordinate strings for filename generation
    cstring = cc.to_string('hmsdms', sep=':', precision=2)
    ccoord_str = cc.to_string('hmsdms', sep="", precision=2)
    
    # Create output directory if it doesn't exist
    os.makedirs(File_Path2, exist_ok=True)
    
    # Create and start download threads
    threads = create_download_threads()
    
    print(f"Starting download of {len(cc.ra)} sources using {N_THREADS} threads...")
    start_time = time.time()
    
    for t in threads:
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    elapsed_time = time.time() - start_time
    print(f"\nDownload completed in {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    main()
