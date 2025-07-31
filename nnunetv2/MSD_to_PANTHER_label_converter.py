# Swaps integer labels 1 and 2 in NIfTI files.
import os
import nibabel as nib
import numpy as np
import multiprocessing
from functools import partial

def swap_labels_in_file(input_filename, input_folder, output_folder):
    """
    Swaps label values in a .nii.gz file.
    
    This function implements the following mapping:
    - Label 1 is converted to 2.
    - Label 2 is converted to 1.
    - All other labels (e.g., 0 for background) are left unchanged.

    Args:
        input_filename (str): The filename to process.
        input_folder (str): The path to the folder containing the original file.
        output_folder (str): The path to the folder where the relabeled file will be saved.
    """
    try:
        if not input_filename.endswith(".nii.gz"):
            return
        
        input_path = os.path.join(input_folder, input_filename)
        output_path = os.path.join(output_folder, input_filename)

        # Load the NIfTI file
        nii_image = nib.load(input_path)
        data = nii_image.get_fdata().astype(np.int16) # Ensure data is integer type

        # Create a copy of the data to modify. This preserves background (0) and other labels.
        new_data = data.copy()

        # --- Apply the swap logic ---
        # Find where the original data is 1, and set the new data's value to 2
        new_data[data == 1] = 2
        
        # Find where the original data is 2, and set the new data's value to 1
        new_data[data == 2] = 1

        # Create a new NIfTI image with the swapped data
        new_nii_image = nib.Nifti1Image(new_data, nii_image.affine, nii_image.header)

        # Save the new NIfTI image
        nib.save(new_nii_image, output_path)

        print(f"Successfully processed and saved: {output_path}")

    except Exception as e:
        print(f"Failed to process {input_filename}. Error: {e}")

def run_parallel_swapping(input_folder, output_folder, num_workers):
    """
    Finds all .nii.gz files and processes them in parallel.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Get a list of all files to process
    filenames = [f for f in os.listdir(input_folder) if f.endswith(".nii.gz")]
    
    if not filenames:
        print(f"No .nii.gz files found in {input_folder}. Exiting.")
        return

    # Create a processing pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Create a partial function to pass the fixed folder arguments
        process_func = partial(swap_labels_in_file, input_folder=input_folder, output_folder=output_folder)
        
        # Map the processing function to the list of filenames
        pool.map(process_func, filenames)


if __name__ == "__main__":
    # Ensure the script runs correctly with multiprocessing on all platforms
    multiprocessing.freeze_support()
    
    # --- PLEASE SPECIFY YOUR FOLDER PATHS HERE ---
    # The folder containing your .nii.gz files where label 1 needs to become 2, and 2 needs to become 1.
    # This might be the output of the previous script.
    input_directory = "../nnUNet_raw/Dataset999_PancreasPretrain/labelsTr"
    
    # The folder where the new, swapped .nii.gz files will be saved.
    output_directory = "../nnUNet_raw/Dataset999_PancreasPretrain/labelsTr_swapped"

    # --- CONFIGURE NUMBER OF WORKERS ---
    # Set the number of parallel processes. A good starting point is the
    # number of CPU cores you have.
    try:
        # Use almost all cores, leaving one free for system responsiveness
        num_workers = max(1, os.cpu_count() - 1)
        print(f"Using {num_workers} CPU workers.")
    except (NotImplementedError, TypeError):
        num_workers = 4 # Fallback value
        print("Could not determine CPU count. Defaulting to 4 workers.")

    run_parallel_swapping(input_directory, output_directory, num_workers)
    
    print("\nAll processing complete.")