# Converts PANORAMA dataset labels to include only healthy pancreas (2) and tumor (1) (PDAC in this case)
import os
import nibabel as nib
import numpy as np
import multiprocessing
from functools import partial

def remap_single_file(input_filename, input_folder, output_folder):
    """
    Remaps the labels of a single .nii.gz file based on the corrected scheme.

    The new remapping is:
    - 0 -> 0 (background)
    - 1 -> 2 (PDAC lesion to cancer)
    - 4, 5 -> 1 (Pancreas parenchyma, Pancreatic duct to pancreas)
    - 2, 3, 6 -> 0 (Veins, Arteries, Common bile duct to background)

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

        nii_image = nib.load(input_path)
        data = nii_image.get_fdata()

        # Create a new array initialized with zeros (background)
        new_data = np.zeros_like(data, dtype=np.int16)

        # Map Pancreas parenchyma (4) and Pancreatic duct (5) to pancreas (1)
        new_data[np.isin(data, [4, 5])] = 1
        # Map PDAC lesion (1) to cancer (2)
        new_data[data == 1] = 2
        # All other labels (2, 3, 6) are implicitly left as background (0) because the new_data array was initialized with zeros.

        new_nii_image = nib.Nifti1Image(new_data, nii_image.affine, nii_image.header)
        nib.save(new_nii_image, output_path)

        print(f"Successfully processed and saved: {output_path}")

    except Exception as e:
        print(f"Failed to process {input_filename}. Error: {e}")

def run_parallel_remapping(input_folder, output_folder, num_workers):
    """
    Finds all .nii.gz files and processes them in parallel.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    filenames = [f for f in os.listdir(input_folder) if f.endswith(".nii.gz")]
    
    # Create a processing pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Create a partial function to pass the fixed folder arguments
        process_func = partial(remap_single_file, input_folder=input_folder, output_folder=output_folder)
        
        # Map the processing function to the list of filenames
        pool.map(process_func, filenames)


if __name__ == "__main__":
    # Ensure the script runs correctly with multiprocessing on all platforms
    multiprocessing.freeze_support()
    
    # SPECIFY YOUR PATHS HERE
    # The folder containing your original .nii.gz files
    input_directory = "ENTER_YOUR_PATH/nnUNet_raw/Dataset147_PANORAMA/labelsTr"
    # The folder where the new .nii.gz files will be saved
    output_directory = "ENTER_YOUR_PATH/nnUNet_raw/Dataset147_PANORAMA/labelsTr_fixed"

    try:
        # Use almost all cores, leaving one free for system responsiveness
        num_workers = max(1, os.cpu_count() - 1)
        print(f"Using {num_workers} CPU workers.")
    except (NotImplementedError, TypeError):
        num_workers = 4 # Fallback value
        print("Could not determine CPU count. Defaulting to 4 workers.")


    run_parallel_remapping(input_directory, output_directory, num_workers)
    
    print("\nAll processing complete.")
