import SimpleITK as sitk
import sys
import os
from pathlib import Path

def convert_file(input_path, output_path):
    """
    Converts a single .mha file to .nii.gz format.
    
    Args:
        input_path (str): The full path to the input .mha file.
        output_path (str): The full path for the output .nii.gz file.
    """
    try:
        # Read the .mha image
        image = sitk.ReadImage(str(input_path))
        
        sitk.WriteImage(image, str(output_path))
        
        return True
    except Exception as e:
        print(f"  Error converting {input_path.name}: {e}")
        return False


def batch_convert_mha_to_nii_gz(input_folder, output_folder):
    """
    Converts all .mha files in a folder to .nii.gz format.

    Args:
        input_folder (str): Path to the folder containing .mha files.
        output_folder (str): Path to the folder where .nii.gz files will be saved.
    """
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    if not input_dir.is_dir():
        print(f"Error: Input folder not found at '{input_dir}'")
        return

    print(f"Output will be saved to: '{output_dir}'")
    output_dir.mkdir(parents=True, exist_ok=True)

    mha_files = sorted(list(input_dir.glob('*.mha')))
    
    if not mha_files:
        print(f"No .mha files found in '{input_dir}'.")
        return

    print(f"\nFound {len(mha_files)} .mha files to convert.")
    
    success_count = 0
    for i, mha_file_path in enumerate(mha_files):
        print(f"Processing file {i+1}/{len(mha_files)}: {mha_file_path.name}")
        
        # Create the full output file path
        output_filename = mha_file_path.stem + '.nii.gz'
        output_file_path = output_dir / output_filename
        
        if convert_file(mha_file_path, output_file_path):
            success_count += 1
            
    print("\n-------------------------------------")
    print("         Conversion Summary          ")
    print("-------------------------------------")
    print(f"Successfully converted {success_count} of {len(mha_files)} files.")
    print(f"Converted files are located in: {output_dir}")
    print("-------------------------------------")


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("\n--- MHA to NII.GZ Batch Converter ---")
        print("Converts all .mha files in a source folder to .nii.gz in a destination folder.")
        print("\nUsage: python convert_folder_mha_to_nii.py <input_folder> <output_folder>")
        print("Example: python convert_folder_mha_to_nii.py ./mha_scans ./converted_nii_scans")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    batch_convert_mha_to_nii_gz(input_folder, output_folder)