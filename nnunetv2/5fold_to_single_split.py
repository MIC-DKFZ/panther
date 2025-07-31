import numpy as np
import json
from pathlib import Path

# Define the path to your GROUND TRUTH segmentations
# This is the correct place to get the definitive list of cases.
gt_folder = Path("/home/o644l/remote_files/nnunet_preprocessed_folder/Dataset906_PANTHER_COMBINED_onlyTumor/gt_segmentations")

# --- FIX 1: Convert the generator to a list immediately ---
# This gets all Path objects for your ground truth files.
all_path_objects = list(gt_folder.glob("*.nii.gz"))

# --- FIX 2 & 4: Get a clean list of CASE IDENTIFIERS (strings), not Path objects ---
# The .stem attribute gives you the filename without the '.nii.gz' part.
all_case_ids = sorted([p.stem for p in all_path_objects])

print(f"Found {len(all_case_ids)} total case identifiers.")
print(f"First 3 identifiers: {all_case_ids[:3]}")

# --- FIX 3 & 5: Separate Task 1 and Task 2 IDs using robust logic ---
# You need to define how to tell a Task 1 case from a Task 2 case from its name.
# For this example, I will assume Task 1 cases are named 'panther_xxx' and Task 2 are 'flare_xxx'
# !!! YOU MUST ADAPT THIS LOGIC TO YOUR FILENAMING CONVENTION !!!
# For example, if your Task 1 files are named 'case_001_..._0000' and Task 2 are 'case_101_..._0000'
# you could check if int(case_id.split('_')[1]) < 100.
# A prefix is the most common and robust way.

# --- !! ADAPT THIS SECTION !! ---
TASK1_PREFIX = "panther" # Example prefix for Task 1 cases
task1_ids = [case_id for case_id in all_case_ids if case_id.startswith(TASK1_PREFIX)]
task2_ids = [case_id for case_id in all_case_ids if not case_id.startswith(TASK1_PREFIX)]
# --- End of Adaptation Section ---

print(f"Found {len(task1_ids)} Task 1 cases and {len(task2_ids)} Task 2 cases.")

# --- 2. Perform a reproducible shuffle and split on Task 2 data ---
# This part of your code was already correct!
np.random.seed(12345)
np.random.shuffle(task2_ids)

# Calculate the split index
split_idx = int(len(task2_ids) * 0.8) # 80% for training

# Create the Task 2 splits
task2_train_ids = task2_ids[:split_idx]
task2_val_ids = task2_ids[split_idx:]

# --- 3. Combine with Task 1 data to create the final key lists ---
# The training set contains ALL of Task 1 and the training part of Task 2
# The .sort() method is slightly more efficient than sorted() here.
final_train_keys = task1_ids + task2_train_ids
final_train_keys.sort()

# The validation set contains ONLY the validation part of Task 2
final_val_keys = task2_ids
final_val_keys.sort()


# --- 4. Print results and create the JSON structure ---
print(f"Final training cases: {len(final_train_keys)}")
print(f"Final validation cases: {len(final_val_keys)}")

# This is the dictionary structure for your splits file
splits_for_json = [{
    "train": final_train_keys,
    "val": final_val_keys
}]

# --- FIX 4 (Solved): This now works because the lists contain strings ---
with open('splits_final_only.json', 'w') as f:
    json.dump(splits_for_json, f, indent=4)
    
print("\nSuccessfully created splits_final.json!")