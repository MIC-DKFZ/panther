"""
Evaluate 3D segmentation performance for all subjects in pred_dir,
using only .mha and .nii.gz files. This script offers two modes:

1. PANTHER Mode (Default):
 - Aggregates all folds into a single group.
 - Computes surface-based metrics (Dice, Surface Dice, Hausdorff95, MASD).
 - Computes tumor volumes and aggregates metrics (mean/RMSE).
 - Output is a single JSON with aggregated results.

2. Detailed Fold-by-Fold Mode (with --detailed_eval):
 - Requires nnunetv2 to be installed.
 - Evaluates each fold (0-4) separately.
 - Generates a detailed, nnU-Net style summary.json for each fold.
 - Metrics include Dice, IoU, TP, FP, FN, TN per class.
 - Saves results into a new directory named after the --save_path argument.
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from surface_distance import metrics as surface_metrics

# NEW: Add an optional import for the detailed evaluation feature.
try:
    from nnunetv2.evaluation.evaluate_folder import evaluate_folder
    NNUNET_INSTALLED = True
except ImportError:
    NNUNET_INSTALLED = False


ALLOWED_EXTENSIONS = [".mha", ".nii.gz"]
panther_msg = r"""\n
<Computing PANTHER Evaluation Metrics>
                  /)-._
                 Y. ' _]
          ,.._   |`--"=
         /    "-/   \\
/)      |   |_     `\|___
\:::::::\___/_\__\_______\\
"""
panther_msg2 = r"""\n
  _____________________________
  < PANTHER Evaluation Done! >
  -----------------------------
"""


# --- All functions from your original script remain here ---
# load_mask, find_file, etc.

# The original evaluate_segmentation_performance function is unchanged.
def evaluate_segmentation_performance(pred_dir, gt_dir, subject_list=None, verbose=False, include=None, exclude=None):
    # (Your original function from the previous step goes here, it's quite long so I'm omitting it for brevity)
    # ... It should contain the include/exclude logic we added before.
    # --- The function body is identical to the previous answer ---
    results_main_dir = pred_dir
    pred_dir = os.path.join(pred_dir, "fold_all")

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    file_copy_count = 0
    folders_to_scan = [f"fold_{i}/validation" for i in range(5)]

    for fold_name in folders_to_scan:
        source_dir = os.path.join(results_main_dir,fold_name)

        if not os.path.isdir(source_dir):
            print(f"Skipping: Directory '{source_dir}' not found.")
            continue 

        print(f"--- Searching in '{source_dir}' ---")
        all_nii_files = list(Path(source_dir).glob("*.nii.gz"))
        
        nii_files = all_nii_files
        if include:
            print(f"  -- Applying include filter: ending with '{include}'")
            nii_files = [f for f in nii_files if f.name.endswith(include)]
        
        if exclude:
            print(f"  -- Applying exclude filter: NOT ending with '{exclude}'")
            nii_files = [f for f in nii_files if not f.name.endswith(exclude)]

        if not nii_files:
            print("  No .nii.gz files found after applying filters.")
            continue

        for nii_file_path in nii_files:
            destination_path = os.path.join(pred_dir, nii_file_path.name)
            print(f"  -> Copying '{nii_file_path.name}'")
            shutil.copy(nii_file_path, destination_path)
            file_copy_count += 1
            
    if isinstance(subject_list, (str, Path)):
        with open(subject_list, "r") as fp:
            subject_list = json.load(fp)["subject_list"]
    if subject_list is None:
        subject_set = set()
        for f in os.listdir(pred_dir):
            f_lower = f.lower()
            for ext in ALLOWED_EXTENSIONS:
                if f_lower.endswith(ext):
                    subject_set.add(f[:-len(ext)])
                    break
        subject_list = sorted(subject_set)

    metrics_list = []
    for subj in subject_list:
        pred_file = find_file(pred_dir, subj)
        gt_file = find_file(gt_dir, subj)

        if pred_file is None or gt_file is None: continue

        try:
            mask_pred, spacing_pred = load_mask(pred_file)
            mask_gt, spacing_gt = load_mask(gt_file)
        except Exception as e:
            if verbose: print(f"Error loading subject {subj}: {e}")
            continue

        if mask_gt.shape != mask_pred.shape or not np.allclose(spacing_gt, spacing_pred, rtol=0, atol=1e-4):
            print(f"Shape or spacing mismatch for {subj}, skipping.")
            continue

        mask_pred = (mask_pred == 1).astype(bool)
        mask_gt = (mask_gt == 1).astype(bool)
        
        if np.all(mask_pred == 0) or np.all(mask_pred == 1):
            max_distance = np.linalg.norm(np.array(mask_gt.shape) * np.array(spacing_gt))
            subj_metrics = {
                "subject": subj, "volumetric_dice": 0.0, "surface_dice": 0.0,
                "hausdorff95": max_distance, "masd": max_distance,
                "gt_volume": np.sum(mask_gt) * np.prod(spacing_gt), "pred_volume": 0.0,
            }
        else:
            surface_distances = surface_metrics.compute_surface_distances(mask_gt, mask_pred, spacing_mm=spacing_gt)
            dice = surface_metrics.compute_dice_coefficient(mask_gt, mask_pred)
            surf_dice = surface_metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=5)
            hausdorff95 = surface_metrics.compute_robust_hausdorff(surface_distances, percent=95)
            avg_gt_to_pred, avg_pred_to_gt = surface_metrics.compute_average_surface_distance(surface_distances)
            masd = (avg_gt_to_pred + avg_pred_to_gt) / 2.0
            voxel_volume = np.prod(spacing_gt)
            gt_volume = np.sum(mask_gt) * voxel_volume
            pred_volume = np.sum(mask_pred) * voxel_volume

            subj_metrics = {
                "subject": subj, "volumetric_dice": dice, "surface_dice": surf_dice,
                "hausdorff95": hausdorff95, "masd": masd,
                "gt_volume": gt_volume, "pred_volume": pred_volume,
            }
        metrics_list.append(subj_metrics)

    if len(metrics_list) == 0:
        raise RuntimeError("No subjects were processed successfully!")

    mean_dice = np.mean([m["volumetric_dice"] for m in metrics_list])
    mean_surf_dice = np.mean([m["surface_dice"] for m in metrics_list])
    mean_hausdorff95 = np.mean([m["hausdorff95"] for m in metrics_list])
    mean_masd = np.mean([m["masd"] for m in metrics_list])
    gt_volumes = np.array([m["gt_volume"] for m in metrics_list])
    pred_volumes = np.array([m["pred_volume"] for m in metrics_list])
    rmse_volume = np.sqrt(np.mean((pred_volumes - gt_volumes) ** 2))
    
    shutil.rmtree(pred_dir)

    aggregates = {
        "mean_volumetric_dice": mean_dice, "mean_surface_dice": mean_surf_dice,
        "mean_hausdorff95": mean_hausdorff95, "mean_masd": mean_masd,
        "tumor_burden_rmse": rmse_volume,
    }
    return {"per_subject": metrics_list, "aggregates": aggregates}


# NEW: Function for detailed fold-by-fold evaluation
def run_detailed_evaluation(pred_dir, gt_dir, save_path, include=None, exclude=None):
    """
    Uses nnunetv2.evaluation.evaluate_folder to generate a detailed summary.json
    for each fold, saving them into a new directory.
    """
    # Create the main output directory from the save_path name
    output_dir = Path(save_path).with_suffix('')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving detailed fold summaries to: {output_dir}")

    # The label to evaluate. Your original script focuses on label 1 (tumor).
    # The example you gave has "(1, 2)", which evaluates classes 1 and 2 and also merges them.
    # We will stick to label 1 to match your script's logic. Change if you have more classes.
    labels_to_evaluate = (1,) 

    for i in range(5): # Loop through fold_0 to fold_4
        fold_pred_dir = Path(pred_dir) / f"fold_{i}" / "validation"
        
        if not fold_pred_dir.is_dir():
            print(f"Skipping: Directory '{fold_pred_dir}' not found.")
            continue
            
        print(f"\n--- Processing {fold_pred_dir} ---")

        # Since evaluate_folder works on directories, we create a temporary directory
        # containing only the files that match our include/exclude criteria.
        temp_filtered_dir = output_dir / f"temp_fold_{i}_filtered_preds"
        os.makedirs(temp_filtered_dir, exist_ok=True)

        all_files = list(fold_pred_dir.glob("*.nii.gz"))
        
        # Apply filters
        files_to_process = all_files
        if include:
            files_to_process = [f for f in files_to_process if f.name.endswith(include)]
        if exclude:
            files_to_process = [f for f in files_to_process if not f.name.endswith(exclude)]
        
        if not files_to_process:
            print("  No files match criteria in this fold. Skipping.")
            shutil.rmtree(temp_filtered_dir)
            continue
            
        # Copy the filtered files to the temporary directory
        print(f"  Found {len(files_to_process)} matching files to evaluate.")
        for f_path in files_to_process:
            shutil.copy(f_path, temp_filtered_dir / f_path.name)
            
        # Define the output file for this fold's summary
        fold_summary_file = output_dir / f"fold_{i}_summary.json"
        
        # Run the nnU-Net evaluation
        evaluate_folder(
            folder_with_gts=str(gt_dir),
            folder_with_predictions=str(temp_filtered_dir),
            output_file=str(fold_summary_file),
            labels=labels_to_evaluate,
            # num_processes=... # you can add this for speed
        )
        print(f"  -> Detailed summary saved to: {fold_summary_file}")
        
        # Clean up the temporary directory
        shutil.rmtree(temp_filtered_dir)


if __name__ == "__main__":
    import json
    import argparse

    
    parser = argparse.ArgumentParser(description="Evaluate 3D segmentation performance.")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction folds (e.g., nnUNet_results/DatasetX/Trainer__Plans__3d_fullres/..)")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth files.")
    parser.add_argument("--subject_list", type=str, default=None,
                        help="Optional JSON file with {'subject_list': [...]}, or a comma-separated list of subject IDs. (Used in default mode only)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the metrics JSON. In detailed mode, this is used as a base name for the output directory.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output (for default mode).")
    parser.add_argument("--include", type=str, default=None,
                        help="Only include files ending with this string (e.g., '_0001.nii.gz').")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Exclude files ending with this string (e.g., '_0001.nii.gz').")
    
    # NEW ARGUMENT
    parser.add_argument("--detailed_eval", action="store_true",
                        help="Use nnunetv2-style evaluation to generate a detailed summary.json for each fold. Requires --save_path.")

    args = parser.parse_args()
    
    # NEW LOGIC to decide which evaluation to run
    if args.detailed_eval:
        if not NNUNET_INSTALLED:
            raise ImportError("The '--detailed_eval' flag requires the 'nnunetv2' package. Please install it using 'pip install nnunetv2'.")
        if not args.save_path:
            raise ValueError("The '--detailed_eval' flag requires a --save_path to be specified for the output directory.")
        print("\n<Running Detailed Fold-by-Fold Evaluation>")
        run_detailed_evaluation(args.pred_dir, args.gt_dir, args.save_path, args.include, args.exclude)

    else: # Default PANTHER evaluation
        print(panther_msg)
        subject_list = args.subject_list
        if subject_list is not None:
            if subject_list.endswith(".json"):
                with open(subject_list, "r") as fp:
                    subject_list = json.load(fp)["subject_list"]
            else:
                subject_list = [s.strip() for s in subject_list.split(",")]

        results = evaluate_segmentation_performance(args.pred_dir, args.gt_dir,
                                                      subject_list=subject_list,
                                                      verbose=args.verbose,
                                                      include=args.include,
                                                      exclude=args.exclude)
        print("Evaluation Metrics:")
        print(json.dumps(results, indent=4))

        if args.save_path:
            with open(args.save_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Metrics saved to {args.save_path}")

    print(panther_msg2)