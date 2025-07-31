#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# This code uses the surface-distance library by DeepMind.
# Installation instructions available at: https://github.com/deepmind/surface-distance
"""
Evaluate 3D segmentation performance for all subjects in pred_dir,
using only .mha and .nii.gz files. This script:
 - Loads 3D masks and extracts voxel spacing.
 - Ensures prediction masks are binary (0 and 1).
 - If a prediction mask is uniform (all zeros or all ones), all metrics are set to the lowest value possible.
 - Computes surface-based metrics (Dice, Surface Dice at 5mm, Robust Hausdorff95, MASD).
 - Computes tumor volumes and later aggregates metrics (mean for most and RMSE for volumes).
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from surface_distance import metrics as surface_metrics


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


def load_mask(file_path):
    """
    Loads a 3D mask from a file using SimpleITK.
    Allowed extensions: .mha, .nii.gz, (also .nii, .mhd if needed).
    Returns:
      mask: a numpy array representation of the image.
      spacing: a tuple with the voxel spacing (in mm).
    Raises an error if the file is not one of the allowed types or if the image is not 3D.
    """
    if not any(file_path.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError(
            f"Only {ALLOWED_EXTENSIONS} files are allowed. Got: {file_path}")
    image = sitk.ReadImage(file_path)
    mask = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()  # e.g., (1.0, 1.0, 1.0)
    if mask.ndim != 3:
        raise ValueError(
            f"Mask from {file_path} is not 3D (found shape: {mask.shape}).")
    return mask, spacing


def find_file(directory, subject, allowed_extensions=ALLOWED_EXTENSIONS):
    """
    Given a directory and a subject ID, returns the file path if a file with
    subject+extension exists, checking the allowed extensions.
    """
    for ext in allowed_extensions:
        file_path = os.path.join(directory, subject + ext)
        if os.path.exists(file_path):
            return file_path
    return None


# MODIFIED to accept include and exclude parameters
def evaluate_segmentation_performance(pred_dir, gt_dir, subject_list=None, verbose=False, include=None, exclude=None):
    """
    Evaluates segmentation metrics for all subjects.
    - pred_dir: Directory containing prediction files (.mha or .nii.gz).
    - gt_dir: Directory containing ground truth files (.mha or .nii.gz).
    - subject_list: Either a list of subject IDs or a JSON file (with "subject_list" key).
    - verbose: If True, prints per-subject metrics.
    - include: If set, only processes files ending with this string.
    - exclude: If set, skips files ending with this string.

    Returns a dictionary with per-subject metrics and aggregated metrics.
    """

    # Collect nnUNet_results fold_1 to fold_4 under fold_all in same directory to process
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

        # 3. Find all files ending in .nii.gz using Path.glob()
        all_nii_files = list(Path(source_dir).glob("*.nii.gz"))

        # Apply filters based on include/exclude arguments
        nii_files = all_nii_files
        if include:
            print(f"  -- Applying include filter: ending with '{include}'")
            nii_files = [f for f in nii_files if f.name.endswith(include)]
        
        if exclude:
            print(f"  -- Applying exclude filter: NOT ending with '{exclude}'")
            nii_files = [f for f in nii_files if not f.name.endswith(exclude)]

        if not nii_files:
            # Modified message to be more informative
            print("  No .nii.gz files found after applying filters.")
            continue

        for nii_file_path in nii_files:
            destination_path = os.path.join(pred_dir, nii_file_path.name)
            print(f"  -> Copying '{nii_file_path.name}'")
            shutil.copy(nii_file_path, destination_path)
            file_copy_count += 1
            
    # --- The rest of your function remains the same ---
    # ... (code to load subjects, compute metrics, etc.) ...
    # Load subject list from JSON file if subject_list is a filename/path.
    if isinstance(subject_list, (str, Path)):
        with open(subject_list, "r") as fp:
            subject_list = json.load(fp)["subject_list"]
    # If not provided, list all subject IDs from pred_dir (remove extension appropriately)
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

        if pred_file is None:
            if verbose:
                print(f"Prediction file not found for subject {subj}")
            continue
        if gt_file is None:
            if verbose:
                print(f"Ground truth file not found for subject {subj}")
            continue

        try:
            mask_pred, spacing_pred = load_mask(pred_file)
            mask_gt, spacing_gt = load_mask(gt_file)
        except Exception as e:
            if verbose:
                print(f"Error loading subject (or mask and spacing) {subj}: {e}")
            continue

        # Check that the shapes match.
        if mask_gt.shape != mask_pred.shape:
            raise ValueError(
                f"Shape mismatch for subject {subj}: GT shape {mask_gt.shape} vs Pred shape {mask_pred.shape}")
        # Check that the voxel spacings match.
        if not np.allclose(spacing_gt, spacing_pred, rtol=0, atol=1e-4):
                raise ValueError(
                    f"Voxel spacing mismatch: GT spacing {spacing_gt} vs Pred spacing {spacing_pred}")


        # Ensure prediction mask is binary.
        mask_pred = (mask_pred == 1).astype(np.uint8).astype(bool)
        mask_gt = (mask_gt == 1).astype(np.uint8).astype(bool)
        # Convert masks to boolean as required by the surface-distance library.
        mask_pred = mask_pred.astype(bool)
        mask_gt = mask_gt.astype(bool)

        
        # Check for uniform prediction (all zeros or all ones)
        if np.all(mask_pred == 0) or np.all(mask_pred == 1):
            if verbose:
                print(f"Subject {subj}: Prediction mask is uniform. Metrics set to 0.")
            max_distance = np.linalg.norm(
                np.array(mask_gt.shape) * np.array(spacing_gt))
            subj_metrics = {
                "subject": subj,
                "volumetric_dice": 0.0,
                "surface_dice": 0.0,
                "hausdorff95": max_distance,
                "masd": max_distance,
                "gt_volume": np.sum(mask_gt) * np.prod(spacing_gt),
                "pred_volume": 0.0,
                "time_score": 0.0
            }
            metrics_list.append(subj_metrics)
            continue

        # Compute surface-based metrics using the ground truth spacing.
        surface_distances = surface_metrics.compute_surface_distances(
            mask_gt, mask_pred, spacing_mm=spacing_gt)
        dice = surface_metrics.compute_dice_coefficient(mask_gt, mask_pred)
        surf_dice = surface_metrics.compute_surface_dice_at_tolerance(
            surface_distances, tolerance_mm=5)
        hausdorff95 = surface_metrics.compute_robust_hausdorff(
            surface_distances, percent=95)
        avg_gt_to_pred, avg_pred_to_gt = surface_metrics.compute_average_surface_distance(
            surface_distances)
        masd = (avg_gt_to_pred + avg_pred_to_gt) / 2.0

        # Compute tumor volumes using the ground truth spacing.
        voxel_volume = np.prod(spacing_gt)
        gt_volume = np.sum(mask_gt) * voxel_volume
        pred_volume = np.sum(mask_pred) * voxel_volume

        subj_metrics = {
            "subject": subj,
            "volumetric_dice": dice,
            "surface_dice": surf_dice,
            "hausdorff95": hausdorff95,
            "masd": masd,
            "gt_volume": gt_volume,
            "pred_volume": pred_volume,
        }
        metrics_list.append(subj_metrics)
        if verbose:
            print(f"Subject: {subj}")
            print(f"  Volumetric Dice: {dice:.4f}")
            print(f"  Surface Dice (5mm): {surf_dice:.4f}")
            print(f"  Hausdorff95: {hausdorff95:.4f}")
            print(f"  MASD: {masd:.4f}")
            print(
                f"  GT Volume: {gt_volume:.2f} mm³, Pred Volume: {pred_volume:.2f} mm³")

    if len(metrics_list) == 0:
        raise RuntimeError("No subjects were processed successfully!")

    mean_dice = np.mean([m["volumetric_dice"] for m in metrics_list])
    mean_surf_dice = np.mean([m["surface_dice"] for m in metrics_list])
    mean_hausdorff95 = np.mean([m["hausdorff95"] for m in metrics_list])
    mean_masd = np.mean([m["masd"] for m in metrics_list])
    gt_volumes = np.array([m["gt_volume"] for m in metrics_list])
    pred_volumes = np.array([m["pred_volume"] for m in metrics_list])
    rmse_volume = np.sqrt(np.mean((pred_volumes - gt_volumes) ** 2))
    
    # Delete the fold_all folder only if it was created by this script
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)

    aggregates = {
        "mean_volumetric_dice": mean_dice,
        "mean_surface_dice": mean_surf_dice,
        "mean_hausdorff95": mean_hausdorff95,
        "mean_masd": mean_masd,
        "tumor_burden_rmse": rmse_volume,
    }

    return {
        "per_subject": metrics_list,
        "aggregates": aggregates,
    }

    


if __name__ == "__main__":
    import argparse
    import json
            
    parser = argparse.ArgumentParser(description="Evaluate 3D segmentation performance for .mha and .nii.gz masks")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction files (.mha or .nii.gz)")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth files (.mha or .nii.gz)")
    parser.add_argument("--subject_list", type=str, default=None,
                        help="Optional JSON file with {'subject_list': [...]}, or a comma-separated list of subject IDs")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Optional path to save the aggregated metrics as a JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--include", type=str, default=None,
                        help="Only include files ending with this string (e.g., '_0001.nii.gz').")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Exclude files ending with this string (e.g., '_0001.nii.gz').")

    args = parser.parse_args()
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
