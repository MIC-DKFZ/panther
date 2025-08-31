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
import multiprocessing
from copy import deepcopy
from typing import Tuple, List, Union

# --- DEPENDENCY CHECK FOR VENDORED CODE ---
# The detailed evaluation logic below requires 'batchgenerators'.
try:
    from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, isfile
except ImportError:
    raise ImportError(
        "The '--detailed_eval' flag uses logic that requires the 'batchgenerators' package. "
        "Please install it using 'pip install batchgenerators'."
    )
# --- END DEPENDENCY CHECK ---


# ###################################################################################################
# ### NNUNETV2 EVALUATION LOGIC (VENDORED) ###
# This section contains functions adapted from nnUNetv2's evaluation scripts.
# This makes the --detailed_eval feature self-contained and independent of a full nnU-Net installation.
# ###################################################################################################
class SimpleITKIO:
    """A simple reader/writer class that mimics nnunetv2's ImageIO logic using SimpleITK."""
    def read_seg(self, path: str) -> Tuple[np.ndarray, dict]:
        img = sitk.ReadImage(path)
        properties = {
            'spacing': img.GetSpacing(),
            'direction': img.GetDirection(),
            'origin': img.GetOrigin(),
        }
        return sitk.GetArrayFromImage(img), properties

    def write_seg(self, seg: np.ndarray, path: str, properties: dict):
        img = sitk.GetImageFromArray(seg)
        img.SetSpacing(properties['spacing'])
        img.SetDirection(properties['direction'])
        img.SetOrigin(properties['origin'])
        sitk.WriteImage(img, path)


def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def save_summary_json(results: dict, output_file: str):
    """Saves a summary JSON file, converting tuple keys to strings."""
    results_converted = deepcopy(results)
    if 'mean' in results_converted:
        results_converted['mean'] = {label_or_region_to_key(k): v for k, v in results_converted['mean'].items()}
    if 'metric_per_case' in results_converted:
        for i in range(len(results_converted["metric_per_case"])):
            results_converted["metric_per_case"][i]['metrics'] = \
                {label_or_region_to_key(k): v for k, v in results_converted["metric_per_case"][i]['metrics'].items()}
    save_json(results_converted, output_file, sort_keys=True)


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: SimpleITKIO,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None) -> dict:
    seg_ref, _ = image_reader_writer.read_seg(reference_file)
    seg_pred, _ = image_reader_writer.read_seg(prediction_file)
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {'reference_file': reference_file, 'prediction_file': prediction_file, 'metrics': {}}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r].update({'FP': fp, 'TP': tp, 'FN': fn, 'TN': tn, 'n_pred': fp + tp, 'n_ref': fn + tp})
    return results


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: SimpleITKIO, file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, ...]]],
                              ignore_label: int = None, num_processes: int = 8, chill: bool = True):
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref_not_present = [i for i in files_pred if not isfile(join(folder_ref, i))]
    if not chill and len(files_ref_not_present) > 0:
        raise RuntimeError(f"Some files in folder_pred are not in folder_ref: {files_ref_not_present}")
        
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]

    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_metrics,
            zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))
        )

    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {r: {m: np.nanmean([i['metrics'][r][m] for i in results]) for m in metric_list} for r in regions_or_labels}
    foreground_mean = {m: np.mean([means[k][m] for k in means if k != 0 and k != '0']) for m in metric_list}
    
    result_final = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}

    # Apply the fix to convert numpy types to native python types before saving
    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)

    result_final = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result_final, output_file)
    return result_final


def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str, labels: Union[Tuple[int, ...], List[int]],
                                     output_file: str = None, num_processes: int = 8,
                                     ignore_label: int = None, chill: bool = False):
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending = os.path.splitext(example_file)[-1]
    rw = SimpleITKIO()
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending, labels,
                              ignore_label=ignore_label, num_processes=num_processes, chill=chill)

# ###################################################################################################
# ### END VENDORED CODE ###
# ###################################################################################################

def evaluate_fold_separately(pred_dir: str, gt_dir: str, val_best: bool = False):
    """
    Evaluates segmentation metrics for each fold separately by directly
    evaluating the contents of each fold's validation directory.
    """
    print(f"\n<Running Fold-Separate Evaluation for Model: {os.path.basename(pred_dir)}>")

    all_folds_results = {}
    
    try:
        fold_dirs = sorted([d for d in os.listdir(pred_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(pred_dir, d))])
    except FileNotFoundError:
        print(f"Error: Prediction directory not found at '{pred_dir}'")
        return {}

    if not fold_dirs:
        print(f"Error: No 'fold_X' directories found in '{pred_dir}'.")
        return {}

    for fold_name in fold_dirs:
        print(f"\n--- Processing {fold_name} ---")
        
        validation_subfolder = "validation" if val_best else "validation_final"
        fold_pred_dir = os.path.join(pred_dir, fold_name, validation_subfolder)

        if not os.path.isdir(fold_pred_dir):
            print(f"  Validation folder '{validation_subfolder}' not found in '{os.path.join(pred_dir, fold_name)}'. Skipping.")
            continue
        
        try:
            subject_list = sorted(list(set([f.replace('.nii.gz', '') for f in os.listdir(fold_pred_dir) if f.endswith('.nii.gz')])))
            if not subject_list:
                print("  No .nii.gz prediction files found in this folder. Skipping.")
                continue
            print(f"  Found {len(subject_list)} prediction files to evaluate.")
        except FileNotFoundError:
            print(f"  Prediction folder not found at '{fold_pred_dir}'. Skipping.")
            continue

        try:
            results = evaluate_segmentation_performance(
                pred_dir=fold_pred_dir,
                gt_dir=gt_dir,
                subject_list=subject_list,
                is_fold_separate_run=True,
                ### FIX: Added tumor_label_only=True to ensure Dice is only for label 1.
                tumor_label_only=True
            )
            
            if results and results['aggregates']:
                all_folds_results[fold_name] = results['aggregates']
                print(f"  -> Results for {fold_name}:")
                for metric, value in results['aggregates'].items():
                    print(f"     {metric}: {value:.4f}")
            else:
                print(f"  -> No valid metrics could be computed for {fold_name}.")
                all_folds_results[fold_name] = {"error": "No metrics computed."}


        except Exception as e:
            print(f"  Error processing {fold_name}: {e}")
            all_folds_results[fold_name] = {"error": str(e)}

    return all_folds_results

def recursive_fix_for_json_export(my_dict: dict):
    """
    Converts numpy types to native python types in a dictionary, recursively.
    """
    for k, v in my_dict.items():
        if isinstance(v, dict):
            recursive_fix_for_json_export(v)
        elif isinstance(v, (np.ndarray, np.generic)):
            my_dict[k] = v.item()


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
    """
    if not any(file_path.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError(
            f"Only {ALLOWED_EXTENSIONS} files are allowed. Got: {file_path}")
    image = sitk.ReadImage(file_path)
    mask = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
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
    
### FIX: Added tumor_label_only parameter to control binarization logic.
def evaluate_segmentation_performance(pred_dir, gt_dir, subject_list=None, verbose=False, include=None, exclude=None, val_best=False, is_fold_separate_run=False, tumor_label_only=False):
    """
    Evaluates segmentation metrics for all subjects.
    """
    if not is_fold_separate_run:
        print("<Running in AGGREGATE mode: Combining all folds>")
        results_main_dir = pred_dir
        pred_dir = os.path.join(results_main_dir, "fold_all")

        if os.path.exists(pred_dir):
            shutil.rmtree(pred_dir)
        os.makedirs(pred_dir, exist_ok=True)

        file_copy_count = 0
        folders_to_scan = [f"fold_{i}/" + ("validation" if val_best else "validation_final") for i in range(5)]

        for fold_name in folders_to_scan:
            source_dir = os.path.join(results_main_dir, fold_name)
            if not os.path.isdir(source_dir):
                if verbose: print(f"Skipping: Directory '{source_dir}' not found.")
                continue
            
            all_nii_files = list(Path(source_dir).glob("*.nii.gz"))
            for nii_file_path in all_nii_files:
                destination_path = os.path.join(pred_dir, nii_file_path.name)
                shutil.copy(nii_file_path, destination_path)
                file_copy_count += 1
        
        print(f"Aggregated {file_copy_count} prediction files into temporary 'fold_all' directory for evaluation.")
    else:
        if verbose: print(f"<Running in FOLD-SEPARATE mode on directory: {pred_dir}>")

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

        if pred_file is None or gt_file is None:
            continue

        try:
            mask_pred, spacing_pred = load_mask(pred_file)
            mask_gt, spacing_gt = load_mask(gt_file)
        except Exception as e:
            if verbose: print(f"Error loading subject {subj}: {e}")
            continue

        if mask_gt.shape != mask_pred.shape or not np.allclose(spacing_gt, spacing_pred, rtol=0, atol=1e-4):
            if verbose: print(f"Shape or spacing mismatch for subject {subj}, skipping.")
            continue

        ### FIX: Conditional binarization.
        # If tumor_label_only is True, only label 1 is considered foreground.
        # Otherwise, any label > 0 is considered foreground.
        if tumor_label_only:
            mask_pred = (mask_pred == 1).astype(bool)
            mask_gt = (mask_gt == 1).astype(bool)
        else:
            mask_pred = (mask_pred > 0).astype(bool)
            mask_gt = (mask_gt > 0).astype(bool)

        if np.all(mask_pred == 0):
            if np.any(mask_gt):
                 max_distance = np.linalg.norm(np.array(mask_gt.shape) * np.array(spacing_gt))
                 subj_metrics = { "subject": subj, "volumetric_dice": 0.0, "surface_dice": 0.0, "hausdorff95": max_distance, "masd": max_distance, "gt_volume": np.sum(mask_gt) * np.prod(spacing_gt), "pred_volume": 0.0 }
            else:
                 subj_metrics = { "subject": subj, "volumetric_dice": 1.0, "surface_dice": 1.0, "hausdorff95": 0.0, "masd": 0.0, "gt_volume": 0.0, "pred_volume": 0.0 }
            metrics_list.append(subj_metrics)
            continue
        
        if not np.any(mask_gt):
            max_distance = np.linalg.norm(np.array(mask_gt.shape) * np.array(spacing_gt))
            subj_metrics = { "subject": subj, "volumetric_dice": 0.0, "surface_dice": 0.0, "hausdorff95": max_distance, "masd": max_distance, "gt_volume": 0.0, "pred_volume": np.sum(mask_pred) * np.prod(spacing_gt) }
            metrics_list.append(subj_metrics)
            continue

        surface_distances = surface_metrics.compute_surface_distances(mask_gt, mask_pred, spacing_mm=spacing_gt)
        dice = surface_metrics.compute_dice_coefficient(mask_gt, mask_pred)
        surf_dice = surface_metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=5)
        hausdorff95 = surface_metrics.compute_robust_hausdorff(surface_distances, percent=95)
        avg_gt_to_pred, avg_pred_to_gt = surface_metrics.compute_average_surface_distance(surface_distances)
        masd = (avg_gt_to_pred + avg_pred_to_gt) / 2.0
        voxel_volume = np.prod(spacing_gt)
        gt_volume = np.sum(mask_gt) * voxel_volume
        pred_volume = np.sum(mask_pred) * voxel_volume

        subj_metrics = { "subject": subj, "volumetric_dice": dice, "surface_dice": surf_dice, "hausdorff95": hausdorff95, "masd": masd, "gt_volume": gt_volume, "pred_volume": pred_volume }
        metrics_list.append(subj_metrics)

    if not metrics_list:
        return {"per_subject": [], "aggregates": {}}

    mean_dice = np.mean([m["volumetric_dice"] for m in metrics_list])
    mean_surf_dice = np.mean([m["surface_dice"] for m in metrics_list])
    mean_hausdorff95 = np.mean([m["hausdorff95"] for m in metrics_list])
    mean_masd = np.mean([m["masd"] for m in metrics_list])
    gt_volumes = np.array([m["gt_volume"] for m in metrics_list])
    pred_volumes = np.array([m["pred_volume"] for m in metrics_list])
    rmse_volume = np.sqrt(np.mean((pred_volumes - gt_volumes) ** 2))
    
    if not is_fold_separate_run and os.path.exists(pred_dir):
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


def run_detailed_evaluation(pred_dir, gt_dir, save_path, include=None, exclude=None, val_best=False):
    """
    Uses the self-contained evaluation logic to generate a detailed summary.json
    for each fold, saving them into a new directory.
    """
    output_dir = Path(save_path).with_suffix('')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving detailed fold summaries to: {output_dir}")

    labels_to_evaluate = (1, (1, 2))

    for i in range(5):
        fold_pred_dir = Path(pred_dir) / f"fold_{i}" / ("validation" if val_best else "validation_final")
        
        if not fold_pred_dir.is_dir():
            print(f"Skipping: Directory '{fold_pred_dir}' not found.")
            continue
            
        print(f"\n--- Processing {fold_pred_dir} ---")

        temp_filtered_dir = output_dir / f"temp_fold_{i}_filtered_preds"
        os.makedirs(temp_filtered_dir, exist_ok=True)

        files_to_process = list(fold_pred_dir.glob("*.nii.gz"))
        if include:
            files_to_process = [f for f in files_to_process if f.name.endswith(include)]
        if exclude:
            files_to_process = [f for f in files_to_process if not f.name.endswith(exclude)]
        
        if not files_to_process:
            print("  No files match criteria in this fold. Skipping.")
            shutil.rmtree(temp_filtered_dir)
            continue
            
        print(f"  Found {len(files_to_process)} matching files to evaluate.")
        for f_path in files_to_process:
            shutil.copy(f_path, temp_filtered_dir / f_path.name)
            
        fold_summary_file = output_dir / f"fold_{i}_summary.json"
        
        compute_metrics_on_folder_simple(
            folder_ref=str(gt_dir),
            folder_pred=str(temp_filtered_dir),
            labels=labels_to_evaluate,
            output_file=str(fold_summary_file),
            num_processes=8
        )
        print(f"  -> Detailed summary saved to: {fold_summary_file}")
        shutil.rmtree(temp_filtered_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate 3D segmentation performance.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing fold_X subdirectories with predictions.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth segmentations.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the final evaluation results JSON file.")
    parser.add_argument("--include", type=str, default=None, help="Not used in this version.")
    parser.add_argument("--exclude", type=str, default=None, help="Not used in this version.")
    parser.add_argument("--detailed_eval", action="store_true", help="Use self-contained nnU-Net-style evaluation for a detailed summary.json per fold.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")
    parser.add_argument("--subject_list", type=str, default=None, help="Path to a JSON file or a comma-separated string of subject IDs to evaluate.")
    parser.add_argument("--val_best", action="store_true", help="Use 'validation' subfolder instead of 'validation_final'.")
    parser.add_argument("--fold_separate", action="store_true", help="Run evaluation and report metrics for each fold separately.")
    
    args = parser.parse_args()
    
    ### FIX: Restructured the main execution logic to be mutually exclusive.
    # This prevents the default evaluation from running after a specific mode (like --fold_separate) has already finished.
    if args.fold_separate:
        fold_results = evaluate_fold_separately(args.pred_dir, args.gt_dir, args.val_best)
        
        print("\n" + "="*30)
        print("  Final Fold-Separate Summary")
        print("="*30)
        
        # Pretty print the final dictionary
        if fold_results:
            print(json.dumps(fold_results, indent=4))
        else:
            print("No results were generated.")

        if args.save_path:
            with open(args.save_path, "w") as f:
                json.dump(fold_results, f, indent=4)
            print(f"\nFold-separate results saved to {args.save_path}")

    elif args.detailed_eval:
        if not args.save_path:
            raise ValueError("The '--detailed_eval' flag requires a --save_path to be specified.")
        print("\n<Running Detailed Fold-by-Fold Evaluation using self-contained logic>")
        run_detailed_evaluation(args.pred_dir, args.gt_dir, args.save_path, args.include, args.exclude, args.val_best)

    else:
        # Default PANTHER evaluation (aggregate all folds)
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
                                                    exclude=args.exclude,
                                                    val_best=args.val_best,
                                                    is_fold_separate_run=False) # Explicitly False for default mode

        print("\nEvaluation Metrics (Aggregated):")
        print(json.dumps(results, indent=4))

        if args.save_path:
            with open(args.save_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"\nMetrics saved to {args.save_path}")

    print(panther_msg2)