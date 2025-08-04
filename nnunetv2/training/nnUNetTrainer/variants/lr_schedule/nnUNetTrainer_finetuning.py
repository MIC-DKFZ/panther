import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer1e3(nnUNetTrainer):
    """
    Does a warmup of the entire architecture
    Then does normal training
    """
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3


import os
import sys
import shutil

from os.path import join, isfile, isdir
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.data_augmentation.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter, NonDetMultiThreadedAugmenter


class nnUNetTrainer1e3150e(nnUNetTrainer):
    """
    This trainer customizes the nnUNetTrainer with three key features:
    1. Sets the initial learning rate to 1e-3.
    2. Sets the total number of epochs to 150.
    3. At the end of training, automatically evaluates both the 'final' and 'best' checkpoints
       and saves their validation results to 'validation_final' and 'validation' respectively.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 150

    def on_train_end(self):
        # This hack is required to save the correct epoch number
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # The model's weights are now from the final epoch. The dataloaders are still active.

        self.print_to_log_file("Starting automated dual-checkpoint validation...")

        # A) Evaluate the final checkpoint (currently loaded)
        self.print_to_log_file("Running validation on 'checkpoint_final.pth'...")
        self.perform_actual_validation(save_probabilities=self.configuration_manager.plans['configurations'][self.configuration]['save_probabilities'])

        # Rename the output folder from 'validation' to 'validation_final'
        default_validation_folder = join(self.output_folder, 'validation')
        final_validation_folder = join(self.output_folder, 'validation_final')
        if isdir(default_validation_folder):
            if isdir(final_validation_folder):
                shutil.rmtree(final_validation_folder)
            shutil.move(default_validation_folder, final_validation_folder)
            self.print_to_log_file(f"Validation results for final checkpoint saved to: {final_validation_folder}")

        # B) Evaluate the best checkpoint
        best_checkpoint_path = join(self.output_folder, 'checkpoint_best.pth')
        if isfile(best_checkpoint_path):
            self.print_to_log_file("Loading 'checkpoint_best.pth' for validation...")
            self.load_checkpoint(best_checkpoint_path)
            # Run validation again, as 'validation'
            self.perform_actual_validation(save_probabilities=self.configuration_manager.plans['configurations'][self.configuration]['save_probabilities'])
            self.print_to_log_file(f"Validation results for best checkpoint saved to: {default_validation_folder}")
        else:
            self.print_to_log_file("'checkpoint_best.pth' not found, skipping validation on best checkpoint.")

        # --- PART 3: Cleanup and shutdown (copied from base class) ---
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # Shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and isinstance(self.dataloader_val, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training and all validations are finished.")
        