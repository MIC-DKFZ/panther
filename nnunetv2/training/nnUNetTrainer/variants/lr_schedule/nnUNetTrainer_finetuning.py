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
from nnunetv2.utilities.helpers import empty_cache
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler_lin_warmup


class nnUNetTrainer1e3_150e(nnUNetTrainer):
    """
    This custom trainer implements the following features:
    1.  Sets the initial learning rate to 1e-3.
    2.  Limits the training to a total of 150 epochs.
    3.  Overrides the on_train_end method to perform a dual-validation sequence:
        - It first evaluates the final model checkpoint ('checkpoint_final.pth').
        - The resulting validation folder is renamed to 'validation_final'.
        - It then evaluates the best model checkpoint ('checkpoint_best.pth').
        - The resulting validation folder is named 'validation' (the nnU-Net default).
    This entire process is automated and occurs immediately after training finishes.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """
        The constructor's signature is identical to the base nnUNetTrainer.
        It calls the parent constructor and then modifies the learning rate and epoch count.
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 150

    def on_train_end(self):
        """
        This method is a complete override of the base implementation. It integrates the
        validation logic before the dataloaders are shut down.
        """
        # PART 1: Save the final checkpoint. This logic is from the base class.
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))

        # PART 2: Perform the custom dual validation.
        self.print_to_log_file("Starting automated dual-checkpoint validation...")

        # A) Evaluate the final checkpoint (the model currently loaded in memory).
        self.print_to_log_file("Running validation on 'checkpoint_final.pth'...")
        # perform_actual_validation() correctly handles setting the network to eval mode.
        self.perform_actual_validation(save_probabilities=False)

        # Rename the output folder from 'validation' to 'validation_final'.
        default_validation_folder = join(self.output_folder, 'validation')
        final_validation_folder = join(self.output_folder, 'validation_final')
        if isdir(default_validation_folder):
            if isdir(final_validation_folder):
                self.print_to_log_file(f"Deleting existing folder: {final_validation_folder}")
                shutil.rmtree(final_validation_folder)
            shutil.move(default_validation_folder, final_validation_folder)
            self.print_to_log_file(f"Validation results for final checkpoint saved to: {final_validation_folder}")
        else:
            self.print_to_log_file("Could not find validation folder for final checkpoint. Skipping rename.")


        # B) Evaluate the best checkpoint.
        best_checkpoint_path = join(self.output_folder, 'checkpoint_best.pth')
        if isfile(best_checkpoint_path):
            self.print_to_log_file("Loading 'checkpoint_best.pth' for validation...")
            self.load_checkpoint(best_checkpoint_path)
            # Run validation again. The output will be created as 'validation' by default.
            self.perform_actual_validation(save_probabilities=False)
            self.print_to_log_file(f"Validation results for best checkpoint saved to: {default_validation_folder}")
        else:
            self.print_to_log_file("'checkpoint_best.pth' not found, skipping validation on best checkpoint.")

        # PART 3: Clean up latest checkpoint and shut down dataloaders. This logic is from the base class.
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # We need to gracefully shut down the dataloaders. This is critical.
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and isinstance(self.dataloader_val, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training and all custom validations are finished.")


class nnUNetTrainer1e3_150e_polylrlin(nnUNetTrainer):
    """
    This custom trainer implements the following features:
    1.  Sets the initial learning rate to 1e-3.
    2.  Limits the training to a total of 150 epochs.
    3.  Overrides the on_train_end method to perform a dual-validation sequence:
        - It first evaluates the final model checkpoint ('checkpoint_final.pth').
        - The resulting validation folder is renamed to 'validation_final'.
        - It then evaluates the best model checkpoint ('checkpoint_best.pth').
        - The resulting validation folder is named 'validation' (the nnU-Net default).
    This entire process is automated and occurs immediately after training finishes.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """
        The constructor's signature is identical to the base nnUNetTrainer.
        It calls the parent constructor and then modifies the learning rate and epoch count.
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 150

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        self.print_to_log_file("Using custom LR scheduler: PolyLRScheduler_lin_warmup")
        self.lr_scheduler = PolyLRScheduler_lin_warmup(self.optimizer, self.initial_lr, warmup_steps=50, max_steps=self.num_epochs)    
        return self.optimizer, self.lr_scheduler

    def on_train_end(self):
        """
        This method is a complete override of the base implementation. It integrates the
        validation logic before the dataloaders are shut down.
        """
        # PART 1: Save the final checkpoint. This logic is from the base class.
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))

        # PART 2: Perform the custom dual validation.
        self.print_to_log_file("Starting automated dual-checkpoint validation...")

        # A) Evaluate the final checkpoint (the model currently loaded in memory).
        self.print_to_log_file("Running validation on 'checkpoint_final.pth'...")
        # perform_actual_validation() correctly handles setting the network to eval mode.
        self.perform_actual_validation(save_probabilities=False)

        # Rename the output folder from 'validation' to 'validation_final'.
        default_validation_folder = join(self.output_folder, 'validation')
        final_validation_folder = join(self.output_folder, 'validation_final')
        if isdir(default_validation_folder):
            if isdir(final_validation_folder):
                self.print_to_log_file(f"Deleting existing folder: {final_validation_folder}")
                shutil.rmtree(final_validation_folder)
            shutil.move(default_validation_folder, final_validation_folder)
            self.print_to_log_file(f"Validation results for final checkpoint saved to: {final_validation_folder}")
        else:
            self.print_to_log_file("Could not find validation folder for final checkpoint. Skipping rename.")


        # B) Evaluate the best checkpoint.
        best_checkpoint_path = join(self.output_folder, 'checkpoint_best.pth')
        if isfile(best_checkpoint_path):
            self.print_to_log_file("Loading 'checkpoint_best.pth' for validation...")
            self.load_checkpoint(best_checkpoint_path)
            # Run validation again. The output will be created as 'validation' by default.
            self.perform_actual_validation(save_probabilities=False)
            self.print_to_log_file(f"Validation results for best checkpoint saved to: {default_validation_folder}")
        else:
            self.print_to_log_file("'checkpoint_best.pth' not found, skipping validation on best checkpoint.")

        # PART 3: Clean up latest checkpoint and shut down dataloaders. This logic is from the base class.
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # We need to gracefully shut down the dataloaders. This is critical.
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and isinstance(self.dataloader_val, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training and all custom validations are finished.")


