import os
import sys
import torch
import shutil

from os.path import join, isfile, isdir
from nnunetv2.utilities.helpers import empty_cache
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler_lin_warmup
from nnunetv2.training.lr_scheduler.cosineannealwarmup import CosineAnnealingWithWarmup
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5, nnUNetTrainerDA5ord0, nnUNetTrainerDA5Segord0

'''
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
'''
class nnUNetTrainerDualVal(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def on_train_end(self):
        """
        This method is a complete override of the base implementation. It integrates the
        validation logic before the dataloaders are shut down.
        """
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # Perform the custom dual validation.
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

        # Clean up latest checkpoint and shut down dataloaders. This logic is from the base class.
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # Gracefully shut down the dataloaders.
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


class nnUNetTrainerDualVal_DA5(nnUNetTrainerDA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def on_train_end(self):
        """
        This method is a complete override of the base implementation. It integrates the
        validation logic before the dataloaders are shut down.
        """
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # Perform the custom dual validation.
        self.print_to_log_file("Starting automated dual-checkpoint validation...")

        # A) Evaluate the final checkpoint (the model currently loaded in memory).
        self.print_to_log_file("Running validation on 'checkpoint_final.pth'...")
        # perform_actual_validation() handles setting the network to eval mode.
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

        # Clean up latest checkpoint and shut down dataloaders. This logic is from the base class.
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # Gracefully shut down the dataloaders
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


class nnUNetTrainerDualVal_DA5ord0(nnUNetTrainerDA5ord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def on_train_end(self):
        """
        This method is a complete override of the base implementation. It integrates the
        validation logic before the dataloaders are shut down.
        """
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # Perform the custom dual validation.
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

        # Clean up latest checkpoint and shut down dataloaders. This logic is from the base class.
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # Gracefully shut down the dataloaders.
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


class nnUNetTrainerDualVal_DA5Segord0(nnUNetTrainerDA5Segord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def on_train_end(self):
        """
        This method is a complete override of the base implementation. It integrates the
        validation logic before the dataloaders are shut down.
        """
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # Perform the custom dual validation.
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

        # Clean up latest checkpoint and shut down dataloaders. This logic is from the base class.
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # Gracefully shut down the dataloaders.
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


class nnUNetTrainer1e3(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3

class nnUNetTrainer1e3_150e(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 150

class nnUNetTrainer1e3_100e(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 100

class nnUNetTrainer1e3_200e(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 200

class nnUNetTrainer1e3_250e(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 250

class nnUNetTrainer1e3_300e(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 300


class nnUNetTrainer1e4_150e(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-4
        self.num_epochs = 150

class nnUNetTrainer2e3_150e(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 2e-3
        self.num_epochs = 150

class nnUNetTrainer3e3_150e(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-3
        self.num_epochs = 150


class nnUNetTrainer1e3_100e_polylrlin(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 100

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        self.print_to_log_file("Using custom LR scheduler: PolyLRScheduler_lin_warmup")
        self.lr_scheduler = PolyLRScheduler_lin_warmup(self.optimizer, self.initial_lr, warmup_steps=50, max_steps=self.num_epochs)    
        return self.optimizer, self.lr_scheduler

class nnUNetTrainer1e3_150e_polylrlin(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 150

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        self.print_to_log_file("Using custom LR scheduler: PolyLRScheduler_lin_warmup")
        self.lr_scheduler = PolyLRScheduler_lin_warmup(self.optimizer, self.initial_lr, warmup_steps=50, max_steps=self.num_epochs)    
        return self.optimizer, self.lr_scheduler
    
class nnUNetTrainer1e3_200e_polylrlin(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 200

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        self.print_to_log_file("Using custom LR scheduler: PolyLRScheduler_lin_warmup")
        self.lr_scheduler = PolyLRScheduler_lin_warmup(self.optimizer, self.initial_lr, warmup_steps=50, max_steps=self.num_epochs)    
        return self.optimizer, self.lr_scheduler
    
class nnUNetTrainer1e3_300e_polylrlin(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 300

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        self.print_to_log_file("Using custom LR scheduler: PolyLRScheduler_lin_warmup")
        self.lr_scheduler = PolyLRScheduler_lin_warmup(self.optimizer, self.initial_lr, warmup_steps=50, max_steps=self.num_epochs)    
        return self.optimizer, self.lr_scheduler


class nnUNetTrainer1e3_1000e_polylrlin(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        self.print_to_log_file("Using custom LR scheduler: PolyLRScheduler_lin_warmup")
        self.lr_scheduler = PolyLRScheduler_lin_warmup(self.optimizer, self.initial_lr, warmup_steps=50, max_steps=self.num_epochs)    
        return self.optimizer, self.lr_scheduler
    

class nnUNetTrainer1e3_150e_cosineanneal(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 150

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        self.print_to_log_file("Using custom LR scheduler: CosineAnnealingWithWarmup with 50 warmup epochs.")

        lr_scheduler = CosineAnnealingWithWarmup(
            optimizer=optimizer,
            warmup_steps=50,
            max_steps=self.num_epochs # Will be 150
        )
        
        return optimizer, lr_scheduler
    

class nnUNetTrainer1e3_200e_cosineanneal15(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 200

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        self.print_to_log_file("Using custom LR scheduler: CosineAnnealingWithWarmup with 50 warmup epochs.")

        lr_scheduler = CosineAnnealingWithWarmup(
            optimizer=optimizer,
            warmup_steps=15,
            max_steps=self.num_epochs # Will be 150
        )
        
        return optimizer, lr_scheduler


class nnUNetTrainer1e3_1000e_cosineanneal(nnUNetTrainerDualVal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)

        self.print_to_log_file("Using custom LR scheduler: CosineAnnealingWithWarmup with 50 warmup epochs.")

        lr_scheduler = CosineAnnealingWithWarmup(
            optimizer=optimizer,
            warmup_steps=50,
            max_steps=self.num_epochs # Will be 150
        )
        
        return optimizer, lr_scheduler


class nnUNetTrainer1e3_150e_DA5(nnUNetTrainerDualVal_DA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 150


class nnUNetTrainer1e3_200e_DA5(nnUNetTrainerDualVal_DA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 200


class nnUNetTrainer1e3_250e_DA5(nnUNetTrainerDualVal_DA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 250


class nnUNetTrainer1e3_300e_DA5(nnUNetTrainerDualVal_DA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 300


class nnUNetTrainer1e3_50e_DA5ord0(nnUNetTrainerDualVal_DA5ord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 50


class nnUNetTrainer1e3_100e_DA5ord0(nnUNetTrainerDualVal_DA5ord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 100


class nnUNetTrainer1e3_150e_DA5ord0(nnUNetTrainerDualVal_DA5ord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 150


class nnUNetTrainer1e3_200e_DA5ord0(nnUNetTrainerDualVal_DA5ord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 200

class nnUNetTrainer1e3_100e_DA5Segord0(nnUNetTrainerDualVal_DA5Segord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 100

class nnUNetTrainer1e3_125e_DA5Segord0(nnUNetTrainerDualVal_DA5Segord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 125

class nnUNetTrainer1e3_150e_DA5Segord0(nnUNetTrainerDualVal_DA5Segord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 150

class nnUNetTrainer1e3_175e_DA5Segord0(nnUNetTrainerDualVal_DA5Segord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 175

class nnUNetTrainer1e3_200e_DA5Segord0(nnUNetTrainerDualVal_DA5Segord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 200


class nnUNetTrainer1e3_300e_DA5Segord0(nnUNetTrainerDualVal_DA5Segord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 300


class nnUNetTrainer1e3_1000e_DA5(nnUNetTrainerDualVal_DA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 1000


class nnUNetTrainer1e3_1000e_DA5ord0(nnUNetTrainerDualVal_DA5ord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 1000


class nnUNetTrainer1e3_1000e_DA5Segord0(nnUNetTrainerDualVal_DA5Segord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 1e-3
        self.num_epochs = 1000



