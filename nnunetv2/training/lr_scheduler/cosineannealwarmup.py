from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    A scheduler that combines a linear warmup phase with a cosine annealing decay phase.
    """
    def __init__(self, optimizer, warmup_steps: int, max_steps: int,
                 eta_min: float = 0.0, last_epoch: int = -1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of epochs for the linear warmup.
            max_steps (int): Total number of epochs.
            eta_min (float): Minimum learning rate.
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Initialize the cosine scheduler to run for the steps AFTER the warmup
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max_steps - warmup_steps, eta_min=eta_min
        )
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculates the learning rate for the current epoch."""
        # warmup, linear ramp-up
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        
        return self.cosine_scheduler.get_lr()

    def step(self, epoch=None):
        """
        Updates the scheduler's state. Call this once per epoch.
        """
        # internal epoch counter
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.last_epoch < self.warmup_steps:
            for i, lr in enumerate(self.get_lr()):
                self.optimizer.param_groups[i]['lr'] = lr
        else:
            # will automatically update the optimizer's lr
            self.cosine_scheduler.step()

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class PolyLRScheduler_lin_warmup(_LRScheduler):
    def __init__(self, optimizer, max_lr: float, warmup_steps: int, max_steps: int,
                 exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_steps:
            new_lr = self.max_lr / self.warmup_steps * (1 + current_step)
        else:
            new_lr = self.max_lr * (1 - (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr