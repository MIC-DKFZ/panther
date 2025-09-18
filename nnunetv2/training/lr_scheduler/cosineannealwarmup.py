from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class CosineAnnealingWithWarmup(_LRScheduler):
    # A scheduler that combines a linear warmup phase with a cosine annealing decay phase.
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
        # warmup, linear ramp-up
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        
        return self.cosine_scheduler.get_lr()

    def step(self, epoch=None):
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