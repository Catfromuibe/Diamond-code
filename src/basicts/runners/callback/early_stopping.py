from typing import TYPE_CHECKING, Optional

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class EarlyStopping(BasicTSCallback):

    """
    Early stopping callback.

    Args:
        patience: epochs with no improvement before stop. Defaults to 10.
        start_after_epoch: ignore early-stop until this many epochs have finished
            (1-based). Useful when warm-up should not decide the best checkpoint.
    """

    def __init__(self, patience: int = 10, start_after_epoch: int = 0):
        self.patience = patience
        self.start_after_epoch = start_after_epoch
        self.counter: int = 0

    def on_train_start(self, runner: 'BasicTSRunner'):
        msg = f'Use early stopping with patience {self.patience}'
        if self.start_after_epoch > 0:
            msg += f', start_after_epoch={self.start_after_epoch}'
        runner.logger.info(msg + '.')

    def on_validate_end(self, runner: 'BasicTSRunner', train_step: int, train_epoch: Optional[int] = None):
        # train_epoch is 0-based in BasicTS; convert to 1-based finished count
        finished = (train_epoch + 1) if train_epoch is not None else 0
        if finished <= self.start_after_epoch:
            self.counter = 0
            return

        metric = runner.meter_pool.get_value(f'val/{runner.target_metric}')
        best_metric = runner.best_metrics.get(f'val/{runner.target_metric}')
        if best_metric is not None and (metric >= best_metric if runner.metrics_best == 'min' else metric <= best_metric):
            self.counter += 1
            if self.counter >= self.patience:
                if runner.training_unit == 'epoch':
                    runner.logger.info(f'Early stopping at epoch {train_epoch}.')
                elif runner.training_unit == 'step':
                    runner.logger.info(f'Early stopping at step {train_step}.')
                runner.should_training_stop = True
        else:
            self.counter = 0
