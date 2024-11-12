class TrainingEpochStats:
    def __init__(self, epoch, loss, val_loss):
        self._loss: float = loss
        self._val_loss: float = val_loss
        self._epoch: int = epoch

    def get_loss(self):
        return self._loss

    def get_val_loss(self):
        return self._val_loss

    def get_epoch(self):
        return self._epoch

    def __str__(self):
        return f'{self._epoch}__{self._loss:.4f}__{self._val_loss:.4f}'
