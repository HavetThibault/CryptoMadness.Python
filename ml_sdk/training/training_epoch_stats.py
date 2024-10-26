class TrainingEpochStats:
    def __init__(self, epoch, loss, val_loss):
        self.loss: float = loss
        self.val_loss: float = val_loss
        self.epoch: int = epoch

    def __str__(self):
        return f'{self.epoch}__{self.loss:.2f}__{self.val_loss:.2f}'
