from typing import Optional


class ErrorCalculator:
    def get_error(self, loss, val_loss) -> Optional[float]:
        raise NotImplementedError('get_error must be implemented')
