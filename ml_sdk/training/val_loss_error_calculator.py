from typing import Optional
from error_calculator import ErrorCalculator


class ValLossErrorCalculator(ErrorCalculator):
    def __init__(self):
        super(ValLossErrorCalculator, self).__init__()

    def get_error(self, loss, val_loss) -> Optional[float]:
        return val_loss
