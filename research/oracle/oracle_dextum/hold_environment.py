"""Hold observation adapter, loaded only with the qualified hold release."""
from pass1e.hold import Hold
from .environment import OracleObservationMixin


class OracleHold(OracleObservationMixin, Hold):
    def __init__(self, oracle_arm, bank):
        if bank is None:
            raise ValueError('O1 hold requires a hash-qualified entry bank for its model')
        super().__init__(bank=bank)
        self.configure_oracle(oracle_arm, 'hold')
