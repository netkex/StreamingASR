from typing import List
from torch import Tensor


class ASRBaseModel:
    def __init__(self):
        pass

    def generate(self, batch: Tensor) -> List[str]:
        pass
