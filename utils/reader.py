from utils.preprocess import hip_preprocess, wrist_preprocess
from typing import Union
from pathlib import Path
from enum import Enum

class accelType(Enum):
    HIP = 1
    WRIST = 2

class reader:
    def __init__(self,
                 path: Union[str, Path],
                 epoch: Union[int, str],
                 type: accelType,
                 sample_frequency: Union[int, str]
                 ):

        if (type == accelType.HIP):
            self.signal = hip_preprocess(path, epoch, sample_frequency)
        elif (type == accelType.WRIST):
            self.signal = wrist_preprocess(path, epoch, sample_frequency)
        