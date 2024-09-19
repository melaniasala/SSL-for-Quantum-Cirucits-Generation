from .train import train, train_byol
from .losses import NTXentLoss
from .HyperparamTuner import HyperparamTuner
#from .evaluation import evaluate

__all__ = ['train', 'NTXentLoss', 'train_byol', 'HyperparamTuner']