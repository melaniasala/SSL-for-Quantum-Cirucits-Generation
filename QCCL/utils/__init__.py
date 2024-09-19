from .train import train, train_byol
from .losses import NTXentLoss
#from .evaluation import evaluate

__all__ = ['train', 'NTXentLoss', 'train_byol']