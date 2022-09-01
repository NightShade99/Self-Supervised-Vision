
from . import simclr 
from . import supervised

ALGOS = {
    'supervised': supervised.main,
    'simclr': simclr.main,
    
}