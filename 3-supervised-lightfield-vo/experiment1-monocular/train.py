import sys
sys.path.insert(0, "../")
from train_epi import train
from configuration import get_config 

cfg = get_config()
train(cfg)