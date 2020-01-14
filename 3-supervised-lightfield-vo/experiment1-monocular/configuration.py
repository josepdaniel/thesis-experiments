import sys 
sys.path.insert(0, "../")

from torchvision.transforms import Compose
from torch import cuda
from epi.loader import EpiDatasetOptions
from epi.loader import Resize, Normalize, SelectiveStack, MakeTensor, RandomHorizontalFlip, EpipolarSlice
from epi.utils import RECTIFIED_IMAGE_SHAPE

N, H, W, C = RECTIFIED_IMAGE_SHAPE

def get_config():
    return Options()




class Options():
    def __init__(self):
        
        # For reproducibility when using RNG
        self.seed = 42

        # Training Options
        self.resume = True
        self.resume_checkpoint = "./models/epi-monocular.pth"
        self.save_name = "./models/epi-monocular.pth"
        self.training_data = "/home/joseph/Documents/epidata/smooth/train"
        self.validation_data = "/home/joseph/Documents/epidata/smooth/valid"

        # Training hyperparameters
        self.max_epochs = 1000
        self.batch_size = 4
        self.learning_rate = 0.001
        
        # Summary frequencies
        self.plot_trajectory_every = 50    

        # Dataloading Options
        self.ds_options = EpiDatasetOptions()
        self.ds_options.debug = False 
        self.ds_options.with_pose = True 
        self.ds_options.camera_array_indices = [8]
        self.ds_options.image_scale = 0.2
        self.ds_options.grayscale = False

        # Dataloading augmentations
        self.preprocessing = Compose([
            Resize(self.ds_options),
            Normalize(self.ds_options),
            SelectiveStack(self.ds_options)
        ])

        self.augmentation = None 
        
        # Don't change these
        self.input_width =  int(W * self.ds_options.image_scale)
        self.input_height = int(H * self.ds_options.image_scale)

        # These depend on the input format to the network
        self.img_channels = 1 if self.ds_options.grayscale else 3
        self.input_channels = self.img_channels * len(self.ds_options.camera_array_indices)

        self.device = "cuda" if cuda.is_available() else "cpu"





