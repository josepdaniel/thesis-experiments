# Unsupervised learning of depth and visual odometry from light fields

## Training

When a training script is run, outputs are generated in ```~/Documents/checkpoints/EXPERIMENT_NAME```. The outputs are: 
- ```dispnet_best.pth.tar```: best scoring dispnet model
- ```posenet_best.pth.tar```: best scoring posenet model
- ```config.txt``` and ```config.pkl```: human and machine readable configuration files containing parameters used for training. ```config.pkl``` is loaded during inference to initialise the networks with the same parameters.
- Progress logs in csv format
- Some tensorboard logs for visualisation


#### Monocular
```
sh training-scripts/train-monocular.sh
```

#### Trinocular
```
sh training-scripts/train-trinocular.sh
```

#### Horizontal Imagers
```
sh training-scripts/train-horizontal.sh
```


## Inference

Inference requires a 'config.pkl' file that was generated during training. This file contains the training parameters such as which cameras were used, whether images are grayscale, etc.

Inference generates both disparity maps and pose predictions.  Disparity maps are saved as pngs in the the same directory as the model weights, in a new directory with the same name as the sequence. Inside there will also be a 'poses.npy' pickling of a numpy array with columns [tx, ty, tz, rx, ry, rz]. 

```
python3 infer.py --config PATH/TO/config.pkl --seq SEQUENCE_NAME
```


## Evaluation

Use evaluation.ipynb for evaluating.