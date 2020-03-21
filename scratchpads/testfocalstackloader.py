import sys 
sys.path.insert(0, "..")
from multiwarp_dataloader import StackedLFLoader
import custom_transforms

train_transform = custom_transforms.Compose([
    custom_transforms.ArrayToTensor(),
    custom_transforms.Normalize(mean=0.4, std=0.5)
])

loader = StackedLFLoader(
    root = "/home/joseph/Documents/thesis/epidata/module-1-1/module1-1-png", 
    cameras = list(range(0, 17)),
    gray=True, 
    sequence="seq3", 
    intrinsics_file="../intrinsics.txt",
    transform=train_transform,
    shuffle=False
)
lf1, lf2, k, kinv, pose = loader[0]

import matplotlib.pyplot as plt 

fig = plt.figure()
for i in range(0, 16):
    ax = fig.add_subplot(4,4,i+1)
    ax.imshow(lf1[i, :, :])
plt.show()

print(k, kinv)