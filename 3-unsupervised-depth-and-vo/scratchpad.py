import custom_transforms
from datasets.sequence_folders import SequenceFolder



normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
train_transform = custom_transforms.Compose([
    custom_transforms.RandomHorizontalFlip(),
    custom_transforms.RandomScaleCrop(),
    custom_transforms.ArrayToTensor(),
    normalize
])

train_set = SequenceFolder(
    "/home/joseph/KITTI_processed",
    transform=train_transform,
    seed=42,
    train=True,
    sequence_length=3
)

x = train_set[0]
print(len(x))
print(type(x[0]))