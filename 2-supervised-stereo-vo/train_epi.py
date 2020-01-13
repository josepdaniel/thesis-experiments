import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import numpy as np
import os
from model import OdometryNet
from epi.loader import EpiDataset, EpiDatasetOptions
from epi.loader import Resize, Normalize, SelectiveStack, MakeTensor, RandomHorizontalFlip
from epi.utils import RECTIFIED_IMAGE_SHAPE
from test_epi import model_test_trajectory
import matplotlib.pyplot as plt

N, H, W, C = RECTIFIED_IMAGE_SHAPE


# --------------- OPTIONS ---------------------------
torch.manual_seed(42)

use_pretrained_flownet = False
pretrained_flownet_path = "./pretrained/flownets_EPE1.951.pth.tar"

resume = True
resume_checkpoint = "./models/epi.pth"
model_savename = 	"./models/epi.pth"

train_path = "/home/joseph/Documents/epidata/smooth/train"
valid_path = "/home/joseph/Documents/epidata/smooth/valid"

ds_options = EpiDatasetOptions()
ds_options.debug = False
ds_options.with_pose = True
ds_options.camera_array_indices = [7, 8, 9]
ds_options.image_scale = 0.2

epochs = 1000
batch_size = 4
plot_trajectory_every = 50

img_width =  int(W * ds_options.image_scale)
img_height = int(H * ds_options.image_scale)
img_channels = len(ds_options.camera_array_indices) * 3			# Assuming RGB Images, change to *1 for grayscale

# ---------------------------------------------------

def play_sequence(t_x):
	with torch.no_grad():
		# print(t_x.shape)
		for i in range(t_x.shape[0]):
			plt.clf()
			for j in range(t_x.shape[1]):
				img = t_x[i, j, :, :, :].cpu().numpy().transpose([1,2,0])
				plt.imshow(img+0.5)
				plt.ion()
				plt.pause(0.15)

		# t_x = t_x.cpu().numpy().transpose([1,2,0])
		# print(t_x).shape

# ---------------------------------------------------



transforms = Compose([
	Resize(ds_options),
	Normalize(ds_options),
	SelectiveStack(ds_options),
	# RandomHorizontalFlip(ds_options),
	MakeTensor(ds_options)
])

train_ds = EpiDataset(train_path, transform=transforms, options=ds_options)
train_dl = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, shuffle=True)
print("==> Created training dataloader with {} batches, {} samples".format(len(train_dl), len(train_ds)))

valid_ds = EpiDataset(valid_path, transform=transforms, options=ds_options)
valid_dl = DataLoader(valid_ds, batch_size=4, pin_memory=True, shuffle=True)
print("==> Created validation dataloader with {} batches, {} samples".format(len(valid_dl), len(valid_ds)))


# Model
model = OdometryNet(img_channels, img_height, img_width, batchNorm=True)

print('==> Using {} channel images'.format(img_channels))

use_cuda = torch.cuda.is_available()

if use_cuda:
	print('==> Using CUDA')
	model = model.cuda()


if resume:
	print("==> Loading checkpoint {}".format(resume_checkpoint))
	model.load_state_dict(torch.load(resume_checkpoint))


if use_pretrained_flownet:
	print('==> Loading pretrained FlowNet model')
	if use_cuda:
		pretrained_w = torch.load(pretrained_flownet_path)
	else:
		pretrained_w = torch.load(pretrained_flownet_path, map_location='cpu')

 	# Use only conv-layer-part of FlowNet as CNN for DeepVO
	model_dict = model.state_dict()
	update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
	model_dict.update(update_dict)
	model.load_state_dict(model_dict)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train
best_loss = 1e10
validation_loss_history = []
validation_translation_loss_history = []
validation_rotation_loss_history = []

for ep in range(epochs):
	model.train()

	# Training Step -----------------------------------------------------------------
	for batch, datapoint in enumerate(train_dl):

		# Plot Trajectory -----------------------------------------------------------------
		if batch % plot_trajectory_every == 0:
			print("==> Predicting trajectory")
			model_test_trajectory(ds_options, model, savename="progress/"+str(str(ep)+"-"+str(batch)))

		model.train()
		t_x = datapoint["images"]
		t_y = datapoint["poses"]

		# play_sequence(t_x)			# Uncomment to play the sequence as a video on each training step

		if use_cuda:
			t_x = t_x.cuda(non_blocking=True).float()
			t_y = t_y.cuda(non_blocking=True).float()

		ls, angle_loss, translation_loss = model.step(t_x, t_y, optimizer)
		
		print("Epoch: 	{}/{}, Batch:   {}/{}, Loss    {:.3f}	(Angular: {:.3f}, Linear: {:.3f})".format(
			ep, epochs, batch, len(train_dl), ls, angle_loss, translation_loss)
		)

		

 	# Validation -----------------------------------------------------------------
	print("EVALUATING " + "-"*50)
	model.eval()
	eval_loss = eval_angle_loss = eval_translation_loss = 0
	with torch.no_grad():
		for batch, datapoint in enumerate(valid_dl):
			t_x_valid = datapoint["images"]
			t_y_valid = datapoint["poses"]

			if use_cuda:
				t_x_valid = t_x_valid.cuda(non_blocking=True).float()
				t_y_valid = t_y_valid.cuda(non_blocking=True).float()

			loss, angle_loss, translation_loss = model.get_loss(t_x_valid, t_y_valid)
			
			print("Epoch: 	{}/{}, Batch:   {}/{}, Loss    {:.3f}	(Angular: {:.3f}, Linear: {:.3f})".format(
				ep, epochs, batch, len(valid_dl), loss, angle_loss, translation_loss)
			)
			eval_loss += loss
			eval_angle_loss += angle_loss
			eval_translation_loss += translation_loss

		eval_loss /= len(valid_dl)
		eval_angle_loss /= len(valid_dl)
		eval_translation_loss /= len(valid_dl)

		print("-" * 61)

		print("EVALUATION SUMMARY")
		print("Loss    {:.3f}	(Angular: {:.3f}, Linear: {:.3f})".format(eval_loss, eval_angle_loss, eval_translation_loss) )

		if eval_loss < best_loss: 
			torch.save(model.state_dict(), "models/epi.pth")
			print("Saved model as \"epi.pth\"")


		validation_loss_history.append(eval_loss)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(validation_loss_history)
		plt.show()

		print("-" * 61)

