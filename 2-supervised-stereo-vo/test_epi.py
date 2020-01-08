import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import numpy as np
import os
import time
from model import OdometryNet
from epi.loader import EpiDataset, EpiDatasetOptions
from epi.loader import Resize, Normalize, SelectiveStack, MakeTensor, RandomHorizontalFlip
from epi.utils import RECTIFIED_IMAGE_SHAPE
from epi.transforms import euler_to_rotm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def trajectory_from_poses(p):
	cumx, cumy, cumz = 0, 0, 0
	cumx_pred, cumy_pred, cumz_pred = 0, 0, 0

	absolute_poses = np.zeros_like(p)[:, :3]

	for i in range(0, p.shape[0]):
		row = p[i, :]
		rx, ry, rz, x, y, z = row
		cumx += x 
		cumy += y
		cumz += z
		absolute_poses[i, :] = [cumx, cumy, cumz]

	return absolute_poses
		
	

def model_test_trajectory(ds_options, model, savename=None):
	vis_ds_options = ds_options

	global figure

	with torch.no_grad():
		device = "cuda" if torch.cuda.is_available else "cpu"
		model = model.to(device)
		model.eval()

		assert vis_ds_options.with_pose == True

		vis_ds_options.backwards = False

		transforms = Compose([
			Resize(ds_options),
			Normalize(ds_options),
			SelectiveStack(ds_options),
			MakeTensor(ds_options)
		])

		vis_ds = EpiDataset("/home/joseph/Documents/epidata/smooth/valid/", transform=transforms, options=vis_ds_options, sequences=["thegang3"])

		actual = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])
		predicted = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

		for i in range(0, len(vis_ds), vis_ds_options.sequence_length):
			print("Predicting {}/{}".format(i, len(vis_ds)), end="\r")
			x = vis_ds[i]
			images = x["images"].unsqueeze(0).to(device)
			actual = torch.cat([actual, torch.FloatTensor(x["poses"])])
			prediction = model.forward(images).squeeze().cpu()
			predicted = torch.cat([predicted, prediction], dim = 0)
		
		cumx, cumy, cumz = 0, 0, 0
		cumx_pred, cumy_pred, cumz_pred = 0, 0, 0
		# fig = plt.figure()
		plt.clf()

		for i in range(0, len(actual)):
			row_actual = actual[i, :].cpu().numpy()
			rx, ry, rz, x, y, z = row_actual
			cumx += x 
			cumy += y
			cumz += z
			plt.scatter([cumx], [cumz], c='b')
			plt.ion()
			plt.pause(0.01)

			row_predicted = predicted[i, :].cpu().numpy()
			rx, ry, rz, x, y, z = row_predicted
			cumx_pred += x 
			cumy_pred += y
			cumz_pred += z
			plt.scatter([cumx_pred], [cumz_pred], c='r')
			plt.ion()
			plt.pause(0.01)

		if savename is not None:
			plt.savefig('{}.png'.format(savename))



def play_sequence(t_x):
	with torch.no_grad():
		plt.clf()
		for j in range(t_x.shape[0]):
			print(t_x.cpu().numpy().shape)
		
			img = t_x[j, :, :, :].cpu().numpy().transpose([1,2,0])
			plt.imshow(img+0.5)
			plt.ion()
			plt.pause(0.15)


def plot_sequence(t_y):
	with torch.no_grad():
		plt.clf()


def main():

	N, H, W, C = RECTIFIED_IMAGE_SHAPE

	# --------------- OPTIONS ---------------------------

	torch.manual_seed(42)
	torch.set_printoptions(precision=2)

	# use_pretrained_flownet = True

	ds_options = EpiDatasetOptions()
	ds_options.debug = False
	ds_options.with_pose = True
	ds_options.camera_array_indices = [7, 8, 9]
	ds_options.image_scale = 0.2
	ds_options.backwards = False

	epochs = 250
	batch_size = 4
	device = "cuda" if torch.cuda.is_available else "cpu"
	print("==> Using device {}".format(device))



	img_width =  int(W * ds_options.image_scale)
	img_height = int(H * ds_options.image_scale)
	img_channels = len(ds_options.camera_array_indices) * 3

	print("==> Loading weights")
	model = OdometryNet(img_channels, img_height, img_width, batchNorm=True)
	model.load_state_dict(torch.load("./models/epi.pth"))
	model.to(device)
	model.eval()


	transforms = Compose([
		Resize(ds_options),
		Normalize(ds_options),
		SelectiveStack(ds_options),
		# RandomHorizontalFlip(ds_options),
		MakeTensor(ds_options)
	])

	train_ds = EpiDataset("/home/joseph/Documents/epidata/smooth/train/", transform=transforms, options=ds_options, sequences=["thegang1"])
	# train_dl = DataLoader(train_ds, batch_size=4, pin_memory=True, shuffle=False)

	actual = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])
	predicted = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])


	with torch.no_grad():
		for i in range(0, len(train_ds), 7):
			print("==> {} loading... ".format(i), end=" ")
			x = train_ds[i]
			
			images = x["images"].unsqueeze(0).to(device)
			actual = torch.cat([actual, torch.FloatTensor(x["poses"])])

			print("inferring...")
			prediction = model.forward(images).squeeze().cpu()
			predicted = torch.cat([predicted, prediction], dim = 0)

		cumx, cumy, cumz = 0, 0, 0
		cumx_pred, cumy_pred, cumz_pred = 0, 0, 0
		fig = plt.figure()
		ax = fig.add_subplot(121, projection='3d')
		ax_flat = fig.add_subplot(122)

		for i in range(0, len(actual)):
			row_actual = actual[i, :].cpu().numpy()
			rx, ry, rz, x, y, z = row_actual
			cumx += x 
			cumy += y
			cumz += z
			ax.scatter([cumx], [cumy], [cumz], c='b')
			ax_flat.scatter([cumx], [cumz], c='b')

			row_predicted = predicted[i, :].cpu().numpy()
			rx, ry, rz, x, y, z = row_predicted
			cumx_pred += x 
			cumy_pred += y
			cumz_pred += z
			ax.scatter([cumx_pred], [cumy_pred], [cumz_pred], c='r')
			ax_flat.scatter([cumx_pred], [cumz_pred], c='r')

		plt.show()

	predicted = predicted.cpu().numpy()
	actual = actual.cpu().numpy()
	np.save("thegang1-actual.npy", actual)
	np.save("thegang1-predicted.npy", predicted)
	

if __name__ == "__main__":
	main()


