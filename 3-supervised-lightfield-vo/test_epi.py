from utils import print_short_summary
from configuration import Options
from epi.loader import EpiDataset, EpiDatasetOptions
from epi.loader import Resize, Normalize, SelectiveStack, MakeTensor, RandomHorizontalFlip

from torchvision.transforms import Compose
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import numpy as np



def validate_model(model, valid_dl):
	model.eval()
	loss = angular_loss = linear_loss = 0
	cfg = Options()

	with torch.no_grad():
		for batch, datapoint in enumerate(valid_dl):
			x_valid = datapoint["images"].to(cfg.device).float()
			y_valid = datapoint["poses"].to(cfg.device).float()

			l, l_theta, l_trans = model.get_loss(x_valid, y_valid)
			print_short_summary(l, l_theta, l_trans)
			
			loss += l
			angular_loss += l_theta
			linear_loss += l_trans

		loss /= len(valid_dl)
		angular_loss /= len(valid_dl)
		linear_loss /= len(valid_dl)

		return loss, angular_loss, linear_loss




def test_trajectory(ds_options, model, savename=None):
	vis_ds_options = ds_options

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