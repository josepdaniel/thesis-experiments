import sys 
sys.path.insert(0, "../")

import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from model import OdometryNet
from configuration import Options
from epi.loader import EpiDataset
from epi.utils import RECTIFIED_IMAGE_SHAPE
from test_epi import test_trajectory, validate_model
from utils import print_summary, print_short_summary, print_hrule


def train(cfg):
	N, H, W, C = RECTIFIED_IMAGE_SHAPE
	torch.manual_seed(cfg.seed)

	train_ds = EpiDataset(cfg.training_data, preprocessing=cfg.preprocessing, augmentation=cfg.augmentation, options=cfg.ds_options)
	train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, pin_memory=True, shuffle=True)
	print("==> Created training dataloader with {} batches, {} samples".format(len(train_dl), len(train_ds)))

	valid_ds = EpiDataset(cfg.validation_data, preprocessing=cfg.preprocessing, augmentation=None, options=cfg.ds_options)
	valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, pin_memory=True, shuffle=True)
	print("==> Created validation dataloader with {} batches, {} samples".format(len(valid_dl), len(valid_ds)))

	# Model
	print("==> Creating OdometryNet")
	model = OdometryNet(cfg.input_channels, cfg.input_height, cfg.input_width, batchNorm=True)
	print('==> Using {} channel images on {}'.format(cfg.input_channels, cfg.device))
	model = model.to(cfg.device)

	ep_start = 0

	if cfg.resume:
		print("==> Loading weights {}".format(cfg.resume_checkpoint))
		checkpoint = torch.load(cfg.resume_checkpoint)
		ep_start = checkpoint["epochs"]+1
		model.load_state_dict(checkpoint["model"])

	# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
	optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

	# Train
	best_loss = 1e10
	validation_loss_history = []


	for ep in range(ep_start, cfg.max_epochs):
		model.train()

		# Training Step -----------------------------------------------------------------
		for batch, datapoint in enumerate(train_dl):

			# Plot Trajectory -----------------------------------------------------------
			if batch % cfg.plot_trajectory_every == 0:
				print("==> Predicting trajectory")
				savename = f"./progress/{ep}_{batch}"
				test_trajectory(cfg, model, savename=savename)

			model.train()
			x = datapoint["images"]
			y = datapoint["poses"]

			x = x.to(cfg.device, non_blocking=True).float()
			y = y.to(cfg.device, non_blocking=True).float()

			loss, angle_loss, translation_loss = model.step(x, y, optimizer)
			print_summary(ep, cfg.max_epochs, batch, len(train_dl), loss, angle_loss, translation_loss)
			

		# Validation -----------------------------------------------------------------
		print("EVALUATING " + "-"*50)

		loss, angular_loss, linear_loss = validate_model(model, valid_dl)

		print_hrule()
		print("EVALUATION SUMMARY")
		print_short_summary(loss, angular_loss, linear_loss)
		print_hrule()
		
		if loss < best_loss: 
			snapshot = {"model": model.state_dict(), "epochs":ep, "loss":loss}
			torch.save(snapshot, cfg.save_name)
			print("Saved model as {}".format(cfg.save_name))
			best_loss = loss
	

