python3 infer.py --dataset-dir ~/Documents/epidata/module-1-1/module1-1-png/ \
--depthnet ./checkpoints/module1-1-png,500epochs,b2,s1.0/02-03-19:10/dispnet_model_best.pth.tar \
--posenet ./checkpoints/module1-1-png,500epochs,b2,s1.0/02-03-19:10/exp_pose_model_best.pth.tar \
--output-dir ./results/seq14
