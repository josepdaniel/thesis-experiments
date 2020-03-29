rm -rf ~/Documents/thesis/checkpoints/multiwarp/focalstack/test && python3 train_multiwarp.py focalstack \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
multiwarp/focalstack/13-cams-5-planes \
--save-path ~/Documents/thesis/checkpoints \
--num-cameras 13 \
--num-planes 5 \
-c  3 8 13 7 9 \
--sequence-length 3 \
-b2 -s1.0 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray
