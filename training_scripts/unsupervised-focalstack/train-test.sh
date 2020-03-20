python3 train.py focalstack \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
unsupervised/focalstack/test \
--save-path ~/Documents/thesis/checkpoints \
--num-cameras 5 \
--num-planes 3 \
--sequence-length 3 \
-b2 -s1.0 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray 