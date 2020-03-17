python3 train.py \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
unsupervised/focalstack/5-cameras-7-planes \
--save-path ~/Documents/thesis/checkpoints \
--focalstack \
--num-cameras 5 \
--num-planes 7 \
--sequence-length 3 \
-b2 -s1.0 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray \
