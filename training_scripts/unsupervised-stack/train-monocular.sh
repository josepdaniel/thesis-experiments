python3 train.py stack \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
unsupervised/stack/monocular \
--save-path ~/Documents/thesis/checkpoints \
--cameras 8 \
--sequence-length 3 \
-b2 -s1.0 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray