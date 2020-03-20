python3 train.py stack \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
unsupervised/stack/sutv \
--save-path ~/Documents/thesis/checkpoints \
--cameras  2 3 8 13 14 6 7 8 9 10 \
--sequence-length 3 \
-b2 -s1.0 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray