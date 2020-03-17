cd .. & python3 train_supervised.py \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
monocular-supervised \
--save-path ~/Documents/thesis/checkpoints \
--cameras 8 \
--sequence-length 3 \
-b2 -s1.0 -m0.0 \
--epochs 300 \
--log-output \
--gray \


