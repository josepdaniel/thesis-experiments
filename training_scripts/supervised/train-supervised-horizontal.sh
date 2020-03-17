cd .. & python3 train_supervised.py \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
horizontal-supervised \
--save-path ~/Documents/thesis/checkpoints \
--cameras 4 5 6 7 8 9 10 11 12 \
--sequence-length 3 \
-b2 -s1.0 -m0.0 \
--epochs 300 \
--log-output \
--gray \


