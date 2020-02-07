cd .. & python3 train.py \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
horizontal \
--save-path ~/Documents/thesis/checkpoints \
--cameras 7 8 9 \
--sequence-length 3 \
-b2 -s1.0 -m0.0 \
--epochs 500 \
--log-output \
--gray \


