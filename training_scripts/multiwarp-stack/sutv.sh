python3 train_multiwarp.py stack \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
multiwarp/stack/test \
--save-path ~/Documents/thesis/checkpoints \
-c 6 7 8 9 10 2 3 13 14 \
--sequence-length 3 \
-b2 -s1.0 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray