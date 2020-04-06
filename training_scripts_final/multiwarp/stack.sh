python3 train_multiwarp.py stack \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
final/multiwarp/stack/5 \
--save-path ~/Documents/thesis/checkpoints \
-c 3 8 13 7 9 \
--sequence-length 3 \
-b2 -s1.0 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray