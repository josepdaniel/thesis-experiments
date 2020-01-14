""""
Script that generates predicted and actual trajectory files in ./results. This script needs to be run
prior to using the evaluation.ipynb notebook. Do NOT change in between different experiments
"""


import sys
sys.path.insert(0, "../")


from evaluation import get_predicted_trajectory, get_actual_trajectory
from configuration import get_config
import matplotlib.pyplot as plt


if __name__ == "__main__":

    cfg = get_config()
    # predict_trajectory_every_frame(cfg)
    predicted = get_predicted_trajectory(force_recalculate=False)
    actual = get_actual_trajectory()

    predicted_xs, predicted_ys, predicted_zs = predicted[:, 0], predicted[:, 1], predicted[:, 2]
    actual_xs, actual_ys, actual_zs = actual[:, 0], actual[:, 1], actual[:, 2]

    print(actual.shape)
    print(predicted.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(predicted_xs*2, predicted_zs*1.5)
    ax.plot(actual_xs, actual_zs)
    plt.show()

