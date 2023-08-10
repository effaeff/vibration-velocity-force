"""Testing routine using trained regressor"""

import re
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from plot_utils import modify_axis
import matplotlib.ticker as mticker
from matplotlib import rc
# rc('font', family='Arial')

from config import (
    INPUT_SIZE,
    OUTPUT_SIZE,
    FONTSIZE,
    LINEWIDTH,
    PLOT_DIR,
    RESULTS_DIR,
    TARGET_LBLS
)

def test(hyperopt, test_data, test_numbers):
    errors = np.zeros((len(test_data), OUTPUT_SIZE))
    variances = np.zeros((len(test_data), OUTPUT_SIZE))

    for scenario_idx, scenario in enumerate(test_data):
        fig, axs = plt.subplots(2, 1, sharex=True)
        inp = scenario[:, :INPUT_SIZE]

        for out_idx in range(OUTPUT_SIZE):
            pred = hyperopt[out_idx].predict(inp)
            target = scenario[:, INPUT_SIZE + out_idx]

            errors[scenario_idx, out_idx] = math.sqrt(
                mean_squared_error(target, pred)
            ) / np.ptp(target) * 100.0
            variances[scenario_idx, out_idx] = np.std(
                [
                    abs(target[idx] - pred[idx]) / np.ptp(target)
                    for idx in range(len(target))
                ]
            ) * 100.0

            axs[out_idx].plot(target, linewidth=LINEWIDTH, label='Target')
            axs[out_idx].plot(pred, linewidth=LINEWIDTH, label='Prediction')
            axs[out_idx].set_title(TARGET_LBLS[out_idx], fontsize=FONTSIZE)
            axs[out_idx].set_xlabel('Sample', fontsize=FONTSIZE)
            axs[out_idx].set_ylabel('Force', fontsize=FONTSIZE)

        axs[0].legend(
            bbox_to_anchor=(0., 1.02, 1., .102),
            loc='lower left',
            # ncol=2,
            # mode="expand",
            fontsize=FONTSIZE,
            borderaxespad=0.,
            frameon=False
        )

        fig.canvas.draw()

        for idx in range(OUTPUT_SIZE):
            axs[idx] = modify_axis(axs[idx], '', 'N', -2, -2, FONTSIZE-2)

        fig.align_ylabels(axs)

        plt.tight_layout()
        plt.savefig(
            f'{PLOT_DIR}/{hyperopt[0].best_estimator_.__class__.__name__}_CA_{test_numbers[scenario_idx]}.png',
            dpi=600
        )
        plt.close()

    return np.mean(errors, axis=0), np.mean(variances, axis=0)
