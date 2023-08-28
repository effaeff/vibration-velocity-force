"""Script for learning forces based on chip thickness and vibration velocity"""

import misc
import numpy as np
from joblib import dump, load

from vibrationvelocityforce.data_processing import DataProcessing
from vibrationvelocityforce.train import train
from vibrationvelocityforce.test import test

from config import OUTPUT_SIZE, MODEL_DIR, RESULTS_DIR, REGRESSORS, PLOT_DIR

def write_results(hyperopts, errors, variances):
    """Write results to file"""
    with open(f'{RESULTS_DIR}/results.txt', 'w', encoding='utf-8') as res_file:
        res_file.write(
            f"{'Regressor':<40} {'RMSE fc':<35} RMSE fn\n"
        )

        for hyper_idx, hyperopt in enumerate(hyperopts):
            res_file.write(
                f"{hyperopt[0].best_estimator_.__class__.__name__:<40} "
                f"{f'{errors[hyper_idx, 0]:.2f} +/- {variances[hyper_idx, 0]:.2f}':<35} "
                f"{f'{errors[hyper_idx, 1]:.2f} +/- {variances[hyper_idx, 1]:.2f}'}\n"
            )

def load_estimators(directory):
    """Load already trained hyperopt objects"""
    hyperopts = np.empty((len(REGRESSORS), OUTPUT_SIZE), dtype=object)
    for idx, __ in enumerate(hyperopts):
        hyperopts[idx] = load(
            f'{directory}/hyperopt_{REGRESSORS[idx][0].__class__.__name__}.joblib'
        )
    return hyperopts

def main():
    """Main method"""
    misc.gen_dirs([MODEL_DIR, RESULTS_DIR, PLOT_DIR])

    processing = DataProcessing()
    train_data, train_numbers, test_data, test_numbers = processing.get_train_test()
    scaler = processing.get_scaler()

    hyperopts = train(train_data)
    # hyperopts = load_estimators(MODEL_DIR)
    total_errors = np.empty((len(hyperopts), OUTPUT_SIZE))
    total_variances = np.empty((len(hyperopts), OUTPUT_SIZE))
    for hyper_idx, hyperopt in enumerate(hyperopts):
        dump(
            hyperopt,
            f'{MODEL_DIR}/hyperopt_{hyperopt[0].best_estimator_.__class__.__name__}.joblib'
        )
        errors, variances = test(hyperopt, test_data, test_numbers, scaler)
        total_errors[hyper_idx] = errors
        total_variances[hyper_idx] = variances
    write_results(hyperopts, total_errors, total_variances)

if __name__ == '__main__':
    main()
