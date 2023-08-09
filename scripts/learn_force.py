"""Script for learning forces based on chip thickness and vibration velocity"""

import numpy as np
from joblib import dump, load

from vibrationvelocityforce.data_processing import DataProcessing
from vibrationvelocityforce.train import train
from vibrationvelocityforce.test import test

from config import OUTPUT_SIZE, MODEL_DIR, RESULTS_DIR

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

def main():
    """Main method"""

    processing = DataProcessing()
    train_data, test_data = processing.get_train_test()

    hyperopts = train(train_data)
    total_errors = np.empty((len(hyperopts), OUTPUT_SIZE))
    total_variances = np.empty((len(hyperopts), OUTPUT_SIZE))
    for hyper_idx, hyperopt in enumerate(hyperopts):
        dump(
            hyperopt,
            f'{MODEL_DIR}/hyperopt_{hyperopt[0].best_estimator_.__class__.__name__}.joblib'
        )
        errors, variances = test(hyperopt, test_data)
        total_errors[hyper_idx] = errors
        total_variances[hyper_idx] = variances
    write_results(hyperopts, total_errors, total_variances)

if __name__ == '__main__':
    main()
