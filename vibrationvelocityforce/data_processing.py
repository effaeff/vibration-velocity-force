"""Data processing methods"""

import os
import numpy as np

from config import DATA_DIR, TEST_SIZE, RANDOM_SEED
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DataProcessing:
    def __init__(self):
        data = self.read_raw()

        self.train, self.test = train_test_split(
            data,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED
        )

        self.train = np.reshape(self.train, (-1, self.train.shape[-1]))

    def read_raw(self):
        fnames = [fname for fname in os.listdir(DATA_DIR) if fname.endswith('odb.txt')]
        data = [
            np.loadtxt(f'{DATA_DIR}/{fname}', delimiter='\t', skiprows=2) for fname in fnames
        ]
        for idx, scenario in enumerate(data):
            cutoff = np.argmax(scenario[:len(scenario)//2, 1])
            # print(cutoff)
            # print(fnames[idx])
            # __, axs = plt.subplots(5, 1, sharex=True)
            # axs[0].plot(scenario[cutoff:, 0], scenario[cutoff:, 1])
            # axs[1].plot(scenario[cutoff:, 0], scenario[cutoff:, 2])
            # axs[2].plot(scenario[cutoff:, 0], scenario[cutoff:, 3])
            # axs[3].plot(scenario[cutoff:, 0], scenario[cutoff:, 4])
            # axs[4].plot(scenario[cutoff:, 0], scenario[cutoff:, 5])
            # plt.tight_layout()
            # plt.show()
            # plt.close()

            data[idx] = scenario[cutoff:, :]

        data = np.array(data)[:, :, 1:]
        # data = data.reshape((-1, data.shape[-1]))

        return data

    def get_train_test(self):
        return self.train, self.test
