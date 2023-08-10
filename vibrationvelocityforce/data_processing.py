"""Data processing methods"""

import os
import numpy as np

from config import DATA_DIR, TEST_SIZE, RANDOM_SEED
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DataProcessing:
    def __init__(self):
        data, sim_numbers = self.read_raw()


        data_numbers = list(zip(data, sim_numbers))

        self.train_numbers_combined, self.test_numbers_combined = train_test_split(
            data_numbers,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED
        )

        self.train, self.train_numbers = zip(*self.train_numbers_combined)
        self.test, self.test_numbers = zip(*self.test_numbers_combined)

        self.train = np.array(self.train)
        self.test = np.array(self.test)

        self.train = np.reshape(self.train, (-1, self.train.shape[-1]))

    def read_raw(self):
        fnames = [fname for fname in os.listdir(DATA_DIR) if fname.endswith('odb.txt')]
        data = [
            np.loadtxt(f'{DATA_DIR}/{fname}', delimiter='\t', skiprows=2) for fname in fnames
        ]
        sim_numbers = [int(fname.split('_')[1]) for fname in fnames]
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

        return data, sim_numbers

    def get_train_test(self):
        return self.train, self.train_numbers, self.test, self.test_numbers
