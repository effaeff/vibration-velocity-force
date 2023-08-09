from sklearn.linear_model import ElasticNet
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)
from sklearn.svm import LinearSVR
import xgboost as xgb
from scipy.stats import uniform, randint

INPUT_SIZE = 3
OUTPUT_SIZE = 2

TEST_SIZE = 0.2

DATA_DIR = 'data/01_raw'
MODEL_DIR = 'models'
PLOT_DIR = 'plots'
RESULTS_DIR = 'results'

RANDOM_SEED = 1234

CV_FOLDS = 10
N_ITER_SEARCH = 100

LINEWIDTH = 1
FONTSIZE = 14
TARGET_LBLS = ['fc', 'fn']

PARAM_DICTS = [
    {'alpha': uniform(), 'l1_ratio': uniform()},
    {'C': randint(1, 100)},
    {
        'learning_rate': uniform(0.0001, 0.1),
        'max_depth': randint(2, 32),
        'subsample': uniform(0.5, 0.5),
        'n_estimators': randint(100, 1000),
        'colsample_bytree': uniform(0.4, 0.6),
        'lambda': randint(1, 100),
        'gamma': uniform()
    },
    {
        'learning_rate': uniform(0.0001, 0.1),
        'n_estimators': randint(100, 1000)
    },
    {
        'learning_rate': uniform(0.0001, 0.1),
        'n_estimators': randint(100, 1000),
        'max_depth': randint(2, 32),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(2, 11),
        'max_features': randint(1, INPUT_SIZE)
    },
    {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(2, 32),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(2, 11),
        'max_features': randint(1, INPUT_SIZE)
    }
]
REGRESSORS = [
    [ElasticNet(random_state=RANDOM_SEED, max_iter=100000) for __ in range(OUTPUT_SIZE)],
    [LinearSVR(random_state=RANDOM_SEED, max_iter=100000, dual='auto') for __ in range(OUTPUT_SIZE)],
    [xgb.XGBRegressor(objective='reg:squarederror') for __ in range(OUTPUT_SIZE)],
    [AdaBoostRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)],
    [GradientBoostingRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)],
    [RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1) for __ in range(OUTPUT_SIZE)]
]

