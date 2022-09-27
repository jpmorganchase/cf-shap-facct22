# %%

from emutils.imports import *

from sklearn.model_selection import train_test_split
from emutils.datasets import load_dataset

# Suppress warnings
import warnings
# warnings.filterwarnings(action="error", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Various imports
from emutils.utils import (
    attrdict,
    pandas_max,
    load_pickle,
    save_pickle,
)
# from emutils.preprocessing import MultiScaler

from constants import DATA_DIR

# Suppress scientific notation
# np.set_printoptions(suppress=True)
np.seterr(all='raise')

# PANDAS columns rows limit
pandas_max(100, 200)

print(sys.argv)
parser = ArgumentParser(sys.argv)
parser.add_argument('--dataset', type=str, default='heloc', choices=['heloc', 'lendingclub', 'wines'], required=False)
parser.add_argument('--data_version', type=str, default="v2")
parser.add_argument('--data_path', type=str, default=DATA_DIR, required=False)
parser.add_argument('--random_state', type=int, default=2021, required=False)
parser.add_argument('--split_test', type=float, default=.3, required=False)
# parser.add_argument('--scaling', type=str, default=None, choices=['minmax', 'std', 'quantile', 'mad'], required=False)
parser.add_argument('--override', dest='override', action='store_true', default=False)

args, unknown = parser.parse_known_args()
args = attrdict(vars(args))

print(args)

os.makedirs(args.data_path, exist_ok=True)

# %%
# FILE NAMES
# -> Data
DATA_RUN_NAME = f"{args.dataset}_D{args.data_version}"

FEATURES_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_features.pkl"
CLASSES_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_classes.pkl"
TRENDS_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_trends.pkl"
REFPOINT_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_ref.pkl"
MEDIAN_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_med.pkl"
MEDIANGOOD_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_medgood.pkl"
MEAN_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_mean.pkl"
MEANGOOD_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_meangood.pkl"
TRAIN_X_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_Xtrain.pkl"
TEST_X_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_Xtest.pkl"
TRAIN_Y_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_ytrain.pkl"
TEST_Y_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_ytest.pkl"


def save_or_load_data(X_train, X_test, y_train, y_test):
    if args.override or not os.path.exists(TRAIN_X_FILENAME):
        save_pickle(X_train, TRAIN_X_FILENAME)
        save_pickle(X_test, TEST_X_FILENAME)
        save_pickle(y_train, TRAIN_Y_FILENAME)
        save_pickle(y_test, TEST_Y_FILENAME)
        return X_train, X_test, y_train, y_test
    else:
        X_train_, X_test_, y_train_, y_test_ = (
            load_pickle(TRAIN_X_FILENAME),
            load_pickle(TEST_X_FILENAME),
            load_pickle(TRAIN_Y_FILENAME),
            load_pickle(TEST_Y_FILENAME),
        )
        assert np.all(X_train_.values == X_train.values)
        assert np.all(X_test_.values == X_test.values)
        assert np.all(y_train_.values == y_train.values)
        assert np.all(y_test_.values == y_test.values)
        return X_train_, X_test_, y_train_, y_test_


# %% [markdown]
# # Load the datasets
# %%
if 'heloc' in args.dataset:
    dataset = load_dataset('heloc', as_frame=True)
    df = dataset.frame
    TARGET = dataset.target_name
    class_names = dataset.class_names
    drop_cols = [TARGET]

elif 'lendingclub' in args.dataset:
    dataset = load_dataset('lendingclub', cleaning_type='ax', as_frame=True)
    df = dataset.frame
    TARGET = dataset.target_name
    ISSUE_DATE = dataset.split_date
    class_names = dataset.class_names
    drop_cols = [TARGET, ISSUE_DATE]

elif 'wines' in args.dataset:
    dataset = dt = load_dataset('wines', binary_target_threshold=5.5, as_frame=True)
    df = dataset.frame
    TARGET = dataset.target_name
    class_names = dataset.class_names
    drop_cols = [TARGET]
else:
    raise ValueError('Invalid dataset')

print('Class names', class_names)

# %%

X, y = df.drop(columns=drop_cols), df[[TARGET]]

# Split train-test set
if 'lendingclub' in args.dataset:
    # Sort by date
    sorted_index = df.sort_values([ISSUE_DATE]).index.values
    X = X.loc[sorted_index]
    y = y.loc[sorted_index]

    # Calculate the splits and shuffle them
    split_index = int((1 - args.split_test) * len(X))
    train_split = np.arange(0, split_index)
    test_split = np.arange(split_index, len(X))
    np.random.RandomState(args.random_state).shuffle(train_split)
    np.random.RandomState(args.random_state + 1).shuffle(test_split)

    # Apply splits
    X_train, X_test = X.iloc[train_split], X.iloc[test_split]
    y_train, y_test = y.iloc[train_split], y.iloc[test_split]

else:
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=args.split_test,
                                                        shuffle=True,
                                                        random_state=args.random_state)

print(X.shape, y.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Reset splits indexes
for a_ in [X_train, X_test, y_train, y_test]:
    a_.reset_index(inplace=True, drop=True)

# multiscaler = MultiScaler(X_train)

# X_train = pd.DataFrame(multiscaler.transform(X_train, args.scaling), columns=X_train.columns.values)
# X_test = pd.DataFrame(multiscaler.transform(X_test, args.scaling), columns=X_test.columns.values)
# X = pd.DataFrame(multiscaler.transform(X, args.scaling), columns=X.columns.values)

# %%
X_train, X_test, y_train, y_test = save_or_load_data(X_train, X_test, y_train, y_test)
# %%
# Feature names
feature_names = X.columns.values

# Trends
feature_trends = [
    int(np.heaviside(scipy.stats.spearmanr(X[col].values, y.values.flatten()).correlation, 0) * 2 - 1) for col in X
]

assert np.all(np.array(feature_trends) != 0)
# %%

# median_good = np.median(X[y[TARGET] == 0].values, axis=0)
# median = np.median(X.values, axis=0)

# mean_good = np.mean(X[y[TARGET] == 0].values, axis=0)
# mean = np.mean(X.values, axis=0)

# pd.DataFrame([median_good, median, mean_good, mean], index=['Median Good', 'Median', 'Mean Good', 'Mean'])

# %%
# save_pickle(median_good, MEDIANGOOD_FILENAME)
# save_pickle(median, MEDIAN_FILENAME)
# save_pickle(mean_good, MEANGOOD_FILENAME)
# save_pickle(mean, MEAN_FILENAME)

save_pickle(feature_trends, TRENDS_FILENAME)
save_pickle(feature_names, FEATURES_FILENAME)
save_pickle(class_names, CLASSES_FILENAME)
