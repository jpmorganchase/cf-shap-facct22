# %%
import warnings
from emutils.imports import *
from sklearn.metrics import roc_curve
from emutils.utils import (
    attrdict,
    pandas_max,
)
from emutils.file import (
    save_json,
    load_json,
    load_pickle,
    save_pickle,
)

from emutils.model.train import train_model
from emutils.model.tune import find_best_curve_threshold_by_derivative, find_best_curve_threshold_by_sum

from utils import *

# Suppress warnings
# warnings.filterwarnings(action="error", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# PANDAS columns rows limit
pandas_max(100, 200)

# Suppress scientific notation
# np.set_printoptions(suppress=True)
np.seterr(all='raise')

# %%
from constants import DATA_DIR, MODEL_DIR

print(sys.argv)
parser = ArgumentParser(sys.argv)

# General
parser.add_argument('--dataset', type=str, default='wines', choices=['heloc', 'lendingclub', 'wines'], required=False)
parser.add_argument('--data_version', default='v2', type=str, required=False)
parser.add_argument('--data_path', type=str, default=DATA_DIR, required=False)
parser.add_argument('--random_state', type=int, default=2021, required=False)

# Model
parser.add_argument('--model_path', type=str, default=MODEL_DIR)
parser.add_argument('--model_type', type=str, default='xgb')
parser.add_argument('--model_version', type=str, default='v5')

# Training
parser.add_argument('--training', action='store_true', default=False)
parser.add_argument('--override', action='store_true', default=False)
parser.add_argument('--monotonic', action='store_true', default=False)

#pylint: disable=no-member

args, unknown = parser.parse_known_args()
args = attrdict(vars(args))

os.makedirs(args.data_path, exist_ok=True)
os.makedirs(args.model_path, exist_ok=True)

print(args)

# %%
# FILE NAMES
# -> Data
DATA_RUN_NAME = f"{args.dataset}_D{args.data_version}"

FEATURES_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_features.pkl"
CLASSES_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_classes.pkl"
TRENDS_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_trends.pkl"
TRAIN_X_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_Xtrain.pkl"
TEST_X_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_Xtest.pkl"
TRAIN_Y_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_ytrain.pkl"
TEST_Y_FILENAME = f"{args.data_path}/{DATA_RUN_NAME}_ytest.pkl"


def load_data():
    X_train = load_pickle(TRAIN_X_FILENAME)
    X_test = load_pickle(TEST_X_FILENAME)
    y_train = load_pickle(TRAIN_Y_FILENAME)
    y_test = load_pickle(TEST_Y_FILENAME)
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)
    return X, y, X_train, X_test, y_train, y_test


# -> Model
MODEL_RUN_NAME = f"{DATA_RUN_NAME}M{args.model_version}_{args.model_type}"

MODELWRAPPER_FILENAME = f"{args.model_path}/{MODEL_RUN_NAME}_model.pkl"
MODEL_JSON = MODELWRAPPER_FILENAME.replace('.pkl', '.json')

# %% [markdown]
# # Load Data
# %%
X, y, X_train, X_test, y_train, y_test = load_data()
X_all, y_all = X, y

feature_names = load_pickle(FEATURES_FILENAME)
class_names = load_pickle(CLASSES_FILENAME)
feature_trends = load_pickle(TRENDS_FILENAME) if args.monotonic else None

monotone_constraints = tuple(np.array(feature_trends, dtype=int)) if feature_trends is not None else None
print(monotone_constraints)
# %% [markdown]
# # Models
# %%

if not os.path.exists(MODEL_JSON):
    raise FileNotFoundError('Model JSON not found.')

# %% [markdown]
# ## Optimal Model
# %% [markdown]
# Lets' train the optimal model
# %%
params = load_json(MODEL_JSON)

params['random_state'] = args.random_state
params['binary_threshold'] = .5

model = train_model(
    X_train,
    y_train,
    model_type=args.model_type,
    params=params,
    model_filename=MODELWRAPPER_FILENAME,
    override=args.override and args.training,
)
# %% [markdown]
# #### Threshold
# %% [markdown]
# Let's find the optimal threshold
# %%
fpr, tpr, thr = roc_curve(y_test.values.flatten(), model.predict_proba(X_test.values)[:, 1])

m = 1.3

print(f'Best Derivative threshold (m = {m})')
fallout, recall, threshold = find_best_curve_threshold_by_derivative(fpr, tpr, thr, m=m)
display(pd.DataFrame([{'Fall-out': fallout, 'Recall': recall, 'Threshold': threshold}]))

print(f'Best Sum threshold (m = {m})')
fallout, recall, threshold = find_best_curve_threshold_by_sum(fpr, tpr, thr, m=m)
display(pd.DataFrame([{'Fall-out': fallout, 'Recall': recall, 'Threshold': threshold}]))

# %% [markdown]
# Set the threshold
# %%

params['threshold'] = threshold
model.threshold = threshold

# Save model with the right threhsold
save_pickle(model, MODELWRAPPER_FILENAME)
save_json(params, MODEL_JSON)