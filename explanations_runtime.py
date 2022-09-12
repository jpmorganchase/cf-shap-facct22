# %%
from emutils.imports import *
from emutils.utils import (
    attrdict,
    pandas_max,
    load_pickle,
    save_pickle,
)
from emutils.preprocessing import MultiScaler

from utils import *

# Suppress warnings
import warnings
# warnings.filterwarnings(action="error", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress scientific notation
# np.set_printoptions(suppress=True)
np.seterr(all='raise')

pandas_max(100, 200)

# %%
from constants import DATA_DIR, MODEL_DIR, RESULTS_DIR

parser = ArgumentParser(sys.argv)

# General
parser.add_argument('--dataset', type=str, default='moons', choices=['heloc', 'lendingclub', 'wines'], required=False)
parser.add_argument('--data_version', type=str, default="v2")
parser.add_argument('--data_path', type=str, default=DATA_DIR, required=False)
parser.add_argument('--random_state', type=int, default=2021, required=False)

# Model
parser.add_argument('--model_path', type=str, default=MODEL_DIR, required=False)
parser.add_argument('--model_type', type=str, default='xgb')
parser.add_argument('--model_version', type=str, default='v5')

# Results
parser.add_argument('--results_path', type=str, default=RESULTS_DIR, required=False)

# Experiments
parser.add_argument('--monotonic', action='store_true', default=False)
parser.add_argument('--methods', nargs='+', required=False, default=None)

args, unknown = parser.parse_known_args()
args = attrdict(vars(args))

os.makedirs(args.results_path, exist_ok=True)

print(args)
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

# -> Model
MODEL_RUN_NAME = f"{DATA_RUN_NAME}M{args.model_version}_{args.model_type}"
MODELWRAPPER_FILENAME = f"{args.model_path}/{MODEL_RUN_NAME}_model.pkl"


def profiling_filename(
    args,
    dataset=None,
    data_version=None,
    model_version=None,
    model_type=None,
    ext='pkl',
):
    if dataset is None:
        dataset = args.dataset
    if data_version is None:
        data_version = args.data_version
    if model_version is None:
        model_version = args.model_version
    if model_type is None:
        model_type = args.model_type

    MODEL_RUN_NAME = f"{args.dataset}_D{args.data_version}M{args.model_version}_{args.model_type}"

    return f"{args.results_path}/{MODEL_RUN_NAME}_profiling.{ext}"


# %% [markdown]
# # Load data and model
# Let's load all the data and the trained model.
# We load here also the manifold information.
# %%
X, _, X_train, _, y_train, _ = load_data(args)

feature_names = load_pickle(FEATURES_FILENAME)
class_names = load_pickle(CLASSES_FILENAME)
feature_trends_ = load_pickle(TRENDS_FILENAME)
feature_trends = feature_trends_ if args.monotonic else None
ref_points = dict(
    med=load_pickle(MEDIAN_FILENAME),
    medgood=load_pickle(MEDIANGOOD_FILENAME),
    meangood=load_pickle(MEANGOOD_FILENAME),
    mean=load_pickle(MEAN_FILENAME),
)

multiscaler = MultiScaler(X_train)

model = load_pickle(MODELWRAPPER_FILENAME)
preds__ = model.predict(X.values)
X_good = X[preds__ == 0]
X_bad = X[preds__ == 1]

X_explain = X_bad.values

print(f"Number of Bads : {X_bad.shape}")
# %%
from emutils.parallel.utils import max_cpu_count

# Set number of threads for efficiency.
model.get_booster().set_param({'nthread': min(15, max_cpu_count() - 1)})

# %% [markdown]
# # Explainers
# Let's set up all the explainers that we want to use
# %%
from explainers import create_explainers

# Explainers
EXPLAINERS = create_explainers(
    X=X_train,
    y=y_train,
    model=model,
    ref_points=ref_points,
    multiscaler=multiscaler,
    feature_names=feature_names,
    feature_trends=feature_trends,  # None by default
    random_state=args.random_state,
)
# %%
from emutils.profiling.time import estimate_parallel_function_linear_plateau
from emutils.profiling.time import estimate_iterative_function_runtime

model_parallelism = np.median([
    estimate_parallel_function_linear_plateau(model.booster.inplace_predict, X_train.values, start=100, precision=1e-3)
    for _ in range(10)
])

model_parallelism
# %% [markdown]
# # Test (Speed) Performance
# %%

N = 10
T = 1e-1
start = time.time()

results = []
for m, (method, explainer) in enumerate(EXPLAINERS.items()):

    # Skip if not in the list of methods to test
    if args.methods is not None and method not in args.methods:
        continue

    def __go():
        return estimate_iterative_function_runtime(
            lambda X: explainer.shap_values(X),
            X_explain,
            n=1,
            precision=T,
            concatenate=True,
        )

    ts = []
    if 'cone' in method:
        # We optimize the speed of cone (as much as possible)
        for mp in np.ceil(np.linspace(model_parallelism * .5, model_parallelism * 2, 20)).astype(int):
            tst = []
            for i in tqdm(range(N), desc=f"{method} ({m+1}/{len(EXPLAINERS)})"):
                EXPLAINERS[method].counterfactual_method.wrapping_instance.kwargs['model_parallelism'] = mp
                tst.append(__go())
            if not ts or np.min(tst) < np.min(ts):
                ts = tst
    else:
        for i in tqdm(range(N), desc=f"{method} ({m+1}/{len(EXPLAINERS)})"):
            ts.append(__go())
    results.append(
        dict(
            Method=method,
            Runtimes=ts,
            Mean=np.mean(ts),
            Minimum=np.min(ts),
            Maximum=np.max(ts),
            Variance=np.var(ts, ddof=1),
            Dataset=dataset_to_name(args.dataset),
            Model=model_version_to_name(args.model_version),
        ))

print(f'Experiment took {time.time()-start} seconds')

_ = save_pickle(results, profiling_filename(args))
# %% [markdown]
# ### Results
# Let's check the results
# %%
pd.DataFrame(results)
# %%
