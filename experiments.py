# %%
from emutils.imports import *
from emutils.geometry.metrics import adist
from emutils.utils import (
    attrdict,
    in_ipynb,
    pandas_max,
    end,
)
from emutils.file import (
    load_pickle,
    save_pickle,
    compute_or_load,
    ComputeRequest,
)
from emutils.preprocessing import MultiScaler
from emutils.parallel.utils import max_cpu_count

from cfshap.evaluation.counterfactuals.plausibility import yNNDistance, NNDistance
from cfshap.evaluation.attribution.induced_counterfactual import TreeInducedCounterfactualGeneratorV2
from cfshap.evaluation.attribution import feature_attributions_statistics

from explainers import MAXSAMPLES
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
from constants import MODEL_DIR, DATA_DIR, EXPLANATIONS_DIR, EXPERIMENTS_DIR, EXPORT_DIR, RESULTS_DIR

parser = ArgumentParser(sys.argv)

# General, Data & Model
parser.add_argument('--dataset', type=str, default='heloc', choices=['heloc', 'lendingclub', 'wines'], required=False)
parser.add_argument('--data_path', type=str, default=DATA_DIR, required=False)
parser.add_argument('--model_path', type=str, default=MODEL_DIR)
parser.add_argument('--data_version', type=str, default="v2")
parser.add_argument('--model_version', type=str, default='v5')
parser.add_argument('--model_type', type=str, default='xgb')
parser.add_argument('--random_state', type=int, default=2021, required=False)

# Experiments paths
parser.add_argument('--explanations_path', type=str, default=EXPLANATIONS_DIR)
parser.add_argument('--experiments_path', type=str, default=EXPERIMENTS_DIR)
parser.add_argument('--results_path', type=str, default=RESULTS_DIR)
parser.add_argument('--results_version', type=str, default='v1')
parser.add_argument('--export_path', type=str, default=EXPORT_DIR)

# Experiments settings
parser.add_argument('--methods', nargs='+', required=False, default=None)
parser.add_argument('--action_strategy', type=str, default=None)
parser.add_argument('--action_cost', type=str, default=None)
parser.add_argument('--costs', action='store_true', default=False)
parser.add_argument('--nn', action='store_true', default=False)
parser.add_argument('--override_induced', action='store_true', default=False)

args, unknown = parser.parse_known_args()
args = attrdict(vars(args))

os.makedirs(args.experiments_path, exist_ok=True)
os.makedirs(args.export_path, exist_ok=True)
os.makedirs(args.results_path, exist_ok=True)

if not any([args.costs, args.nn]):
    args.costs = True
    args.nn = True

# Show graphs and stuff or not?
args.show = in_ipynb()

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


# -> Results
def explanations_filename(result_name, ext='pkl'):
    return f"{args.explanations_path}/{MODEL_RUN_NAME}_{result_name}{args.results_version}.{ext}"


def experiments_filename(result_name, ext='pkl'):
    return f"{args.experiments_path}/{MODEL_RUN_NAME}_{result_name}{args.results_version}.{ext}"


def results_filename(result_name, ext='pkl'):
    return f"{args.results_path}/{MODEL_RUN_NAME}_{result_name}{args.results_version}.{ext}"


def export_filename(result_name, ext='svg'):
    return f"{args.export_path}/{MODEL_RUN_NAME}_{result_name}_{args.results_version}.{ext}"


# %% [markdown]
# # Load data and model
# Let's load all the data and the trained model.
# %%
X, y, X_train, X_test, y_train, y_test = load_data(args)

feature_names = load_pickle(FEATURES_FILENAME)
class_names = load_pickle(CLASSES_FILENAME)
feature_trends = load_pickle(TRENDS_FILENAME)
ref_points = dict(
    med=load_pickle(MEDIAN_FILENAME),
    medgood=load_pickle(MEDIANGOOD_FILENAME),
    meangood=load_pickle(MEANGOOD_FILENAME),
    mean=load_pickle(MEAN_FILENAME),
)

multiscaler = MultiScaler(X_train)

model = load_pickle(MODELWRAPPER_FILENAME)

loaded = False
while not loaded:
    try:
        metadata_array = load_pickle(explanations_filename('meta_all'))
        values_arrays = load_pickle(explanations_filename('values_all'))
        trends_arrays = load_pickle(explanations_filename('trends_all'))
        loaded = True
        logging.info('Loaded')
    except:
        logging.info('Experiments waiting')
        time.sleep(60)

X_explain = metadata_array['x']

print(f'Explain : {X_explain.shape}')
# %%

# Set number of threads for efficiency.
model.get_booster().set_param({'nthread': min(15, max_cpu_count() - 1)})
# %% [markdown]
# ## Explanations Statistics
#
# Let's plot some basic statistics
# %%


def plot_multiple_count_explanations(df__, title="Explanations Statistics"):
    df__.T.plot(kind='bar', figsize=(12, 6), grid=.1, title=title)
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return df__


if args.show:
    df_ = {name: feature_attributions_statistics(phi, mean=True) for name, phi in values_arrays.items()}
    df_ = pd.DataFrame(df_).T
    plot_multiple_count_explanations(df_)
    plt.savefig(export_filename(f"ExplanationsStatistics", ext='svg'), bbox_inches='tight', pad_inches=0)

# %% [markdown]
# # Counterfactual-ability
# %% [markdown]
# ## Feature Attributions Induced Counterfactuals
# Here we run the experiments to compute the feature-importance induced counterfactuals.
# %%
logging.getLogger().setLevel(logging.INFO)
# %%

ACTION_STRATEGIES = [args.action_strategy] if args.action_strategy is not None else ['proportional', 'random']
DIRECTIONS_STRATEGIES = ['local', 'global']
ACTION_SCOPES = ['positive']
ACTION_COST_NORMALIZATIONS = ['quantile_sum']
ACTION_COST_AGGREGATIONS = [args.action_cost] if args.action_cost is not None else ['L1', 'L2']


def compute_induced_counterfactuals(
    induced_cf_generator,
    values,
    trends,
    metadata,
    override=False,
    **kwargs,
):
    # Dict (action_normalization, action_strategy, direction_strategy) of
    #   Dict (methods) of array of
    #       np.ndarray of shape nb_samples x nb_features (top-k) x nb_features (features)
    induced_counterfactuals = defaultdict(dict)

    product = list(
        itertools.product(
            ACTION_STRATEGIES,
            DIRECTIONS_STRATEGIES,
            ACTION_SCOPES,
            ACTION_COST_NORMALIZATIONS,
            ACTION_COST_AGGREGATIONS,
            list(values.keys()),
        ))
    nb_products = len(product)
    print('Number products:', nb_products)

    start = time.time()
    last_log = time.time()
    count = 0
    for i, key in enumerate(product):
        istart = time.time()

        action_strategy, direction_strategy, action_scope, action_normalization, action_aggregation, method = key

        is_non_cfx_method = any([m in method for m in ['training', 'diff_pred', 'diff_label']])

        # Skip 'local' direction_strategy for non-counterfactual methods
        if direction_strategy == 'local' and is_non_cfx_method:
            continue

        # Skip 'global' direction_strategy for counterfactual methods
        if direction_strategy == 'global' and not is_non_cfx_method:
            continue

        # Skip method if not in args.methods
        if args.methods is not None and method not in args.methods:
            continue

        # Filename
        filename = experiments_filename(f'action_' + '_'.join(key))

        # Compute and save
        counters, _ = compute_or_load(
            filename,
            lambda: induced_cf_generator.transform(
                # explanations,
                X=metadata['x'],
                explanations=attrdict(
                    values=values[method],
                    trends=trends[method],
                ),
                action_strategy=action_strategy,
                action_direction=direction_strategy,
                action_scope=action_scope,
                action_cost_normalization=action_normalization,
                action_cost_aggregation=action_aggregation,
                K=(1, 2, 3, 4, 5),
                nan_explanation='ignore'
                if 'tolomei' in method else 'raise',  # We allow NaN explanations only for Tolomei's CFX
                desc=f"{args.dataset}/{method}",
                **kwargs),
            request=ComputeRequest.OVERRIDE if override else ComputeRequest.LOAD_OR_RUN,
            verbose=1,
        )

        # Put in the defaultdict
        induced_counterfactuals[key[:-1]][method] = counters

        # Log remaining time
        count += 1
        avg_exec_time = (time.time() - start) / count
        last_exec_time = (time.time() - istart)
        remaining_exec = (nb_products - i - 1)
        if time.time() - last_log > 30:  # We do not show logs more often than every 30 seconds
            logging.info(
                f"{remaining_exec} executions (or less) remaining. Estimated {avg_exec_time * remaining_exec / 3600:.2f} hours (avg) or {last_exec_time * remaining_exec / 3600:.2f} hours (last). Last took {last_exec_time/60:.2f} minutes. On average they took {avg_exec_time/60:.2f} minutes."
            )
            last_log = time.time()
    return induced_counterfactuals


induced_cf_generator = TreeInducedCounterfactualGeneratorV2(
    model=model,
    multiscaler=multiscaler,
    global_feature_trends=feature_trends,
    random_state=args.random_state,
)

induced_counterfactuals = compute_induced_counterfactuals(
    induced_cf_generator=induced_cf_generator,
    values=values_arrays,
    trends=trends_arrays,
    metadata=metadata_array,
    override=args.override_induced,
)

# %%
if args.action_strategy is not None or args.action_cost is not None:
    end('Partial computation finished. Re-run script with no filters on action_strategy and action_cost to aggregate results.'
        )

# Saving all results together
_ = save_pickle(induced_counterfactuals, results_filename('induced_counterfactuals'))

# %% [markdown]
# ### Test the induced counterfactuals
# %% [markdown]
# # Experiments
# %%
COSTS_NORMALIZATIONS = ['quantile']
COSTS_AGGREGATIONS = ['L1', 'L2']


# %% [markdown]
# ## Cost of induced counterfactuals
# Let's compute the cost of the induced counterfactuals
# %%
def nansafe_compute_costs_of(X, XIC, context):
    costs_ = []
    for k in range(XIC.shape[1]):
        X_C = XIC[:, k]
        isnan_mask = np.any(np.isnan(X_C), axis=1)
        # Compute cost ignoring nans
        c_ = np.full(X_C.shape[0], np.nan)
        if (~isnan_mask).sum() > 0:
            c_[~isnan_mask] = adist(context.multiscaler.transform(X[~isnan_mask], context.cost_normalization),
                                    context.multiscaler.transform(X_C[~isnan_mask], context.cost_normalization),
                                    metric=context.cost_aggregation)
        costs_.append(c_)
    return np.array(costs_).T


def run_experiments_for_cost(
    X,
    context,
    induced_counterfactuals: np.ndarray,
    experiment_name: str,
):
    context = context.copy()  # Let's be non-destructive on context

    all_results = []
    with tqmd() as t:
        for key, induc_counterfactuals in induced_counterfactuals.items():
            action_strategy, direction_strategy, action_scope, action_normalization, action_aggregation = key
            for key2 in itertools.product(COSTS_AGGREGATIONS, COSTS_NORMALIZATIONS):
                cost_aggregation, cost_normalization = key2
                for method, XIC in induc_counterfactuals.items():
                    context.update(
                        action_normalization=action_normalization,
                        action_aggregation=action_aggregation,
                        action_strategy=action_strategy,
                        action_scope=action_scope,
                        action_direction=direction_strategy,
                        cost_aggregation=cost_aggregation,
                        cost_normalization=cost_normalization,
                        method=method,
                    )
                    t.set_description(f'{experiment_name}_{"_".join(key)}_{"_".join(key2)}_{method}')

                    c_ = nansafe_compute_costs_of(X, XIC, context)
                    c__ = c_

                    t.update(1)
                    all_results.extend([
                        dict(
                            action_normalization=action_normalization,
                            action_aggregation=action_aggregation,
                            action_strategy=action_strategy,
                            action_scope=action_scope,
                            action_direction=direction_strategy,
                            cost_aggregation=cost_aggregation,
                            cost_normalization=cost_normalization,
                            method=method,
                            k=k,
                            costs=c__[:, k],
                        ) for k in range(c__.shape[1])
                    ])
    print(f'Experiment COSTS: DONE.')

    # Return results
    return all_results


# %%
if args.costs:
    all_costs = run_experiments_for_cost(
        X=X_explain,
        induced_counterfactuals=induced_counterfactuals,
        experiment_name="costs",
        context=attrdict(
            model=model,
            multiscaler=multiscaler,
        ),
    )
# %% [markdown]
# Let's post-process the results on the costs
# %%
if args.costs:
    cdf = pd.DataFrame(all_costs)

    cdf['costs_pad'] = cdf['costs'].apply(replace_nan)

    cdf['mean'] = cdf['costs'].apply(safe_mean)
    cdf['mean_pad'] = cdf['costs_pad'].apply(np.mean)

    cdf['failure'] = cdf['costs'].apply(failure_nan)

    # Filter out Aggregation
    cdf = cdf[cdf['method'].apply(lambda x: 'AG' not in x)]

    cdf['type'] = 'SHAP'
    cdf['method'] = cdf['method'].apply(lambda x: x.replace('_FA', '').replace('_SUM', ''))
    cdf['k'] += 1
    cdf = cdf[cdf['k'] <= 5]

    # Cosmetic
    cdf = cdf.sort_values(['method', 'type', 'k'])
    # cdf.drop(columns=['costs', 'costs_pad'], inplace=True)
    # cdf['k'] = cdf['k'].apply(lambda x : str(x) if x < 6 else 'inf')

    save_pickle(cdf, results_filename('costs'))

    display(cdf.sample(10))

# %% [markdown]
# # Plausibility
# %%


def run_experiments_for_nn(
    X,
    context,
    induced_counterfactuals: np.ndarray,
    ynn_dist: bool = True,
):
    context = context.copy()  # Let's be non-destructive on context

    all_results = []

    with tqdm() as t:
        for key2 in itertools.product(COSTS_AGGREGATIONS, COSTS_NORMALIZATIONS):
            cost_aggregation, cost_normalization = key2

            for k in [5]:  # [5, 10, 20]:
                _init_args = dict(
                    scaler=context.multiscaler.get_transformer(cost_normalization),
                    X=context.X_train,
                    n_neighbors=k,
                    distance=cost_aggregation,
                    max_samples=MAXSAMPLES,
                )
                if ynn_dist:
                    nn = yNNDistance(model=context.model, **_init_args)
                else:
                    nn = NNDistance(**_init_args)

                t.set_description(f'{"y" if ynn_dist else ""}NN_X_{"_".join(key2)}')

                rX = nn.score(X)

                t.update(1)
                for key, induc_counterfactuals in induced_counterfactuals.items():
                    action_strategy, direction_strategy, action_scope, action_normalization, action_aggregation = key
                    for method, XIC in induc_counterfactuals.items():

                        t.set_description(
                            f'{"y" if ynn_dist else ""}NN_IC_{"_".join(key)}_{"_".join(key2)}_{k}_{method}')

                        rXIC = [nn.score(XIC[:, topk]) for topk in range(XIC.shape[1])]

                        t.update(1)
                        for topk, rXIC_ in enumerate(rXIC):
                            r_ = rXIC_
                            rx_ = rX
                            d = dict(
                                action_normalization=action_normalization,
                                action_aggregation=action_aggregation,
                                action_strategy=action_strategy,
                                action_scope=action_scope,
                                action_direction=direction_strategy,
                                cost_aggregation=cost_aggregation,
                                cost_normalization=cost_normalization,
                                method=method,
                                topk=topk,
                                k=k,
                                xNN=rx_,
                            )
                            d[f'{"y" if ynn_dist else ""}NN'] = r_
                            all_results.append(d)

    print(f'Experiment {"y" if ynn_dist else ""}NN: DONE.')

    return all_results


# %%
# %%
if args.nn:
    all_nn = run_experiments_for_nn(
        X=X_explain,
        induced_counterfactuals=induced_counterfactuals,
        context=attrdict(
            model=model,
            multiscaler=multiscaler,
            X_train=X_train.values,
        ),
        ynn_dist=False,
    )
# %%
if args.nn:
    nndf = pd.DataFrame(all_nn)

    # Top-k
    nndf['topk'] += 1
    nndf = nndf[nndf['topk'] <= 5]

    # Method
    nndf['type'] = 'SHAP'
    nndf['method'] = nndf['method'].apply(lambda x: x.replace('_FA', '').replace('_SUM', ''))

    # Results
    nndf['failure'] = nndf['NN'].apply(failure_nan)

    # Sort
    nndf = nndf.sort_values(['method', 'type', 'topk'])

    # Save
    save_pickle(nndf, results_filename('nns'))

    # Show
    display(nndf.head(10))