# %%
from emutils.imports import *
from emutils.utils import (attrdict, in_ipynb, pandas_max, load_pickle, save_pickle, notebook_fullwidth)
from emutils.file import (
    compute_or_load,
    ComputeRequest,
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

notebook_fullwidth()
# %%
from constants import DATA_DIR, MODEL_DIR, EXPLANATIONS_DIR

parser = ArgumentParser(sys.argv)

# General
parser.add_argument('--dataset', type=str, default='heloc', choices=['heloc', 'lendingclub', 'wines'], required=False)
parser.add_argument('--data_version', type=str, default="v2")
parser.add_argument('--data_path', type=str, default=DATA_DIR, required=False)
parser.add_argument('--random_state', type=int, default=2021, required=False)

# Model
parser.add_argument('--model_path', type=str, default=MODEL_DIR)
parser.add_argument('--model_type', type=str, default='xgb')
parser.add_argument('--model_version', type=str, default='v5')

# Results
parser.add_argument('--explanations_path', type=str, default=EXPLANATIONS_DIR)
parser.add_argument('--results_version', type=str, default='v5_close')

# Experiments
parser.add_argument('--nb_samples', type=int, default=2000, required=False)
parser.add_argument('--override', dest='override', action='store_true', default=False)
parser.add_argument('--close', type=float, default=None, required=False)
parser.add_argument('--far', type=float, default=None, required=False)
parser.add_argument('--monotonic', action='store_true', default=False)
parser.add_argument('--no-backgrounds', dest='backgrounds', action='store_false', default=True)

args, unknown = parser.parse_known_args()
args = attrdict(vars(args))

os.makedirs(args.explanations_path, exist_ok=True)

if '_test' in args.results_version:
    args.nb_samples = 10
    args.override = True

if 'bivariate' in args.dataset:
    args.override = True
    args.close = .2

if 'close' in args.results_version and args.close is None:
    print(args)
    raise ValueError('Forgot close?')

if 'far' in args.results_version and args.far is None:
    print(args)
    raise ValueError('Forgot far?')

if 'far' in args.results_version and args.close is not None:
    print(args)
    raise ValueError('Forgot to remove close?')

if 'close' in args.results_version and args.far is not None:
    print(args)
    raise ValueError('Forgot to remove far?')

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
BOOSTER_FILENAME = f"{args.model_path}/{MODEL_RUN_NAME}_model.bin"
TXTMODEL_FILENAME = f"{args.model_path}/{MODEL_RUN_NAME}_model.txt"
PARAMS_FILENAME = f"{args.model_path}/{MODEL_RUN_NAME}_params.json"
BAD_FILENAME = f"{args.model_path}/{MODEL_RUN_NAME}_bad.pkl"
GOOD_FILENAME = f"{args.model_path}/{MODEL_RUN_NAME}_good.pkl"


# -> Results
def result_filename(result_name, ext='pkl'):
    return f"{args.explanations_path}/{MODEL_RUN_NAME}_{result_name}{args.results_version}.{ext}"


# %% [markdown]
# # Load data and model
# Let's load all the data and the trained model.
# We load here also the manifold information.
# %%
X, y, X_train, X_test, y_train, y_test = load_data(args)

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

if args.close is not None or args.far is not None:
    bad_preds_ = model.predict_proba(X_bad.values)[:, 1]
    bad_close_prob_ = np.sort(bad_preds_)[int(round(args.close *
                                                    len(bad_preds_)))] if args.close is not None else np.inf
    bad_far_prob_ = np.flip(np.sort(bad_preds_))[int(round(args.far *
                                                           len(bad_preds_)))] if args.far is not None else -np.inf
    closeandfar_mask = (bad_preds_ < bad_close_prob_) & (bad_preds_ > bad_far_prob_)
    X_bad_filtered = X_bad[closeandfar_mask]
else:
    X_bad_filtered = X_bad

X_explain = X_bad_filtered.sample(n=min(len(X_bad_filtered), args.nb_samples), random_state=args.random_state)
index_explain = X_explain.index.values
X_explain = X_explain.values

print(f"Bad : {X_bad.shape}")
print(f"Good : {X_good.shape}")
print(f'Bad FILTERED (close and far) : {X_bad_filtered.shape}')
print(f'Explain : {X_explain.shape}')

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
# %% [markdown]
# Let's check that everything is working with the explainers
# %%


def is_xp_col(col):
    if not isinstance(col, str):
        return True
    if col in ['pred', 'prob', 'x', 'x_prime', 'epsilon', 'eps', 'x_nonrecoded']:
        return False
    if any([col.endswith(c) for c in ['_X', '_tweaks', '_X_len', '_Xaggr']]):
        return False
    return True


def xps_to_df(xp_result, features_info=None, sort=True):
    df = pd.DataFrame({k: v for k, v in xp_result.items() if is_xp_col(k)})
    for col in df:
        df[col + "_rank"] = (-1 * df[col]).rank()
    if 'x' in xp_result:
        df['x'] = xp_result['x']
    if sort:
        df = df.reindex(sorted(df.columns), axis=1)
    if features_info is not None:
        df = pd.concat([df, features_info], axis=1)
    return df


if True:
    x = X_explain[1]
    assert model.predict([x])[0] == 1, "Not a bad sample"

    pred = model.predict([x])
    prob = model.predict_proba([x])
    print(x, pred, prob)

    explanations = {name: explainer(x.reshape(1, -1)) for name, explainer in EXPLAINERS.items()}
    trends = {name: explanations[name]['trends'][0] for name in EXPLAINERS.keys()}
    values = {name: explanations[name]['values'][0] for name in EXPLAINERS.keys()}
    backgrounds = {name: explanations[name]['backgrounds'][0] for name in EXPLAINERS.keys()}
    trends['DATA'] = feature_trends_

    pdf = xps_to_df(values)
    tdf = xps_to_df(trends)
    pdf = pdf[[c for c in pdf.columns.values if '_rank' not in c]]
    tdf = tdf[[c for c in tdf.columns.values if '_rank' not in c]]

    pandas_max(200, 200)
    display(pdf.T)
    display(tdf.T)

# %% [markdown]
# Let's check that everything is working fine...
# %%
for name, back in backgrounds.items():
    if np.any(np.isnan(back)):
        print("This Explainer has some NAN counterfactuals:", name)
        # print(back)

# %% [markdown]
# # Explain
# Let's now run the explainer on the sample of the dataset
# %%
explanations_cache = None


def compute_explanations(X, index, model, explainers):
    global explanations_cache

    metadata_array = compute_or_load(result_filename('meta_all'), dict, request=ComputeRequest.LOAD_OR_RUN_NOSAVE)

    # We do not load them from here but from the original results (below)
    values_arrays = {}
    trends_arrays = {}
    backgrounds_arrays = {}

    X_ = np.asarray(X)
    pred_ = model.predict(X)
    prob_ = model.predict_proba(X)[:, 1]

    if 'x' in metadata_array:
        metaX = metadata_array['x']
        assert np.all(metaX == X_[:len(metaX)])
    if 'index' in metadata_array:
        metaindex = metadata_array['index']
        assert np.all(metaindex == index[:len(metaX)])
    if 'pred' in metadata_array:
        metapred = metadata_array['pred']
        assert np.all(metapred == pred_[:len(metapred)])
    if 'prob' in metadata_array:
        metaprob = metadata_array['prob']
        assert np.all(metaprob == prob_[:len(metaprob)])

    metadata_array['x'] = X_
    metadata_array['pred'] = pred_
    metadata_array['prob'] = prob_
    metadata_array['manifold'] = None
    metadata_array['index'] = index

    # Compute new explanations
    iters = tqdm(explainers.items())
    for name, explainer in iters:

        iters.set_description(name)

        def __explain():
            global explanations_cache
            explanations = explanations_cache = explainer(X_)
            return explanations.values, explanations.trends

        def __backgrounds():
            global explanations_cache
            if explanations_cache is not None:
                backgrounds = explanations_cache.backgrounds
            else:
                backgrounds = explainer.get_backgrounds(X_)

            # Check if they are all the same
            all_same = True
            prev = backgrounds[0]
            for i in range(1, len(backgrounds)):
                curr = backgrounds[i]
                if prev.shape != curr.shape or np.any(prev != curr):
                    all_same = False
                    break
                prev = curr

            # Save them (None if repeated)
            if all_same:
                return [backgrounds[0]] + [None] * (len(backgrounds) - 1)
            else:
                return backgrounds

        # Compute values and trends
        values_arrays[name], trends_arrays[name] = compute_or_load(
            result_filename(f'xps_{name}'),
            lambda: __explain(),
            request=ComputeRequest.OVERRIDE if args.override else ComputeRequest.LOAD_OR_RUN,
            verbose=1,
        )

        # Assert that explanation method is still coherent
        e = explainer(X_[0].reshape(1, -1))
        v, t = e.values[0], e.trends[0]
        v0, t0 = values_arrays[name][0], trends_arrays[name][0]

        assert np.all((np.abs(v - v0) < 1e-8)
                      | (np.isnan(v) & np.isnan(v0))), f'Feature attribution is not coherent! ({v0} != {v})'
        assert np.all((np.abs(t - t0) < 1e-8) | (np.isnan(v) & np.isnan(v0))), f'Trend is not coherent! ({t0} != {t})'

        # Compute backgrounds
        if args.backgrounds:
            backgrounds_arrays[name] = compute_or_load(
                result_filename(f'xps_{name}_B'),
                lambda: __backgrounds(),
                request=ComputeRequest.OVERRIDE if args.override else ComputeRequest.LOAD_OR_RUN,
                verbose=1,
            )
        explanations_cache = None

    _ = save_pickle(metadata_array, result_filename('meta_all'))
    _ = save_pickle(values_arrays, result_filename('values_all'))
    _ = save_pickle(trends_arrays, result_filename('trends_all'))
    _ = save_pickle(backgrounds_arrays, result_filename('backgrounds_all'))

    return metadata_array, values_arrays, trends_arrays, backgrounds_arrays


# Compute explanations
metadata_array, values_arrays, trends_arrays, backgrounds_arrays = compute_explanations(X=X_explain,
                                                                                        index=index_explain,
                                                                                        model=model,
                                                                                        explainers=EXPLAINERS)
