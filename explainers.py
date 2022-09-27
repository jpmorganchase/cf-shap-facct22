from emutils.preprocessing import process_data

from cfshap.attribution import (
    TreeExplainer,
    CompositeExplainer,
)
from cfshap.counterfactuals import (
    BisectionProjectionDBCounterfactuals,
    KNNCounterfactuals,
    compose_diverse_counterfactual_method,
)

from cfshap.background import (
    DifferentLabelBackgroundGenerator,
    DifferentPredictionBackgroundGenerator,
)
from cfshap.trend import TrendEstimator

MAXSAMPLES = 10000
NN_Ks = [1, 3, 5, 10, 20, 50, 100, 250, 500, 1000]

trend_estimator = TrendEstimator('mean')


def create_counterfactual_explainers(data,
                                     y,
                                     model,
                                     feature_names,
                                     multiscaler,
                                     random_state,
                                     verbose,
                                     feature_trends=None):

    cf_methods = {
        **dict(
            diff_pred=DifferentPredictionBackgroundGenerator(
                model,
                data,
                random_state=random_state,
                max_samples=MAXSAMPLES,
            ),
            diff_label=DifferentLabelBackgroundGenerator(
                model,
                data,
                y,
                random_state=random_state,
                max_samples=MAXSAMPLES,
            ),
            # Default SHAP (only 100 samples)
            diff_pred_100=DifferentPredictionBackgroundGenerator(
                model,
                data,
                random_state=random_state,
                max_samples=100,
            ),
            diff_label_100=DifferentLabelBackgroundGenerator(
                model,
                data,
                y,
                random_state=random_state,
                max_samples=100,
            ),
        ),
        # -------------- KNN ------------------------------------------------
        **{
            f"knn{k}_qL1": KNNCounterfactuals(
                model=model,
                X=data,
                n_neighbors=k,
                distance='cityblock',
                scaler=multiscaler.get_transformer('quantile'),
                max_samples=MAXSAMPLES,
            )
            for k in NN_Ks
        },
    }

    cf_methods.update({
        # -------------- KNN Projected on DB --------------------------------------
        f"cone-{wrapped_name}": compose_diverse_counterfactual_method(
            BisectionProjectionDBCounterfactuals(
                model=model,
                data=data,
                max_samples=MAXSAMPLES,
                random_state=random_state,
                verbose=0,
            ),
            wrapped,
            verbose=verbose,
        )
        for wrapped_name, wrapped in cf_methods.items() if 'knn' in wrapped_name
        and int(wrapped_name.split('knn')[1].split('_')[0]) in [100]  # Only for some KNN we do also the projection
    })

    return cf_methods


def create_explainers(X,
                      y,
                      model,
                      ref_points,
                      feature_names,
                      multiscaler,
                      random_state,
                      feature_trends=None,
                      verbose=True):

    # Pre-process data
    data = process_data(X, ret_type='np', names=feature_names, names_is='subset')
    y = process_data(y, ret_type='np').flatten()

    # Create CF explainers
    cf_methods = create_counterfactual_explainers(data, y, model, feature_names, multiscaler, random_state, verbose,
                                                  feature_trends)

    return {
        ###################### CLASSIC background ###########################################
        **dict(
            training=TreeExplainer(
                model,
                data=data,
                feature_perturbation='interventional',
                max_samples=MAXSAMPLES,
                trend_estimator=trend_estimator,
            ),
            training_100=TreeExplainer(
                model,
                data=data,
                feature_perturbation='interventional',
                max_samples=100,
                trend_estimator=trend_estimator,
            ),
        ),
        ################## SHAP CF (including diff_pred and label) ##############################
        **{
            method: CompositeExplainer(
                cfe,
                TreeExplainer(
                    model,
                    data=None,
                    feature_perturbation='interventional',
                    trend_estimator=trend_estimator,
                    max_samples=MAXSAMPLES,
                ),
                verbose=verbose,
            )
            for method, cfe in cf_methods.items()
        },
    }