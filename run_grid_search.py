from map_estimator import MAPEstimator
from sklearn.preprocessing import PolynomialFeatures
import vis_utils
import sklearn.model_selection
import sklearn
import numpy as np
import pandas as pd
import time
import copy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("notebook")


def main(block=False):
    x_train_ND, t_train_N, x_test_ND, t_test_N = vis_utils.load_dataset(
        data_dir="traffic_data")

    hypers_to_search = dict(
        order=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        alpha=np.logspace(-8, 1, 10).tolist(),
        beta=np.logspace(-1, 1, 3 + 24).tolist(),
    )

    # do a kfold grid search over each order of model
    order_list = hypers_to_search['order']
    highest_score = -np.inf
    best_model = -1

    for order in order_list:
        print("Model order:", order)
        cur_hypers_to_search = copy.deepcopy(hypers_to_search)
        cur_hypers_to_search['order'] = [order]

        feature_transformer = PolynomialFeatures(
            degree=order, include_bias=True)

        kfold_splitter = sklearn.model_selection.KFold(
            n_splits=3, shuffle=True, random_state=101)

        kfold_grid_searcher = sklearn.model_selection.GridSearchCV(
            MAPEstimator(feature_transformer),
            cur_hypers_to_search,
            cv=kfold_splitter,
            scoring=None,
            refit=True,
            return_train_score=False,
            error_score='raise')

        kfold_grid_searcher.fit(x_train_ND, t_train_N)

        best_score = kfold_grid_searcher.best_score_
        best_estimator = kfold_grid_searcher.best_estimator_

        test_score = best_estimator.score(x_test_ND, t_test_N)

        predictions = best_estimator.predict(x_test_ND)
        pred_variance = best_estimator.predict_variance(x_test_ND)

        # Undo normalization for day of the week
        dow_std = 2.007250254034316
        dow_mean = 2.9137009516360317
        dow_orig = np.rint(x_test_ND[:, 2] * dow_std + dow_mean).astype(int)

        # Undo normalization for days since january
        dsj1_std = 46.70135254173618
        dsj1_mean = 107.82883587537479
        dsj1_orig = np.rint(x_test_ND[:, 1] * dsj1_std + dsj1_mean).astype(int)

        # Undo normalization for travel time
        tt_mean = 2682.591057228523
        tt_std = 267.21950503278794
        tt_orig = np.rint(t_test_N * tt_std + tt_mean).astype(int)

        new_data = copy.deepcopy(x_test_ND)
        new_data[:, 2] = dow_orig
        new_data[:, 1] = dsj1_orig

        # only keep data that was on june 10th
        mask = (new_data[:, 1] == 161)

        # only keep data on June 4th
        # mask = (new_data[:, 1] == 155)

        # only keep data from fridays
        # mask = (new_data[:, 2] == 4)

        print("Best Score:", best_score)
        print("Test Score:", test_score)

        departure_sec = x_test_ND[:, 0]

        # get grid with nice lines, only for a specific day in june!!!!
        filtered_data = x_test_ND[mask]

        dep_grid = np.linspace(
            filtered_data[:, 0].min(),
            filtered_data[:, 0].max(),
            300
        )
        x_grid = np.zeros((dep_grid.size, filtered_data.shape[1]))
        x_grid[:, 0] = dep_grid
        x_grid[:, 1:] = np.mean(filtered_data[:, 1:], axis=0)
        t_pred = best_estimator.predict(x_grid)
        t_pred_var = best_estimator.predict_variance(x_grid)

        t_pred_orig = np.rint((t_pred * tt_std + tt_mean)).astype(int)
        t_pred_var_orig = np.rint(t_pred_var * tt_std + tt_mean).astype(int)
        t_pred_std = np.sqrt(t_pred_var_orig)

        # transform departure time from seconds to hours
        plt.scatter(departure_sec[mask]/3600, tt_orig[mask], s=5,
                    alpha=0.3, color="tab:orange", label="True travel times")

        plt.plot(dep_grid/3600, t_pred_orig, linewidth=2, label="MAP mean")

        plt.fill_between(
            dep_grid/3600,
            (t_pred_orig - t_pred_std),
            (t_pred_orig + t_pred_std),
            alpha=0.3,
            label="±1 std"
        )
        plt.title(
            f"June 10: True travel time vs MAP prediction (order {order})\n Val score: {best_score:.4f}, Test score: {test_score:.4f}")
        plt.legend()

        plt.xlabel("Departure time (hours after midnight)")
        plt.ylabel("Travel time (seconds)")
        if best_score > highest_score:
            highest_score = best_score
            best_model = order
            fname = f"June10_order{order}.png"
            plt.savefig(fname)
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()
