import os, sys, gc
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from IPython.display import display
import seaborn as sns
import xgboost as xgb

from slickml.feature_engineering import noisy_features
from slickml.utilities import df_to_csr, memory_use_csr
from slickml.formatting import Color


class XGBoostFeatureSelector:
    """XGBoost Feature Selector.
    This is wrapper using XGBoost classifier to run xgboost.cv()
    model with n-folds cross-validation on top of augmented data
    with noisy features iteratively. At each n-fold CV of each iteration,
    it finds the best boosting round to overcome the over-fitting and
    run xgboost.train(). Main reference is XGBoost Python API:
    (https://xgboost.readthedocs.io/en/latest/python/python_api.html)
    Parameters
    ----------
    X: numpy.array or Pandas DataFrame
        Features data
    y: numpy.array[int] or list[int]
        List of ground truth binary values [0, 1]
    n_iter: int, optional (default=3)
        Number of iteration for feature selection
    num_boost_round: int, optional (default=100)
        Number of boosting round at each fold of xgboost.cv()
    n_splits: int, optional (default=4)
        Number of folds for cross-validation
    metrics: str or tuple[str], optional (default=("auc"))
        Metric used for evaluation at cross-validation
        using xgboost.cv(). Please note that this is different
        than eval_metric that needs to be passed to params dict.
        Possible values are "auc", "aucpr"
    early_stopping_rounds: int, optional (default=20)
        The criterion to early abort the xgboost.cv() phase
        if the test metric is not improved
    random_state: int, optional (default=1367)
        Random seed
    stratified: bool, optional (default=True)
        Flag to stratificaiton of the targets to run xgboost.cv() to
        find the best number of boosting round at each fold of
        each iteration
    shuffle: bool, optional (default=True)
        Flag to shuffle data to have the ability of building
        stratified folds in xgboost.cv()
    sparse_matrix: bool, optional (default=False)
        Flag to convert data to sparse matrix with csr format.
        This would increase the speed of feature selection for
        relatively large datasets
    nth_noise_threshold: int, optional (default=1)
        The threshold to keep all the features up to the n-th
        noisy feature at each fold of each iteration. For example,
        for a feature selection with 4 iterations and 5-folds cv,
        maximum number of noisy features would be 4*5=20.
    show_stdv: bool, optional (default=False)
        Flag to show standard deviations in callbacks for
        xgboost.cv() results
    importance_type: str, optional (default="total_gain")
        Importance type of xgboost.train() with possible values
        "weight", "gain", "total_gain", "cover", "total_cover"
    params: dict, optional
        Set of parameters for evaluation of xboost.train()
        (default={"eval_metric" : "auc",
                  "tree_method": "hist",
                  "objective" : "binary:logistic",
                  "learning_rate" : 0.05,
                  "max_depth": 2,
                  "min_child_weight" : 1,
                  "gamma" : 0.0,
                  "reg_alpha" : 0.0,
                  "reg_lambda" : 1.0,
                  "subsample" : 0.9,
                  "max_delta_step": 1,
                  "silent" : True,
                  "nthread" : 4,
                  "scale_pos_weight" : 1})
    verbose_eval: bool, optional (default=True)
        Flag to show the results of xgboost.train() on train/test sets
        using params["eval_metric"]
    callbacks: bool, optional (default=False)
        Flag for printing results during xgboost.cv().
        This would help to track the early stopping criterion
    Attributes
    ----------
    feature_importance_: dict()
        Returns a dict() of all feature importance based on
        importance_type at each fold of each iteration during
        selection process
    feature_frequency_: Pandas DataFrame()
        Returns a DataFrame() cosists of total frequency of
        each feature during the selection process
    cv_results_: dict()
        Return a dict() of the total internal/external
        cross-validation results
    get_xgb_params(): class method
        Returns params dict
    get_bst_feature_importance(): class method
        Returns feature importance based on importance_type
        at each fold of each iteration of the selection process
    get_feature_frequency(): class method
        Returns the total feature frequency of the bst model
        at each fold of each iteration of selection process
    get_cv_results(): class method
        Returns the total internal/external cross-validation results
    """

    def __init__(
        self,
        X,
        y,
        n_iter=None,
        num_boost_round=None,
        n_splits=None,
        metrics=None,
        early_stopping_rounds=None,
        random_state=None,
        stratified=True,
        shuffle=True,
        sparse_matrix=False,
        nth_noise_threshold=None,
        show_stdv=False,
        importance_type=None,
        params=None,
        verbose_eval=False,
        callbacks=False,
    ):

        if isinstance(X, np.ndarray):
            self.X = pd.DataFrame(X, columns=[f"F_{i}" for i in range(X.shape[1])])
        elif isinstance(X, pd.DataFrame):
            self.X = X
        else:
            raise TypeError("The input X must be numpy array or pandas DataFrame.")

        if isinstance(y, np.ndarray) or isinstance(y, list):
            self.y = y
        else:
            raise TypeError("The input y must be numpy array or list.")
        self.y = y

        if n_iter is None:
            self.n_iter = 3
        else:
            if not isinstance(n_iter, int):
                raise TypeError("The input n_iter must have integer dtype.")
            else:
                self.n_iter = n_iter

        if num_boost_round is None:
            self.num_boost_round = 100
        else:
            if not isinstance(num_boost_round, int):
                raise TypeError("The input num_boost_round must have integer dtype.")
            else:
                self.num_boost_round = num_boost_round

        if n_splits is None:
            self.n_splits = 4
        else:
            if not isinstance(n_splits, int):
                raise TypeError("The input n_splits must have integer dtype.")
            else:
                self.n_splits = n_splits

        if metrics is None:
            self.metrics = "auc"
        else:
            if not isinstance(metrics, str):
                raise TypeError("The input metrics must be a str dtype.")
            else:
                self.metrics = metrics

        if early_stopping_rounds is None:
            self.early_stopping_rounds = 20
        else:
            if not isinstance(early_stopping_rounds, int):
                raise TypeError(
                    "The input early_stopping_rounds must have integer dtype."
                )
            else:
                self.early_stopping_rounds = early_stopping_rounds

        if random_state is None:
            self.random_state = 1367
        else:
            if not isinstance(random_state, int):
                raise TypeError("The input random_state must have integer dtype.")
            else:
                self.random_state = random_state

        if not isinstance(stratified, bool):
            raise TypeError("The input stratified must have bool dtype.")
        else:
            self.stratified = stratified

        if not isinstance(shuffle, bool):
            raise TypeError("The input shuffle must have bool dtype.")
        else:
            self.shuffle = shuffle

        if not isinstance(sparse_matrix, bool):
            raise TypeError("The input sparse_matrix must have bool dtype.")
        else:
            self.sparse_matrix = sparse_matrix

        if nth_noise_threshold is None:
            self.nth_noise_threshold = 1
        else:
            if not isinstance(nth_noise_threshold, int):
                raise TypeError(
                    "The input nth_noise_threshold must have integer dtype."
                )
            else:
                self.nth_noise_threshold = nth_noise_threshold

        if not isinstance(show_stdv, bool):
            raise TypeError("The input show_stdv must have bool dtype.")
        else:
            self.show_stdv = show_stdv

        if importance_type is None:
            self.importance_type = "total_gain"
        else:
            if not isinstance(importance_type, str):
                raise TypeError("The input importance_type must have str dtype.")
            else:
                if importance_type in [
                    "weight",
                    "gain",
                    "total_gain",
                    "cover",
                    "total_cover",
                ]:
                    self.importance_type = importance_type
                else:
                    raise ValueError("The input importance_type value is not valid.")
        params_ = {
            "eval_metric": "auc",
            "tree_method": "hist",
            "objective": "binary:logistic",
            "learning_rate": 0.05,
            "max_depth": 2,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "subsample": 0.9,
            "max_delta_step": 1,
            "silent": True,
            "nthread": 4,
            "scale_pos_weight": 1,
        }
        if params is None:
            self.params = params_
        else:
            if not isinstance(params, dict):
                raise TypeError("The input params must have dict dtype.")
            else:
                self.params = params_
                for key, val in params.items():
                    self.params[key] = val

        if not isinstance(verbose_eval, bool):
            raise TypeError("The input verbose_eval must have bool dtype.")
        else:
            self.verbose_eval = verbose_eval

        if not isinstance(callbacks, bool):
            raise TypeError("The input callbacks must have bool dtype.")
        else:
            if callbacks:
                self.callbacks = [
                    xgb.callback.print_evaluation(show_stdv=self.show_stdv),
                    xgb.callback.early_stop(self.early_stopping_rounds),
                ]
            else:
                self.callbacks = None

    def _xgb_imp_to_df(self, bst):
        """
        Function to build convert feature importance to df.
        """
        data = {"feature": [], f"{self.importance_type}": []}
        cols = []
        importance = []
        features_gain = bst.get_score(importance_type=self.importance_type)
        for key, val in features_gain.items():
            data["feature"].append(key)
            data[f"{self.importance_type}"].append(val)

        df = (
            pd.DataFrame(data)
            .sort_values(by=f"{self.importance_type}", ascending=False)
            .reset_index(drop=True)
        )

        return df

    def get_xgb_params(self):
        """
        Function to return the train parameters for XGBoost.
        """
        return self.params

    def get_bst_feature_importance(self):
        """
        Function to return the feature importance of the bst model
        at each fold of each iteration of feature selection.
        """
        return self.feature_importance_

    def get_feature_frequency(self):
        """
        Function to return the total feature frequency of the bst model
        at each fold of each iteration of feature selection.
        """
        return self.feature_frequency_

    def get_cv_results(self):
        """
        Function to return both internal and external
        cross-validation results.
        """
        return self.cv_results_

    def run(self):
        """
        Function to run the main feature selection algorithm.
        """
        # final results
        int_cv_train = []
        int_cv_test = []
        ext_cv_train = []
        ext_cv_test = []
        pruned_features = []
        self.feature_importance_ = {}

        # main loop
        for iteration in range(self.n_iter):
            print(
                Color.BOLD
                + "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* "
                + Color.B_Green
                + f"Iteration {iteration + 1}"
                + Color.END
                + Color.BOLD
                + " *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
            )

            # results at each iteration
            int_cv_train2 = []
            int_cv_test2 = []
            ext_cv_train2 = []
            ext_cv_test2 = []

            # update random state
            random_state_ = self.random_state * iteration

            # adding noise to data
            X_permuted = noisy_features(X=self.X, random_state=random_state_)
            cols = X_permuted.columns.tolist()
            Xval = X_permuted.values

            # building DMatrix for training/testing + kfolds cv
            cv = StratifiedKFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=random_state_
            )

            # set a counter for nfolds cv
            ijk = 1
            for train_index, test_index in cv.split(Xval, self.y):
                X_train = pd.DataFrame(data=Xval[train_index], columns=cols)
                X_test = pd.DataFrame(data=Xval[test_index], columns=cols)
                Y_train = self.y[train_index]
                Y_test = self.y[test_index]

                if not self.sparse_matrix:
                    dtrain = xgb.DMatrix(data=X_train, label=Y_train)
                    dtest = xgb.DMatrix(data=X_test, label=Y_test)
                else:
                    dtrain = xgb.DMatrix(
                        data=df_to_csr(X_train, fillna=0.0, verbose=False),
                        label=Y_train,
                        feature_names=X_train.columns.tolist(),
                    )
                    dtest = xgb.DMatrix(
                        data=df_to_csr(X_test, fillna=0.0, verbose=False),
                        label=Y_test,
                        feature_names=X_test.columns.tolist(),
                    )

                # watchlist during final training
                watchlist = [(dtrain, "train"), (dtest, "eval")]

                # dict to store training results
                evals_result = {}

                # xgb cv
                cv_results = xgb.cv(
                    params=self.params,
                    dtrain=dtrain,
                    num_boost_round=self.num_boost_round,
                    nfold=self.n_splits,
                    stratified=self.stratified,
                    metrics=self.metrics,
                    early_stopping_rounds=self.early_stopping_rounds,
                    seed=random_state_,
                    verbose_eval=self.verbose_eval,
                    shuffle=self.shuffle,
                    callbacks=self.callbacks,
                )

                int_cv_train.append(cv_results.iloc[-1][0])
                int_cv_test.append(cv_results.iloc[-1][2])
                int_cv_train2.append(cv_results.iloc[-1][0])
                int_cv_test2.append(cv_results.iloc[-1][2])

                # xgb train
                bst = xgb.train(
                    params=self.params,
                    dtrain=dtrain,
                    num_boost_round=len(cv_results) - 1,
                    evals=watchlist,
                    evals_result=evals_result,
                    verbose_eval=self.verbose_eval,
                )

                feature_gain = self._xgb_imp_to_df(bst)
                self.feature_importance_[
                    f"bst_iter{iteration+1}_fold{ijk}"
                ] = feature_gain

                # check wheather noisy feature is selected
                if feature_gain["feature"].str.contains("noisy").sum() != 0:
                    gain_threshold = feature_gain.loc[
                        feature_gain["feature"].str.contains("noisy") == True,
                        self.importance_type,
                    ].values.tolist()[self.nth_noise_threshold - 1]
                else:
                    gain_threshold = 0.0

                # subsetting features for > gain_threshold
                gain_subset = feature_gain.loc[
                    feature_gain[self.importance_type] > gain_threshold, "feature"
                ].values.tolist()
                for c in gain_subset:
                    pruned_features.append(c)

                # appending outputs
                ext_cv_train.append(
                    evals_result["train"][self.params["eval_metric"]][-1]
                )
                ext_cv_test.append(evals_result["eval"][self.params["eval_metric"]][-1])
                ext_cv_train2.append(
                    evals_result["train"][self.params["eval_metric"]][-1]
                )
                ext_cv_test2.append(
                    evals_result["eval"][self.params["eval_metric"]][-1]
                )

                print(
                    Color.BOLD
                    + "*-*-*-*-*-*-*-*-*-*-*-* "
                    + Color.F_Green
                    + f"Fold = {ijk}/{self.n_splits}"
                    + Color.F_Black
                    + " -- "
                    + Color.F_Red
                    + f"Train {self.params['eval_metric'].upper()} = {evals_result['train'][self.params['eval_metric']][-1]:.3f}"
                    + Color.F_Black
                    + " -- "
                    + Color.F_Blue
                    + f"Test {self.params['eval_metric'].upper()} = {evals_result['eval'][self.params['eval_metric']][-1]:.3f}"
                    + Color.END
                    + Color.BOLD
                    + " *-*-*-*-*-*-*-*-*-*-*-*"
                )
                # free memory here at each fold
                del (
                    gain_subset,
                    feature_gain,
                    bst,
                    watchlist,
                    Y_train,
                    Y_test,
                    cv_results,
                    evals_result,
                    X_train,
                    X_test,
                    dtrain,
                    dtest,
                )

                ijk += 1
                gc.collect()

            print(
                Color.BOLD
                + "*-*-* "
                + Color.GREEN
                + f"Internal {self.n_splits}-Folds CV:"
                + Color.END
                + Color.BOLD
                + " -*-*- "
                + Color.F_Red
                + f"Train {self.params['eval_metric'].upper()} = {np.mean(int_cv_train2):.3f} +/- {np.std(int_cv_train2):.3f}"
                + Color.END
                + Color.BOLD
                + " -*-*- "
                + Color.F_Blue
                + f"Test {self.params['eval_metric'].upper()} = {np.mean(int_cv_test2):.3f} +/- {np.std(int_cv_test2):.3f}"
                + Color.END
                + Color.BOLD
                + " *-*-*"
            )

            print(
                Color.BOLD
                + "*-*-* "
                + Color.GREEN
                + f"External {self.n_splits}-Folds CV:"
                + Color.END
                + Color.BOLD
                + " -*-*- "
                + Color.F_Red
                + f"Train {self.params['eval_metric'].upper()} = {np.mean(ext_cv_train2):.3f} +/- {np.std(ext_cv_train2):.3f}"
                + Color.END
                + Color.BOLD
                + " -*-*- "
                + Color.F_Blue
                + f"Test {self.params['eval_metric'].upper()} = {np.mean(ext_cv_test2):.3f} +/- {np.std(ext_cv_test2):.3f}"
                + Color.END
                + Color.BOLD
                + " *-*-*\n"
            )

            # free memory here at iteration
            del (
                int_cv_train2,
                int_cv_test2,
                ext_cv_train2,
                ext_cv_test2,
                X_permuted,
                cols,
                Xval,
                cv,
            )
            gc.collect()

        # putting together the outputs in one dict
        self.cv_results_ = {}
        self.cv_results_["int_cv_train"] = int_cv_train
        self.cv_results_["int_cv_test"] = int_cv_test
        self.cv_results_["ext_cv_train"] = ext_cv_train
        self.cv_results_["ext_cv_test"] = ext_cv_test

        # pruned features freq
        unique_elements, counts_elements = np.unique(
            pruned_features, return_counts=True
        )
        counts_elements = [float(i) for i in list(counts_elements)]
        feature_frequency = pd.DataFrame(
            data={"Feature": list(unique_elements), "Frequency": counts_elements}
        )
        feature_frequency["Frequency (%)"] = round(
            (feature_frequency["Frequency"] / float(self.n_splits * self.n_iter) * 100),
            ndigits=2,
        )
        self.feature_frequency_ = feature_frequency.sort_values(
            by=["Frequency", "Frequency (%)"], ascending=[False, False]
        ).reset_index(drop=True)

        return None
