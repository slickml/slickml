from bayes_opt import BayesianOptimization
import pandas as pd
import xgboost as xgb
from hyperopt.pyll.stochastic import sample
from hyperopt import fmin, Trials, tpe, STATUS_OK, STATUS_FAIL

from slickml.classification import XGBoostClassifier, XGBoostCVClassifier
from slickml.regression import XGBoostRegressor, XGBoostCVRegressor


class XGBoostClassifierBayesianOpt(XGBoostCVClassifier):
    """XGBoost Hyper-Parameters Tunning using Bayesian Optimization.
    This is wrapper using Bayesian Optimization to tune the parameters
    for XGBoost classifier using xgboost.cv() model with n-folds
    cross-validation iteratively. This function is pretty useful find
    the optimized set of parameters before training. Please note that,
    the optimizier objective is always to maximize the target. Therefore,
    in case of using a metric such as logloss or error, the negative value
    of the metric will be maximized.

    Parameters
    ----------
    n_iter: int, optional (default=5)
        Number of iteration rounds for hyper-parameters tuning

    init_points: int, optional (default=5)
        Number of initial points to initialize the optimizer

    acq: str, optional (default="ei")
        Type of acquisition function such as expected improvement (ei)

    pbounds: dict, optional
        Set of parameters boundaries for Bayesian Optimization
        (default={"max_depth" : (2, 7),
                  "learning_rate" : (0, 1),
                  "min_child_weight" : (1, 20),
                  "colsample_bytree": (0.1, 1.0)
                  "subsample" : (0.1, 1),
                  "gamma" : (0, 1),
                  "reg_alpha" : (0, 1),
                  "reg_lambda" : (0, 1)})

    num_boost_round: int, optional (default=200)
        Number of boosting round at each fold of xgboost.cv()

    n_splits: int, optional (default=4)
        Number of folds for cross-validation

    metrics: str or tuple[str], optional (default=("auc"))
        Metric used for evaluation at cross-validation
        using xgboost.cv(). Please note that this is different
        than eval_metric that needs to be passed to params dict.
        Possible values are "auc", "aucpr", "error", "logloss"

    objective: str, optional (default="binary:logistic")
        Type of objective function for classification

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

    scale_mean: bool, optional (default=False)
        Flag to center the data before scaling. This flag should be
        False when using sparse_matrix=True, since it centering the data
        would decrease the sparsity and in practice it does not make any
        sense to use sparse matrix method and it would make it worse.

    scale_std: bool, optional (default=False)
        Flag to scale the data to unit variance
        (or equivalently, unit standard deviation)

    importance_type: str, optional (default="total_gain")
        Importance type of xgboost.train() with possible values
        "weight", "gain", "total_gain", "cover", "total_cover"

    verbose: bool, optional (default=True)
        Flag to show the Bayesian Optimization progress at each iteration

    Attributes
    ----------
    scaler_: StandardScaler object
        Returns the scaler object if any of scale_mean or scale_std
        was passed True.

    X_train_: pandas.DataFrame
        Returns scaled training data set that passed if if any of
        scale_mean or scale_std was passed as True, else X_train.

    d_train_: xgboost.DMatrix object
        Returns the xgboost.DMatrix(X_train_, y_train)

    optimizer_: Bayesian Optimiziation object
        Returns the fitted optimizer on (X_train_, y_train)

    optimization_results_: Optimization results Pandas DataFrame()
        Returns all the optimization results including target and params

    best_params_: Tuned xgboost params dict
        Returns the tuned params dict

    best_performance_: Target value and tuned params Pandas DataFrame()
        Return the dataframe of the best performance

    fit(X_train, y_train): instance method
        Returns None and applies the optimization process using
        the (X_train, y_train) set using xgboost.cv() and Bayesian Opt

    plot_optimization_results(): instance method
        Plot all the optimization results
    """

    def __init__(
        self,
        n_iter=None,
        init_points=None,
        acq=None,
        pbounds=None,
        num_boost_round=None,
        n_splits=None,
        metrics=None,
        objective=None,
        early_stopping_rounds=None,
        random_state=None,
        stratified=True,
        shuffle=True,
        sparse_matrix=False,
        scale_mean=False,
        scale_std=False,
        importance_type=None,
        verbose=True,
    ):
        super().__init__(
            num_boost_round,
            n_splits,
            metrics,
            early_stopping_rounds,
            random_state,
            stratified,
            shuffle,
            sparse_matrix,
            scale_mean,
            scale_std,
            importance_type,
        )

        if n_iter is None:
            self.n_iter = 5
        else:
            if not isinstance(n_iter, int):
                raise TypeError("The input n_iter must have integer dtype.")
            else:
                self.n_iter = n_iter

        if init_points is None:
            self.init_points = 5
        else:
            if not isinstance(init_points, int):
                raise TypeError("The input init_points must have integer dtype.")
            else:
                self.init_points = init_points

        if acq is None:
            self.acq = "ei"
        else:
            if not isinstance(acq, str):
                raise TypeError("The input acq must have str dtype.")
            else:
                self.acq = acq

        pbounds_ = {
            "max_depth": (2, 7),
            "learning_rate": (0, 1),
            "min_child_weight": (1, 20),
            "colsample_bytree": (0.1, 1.0),
            "subsample": (0.1, 1),
            "gamma": (0, 1),
            "reg_alpha": (0, 1),
            "reg_lambda": (0, 1),
        }
        if pbounds is None:
            self.pbounds = pbounds_
        else:
            if not isinstance(pbounds, dict):
                raise TypeError("The input pbounds must have dict dtype.")
            else:
                self.pbounds = pbounds_
                for key, val in pbounds.items():
                    self.pbounds[key] = val

        if objective is None:
            self.objective = "binary:logistic"
        else:
            if not isinstance(objective, str):
                raise TypeError("The input objective must be a str dtype.")
            else:
                self.objective = objective

        if not isinstance(verbose, bool):
            raise TypeError("The input verbose must have bool dtype.")
        else:
            if verbose:
                self.verbose = 2
            else:
                self.verbose = 0

    def fit(self, X_train, y_train):
        """
        Function to run xgboost.cv() method first to find the best number of boosting round
        and train a model based on that on (X_train, y_train) set and returns it.
        """

        # helper function for xgboost eval
        def _xgb_eval(
            max_depth,
            subsample,
            colsample_bytree,
            min_child_weight,
            learning_rate,
            gamma,
            reg_alpha,
            reg_lambda,
        ):
            """
            Helper Function to eval bayesian optimization
            """
            params = {
                "eval_metric": "auc",
                "tree_method": "hist",
                "objective": self.objective,
                "max_delta_step": 1,
                "nthread": 4,
                "scale_pos_weight": 1,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "learning_rate": learning_rate,
                "max_depth": int(max_depth),
                "min_child_weight": min_child_weight,
                "gamma": gamma,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }

            cv_result = xgb.cv(
                params=params,
                dtrain=self.dtrain_,
                num_boost_round=self.num_boost_round,
                nfold=self.n_splits,
                stratified=self.stratified,
                metrics=self.metrics,
                early_stopping_rounds=self.early_stopping_rounds,
                seed=self.random_state,
                shuffle=True,
            )

            # set to return + or - results based on metric for maximization
            if self.metrics in ["logloss", "error"]:
                return (-1) * cv_result.iloc[-1][2]
            else:
                return cv_result.iloc[-1][2]

        # creating dtrain
        self.dtrain_ = self._dtrain(X_train, y_train)

        # xgb_bo definition
        self.optimizer_ = BayesianOptimization(
            f=_xgb_eval,
            pbounds=self.pbounds,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # maximizing xgb_bo
        self.optimizer_.maximize(
            init_points=self.init_points, n_iter=self.n_iter, acq=self.acq
        )

        # initiate the results
        self.optimization_results_ = self.get_optimization_results()

        # initiate the best params
        self.best_params_ = self.get_best_params()

        # initiate the best performance
        self.best_performance_ = self.get_best_performance()

        return None

    def get_pbounds(self):
        """
        Function to return the hyper-parameters bounds.
        """

        return self.pbounds

    def get_optimizer(self):
        """
        Function to return the Bayesian Optimization object.
        """

        return self.optimizer_

    def get_optimization_results(self):
        """
        Function to return the optimization results.
        """
        frames = []
        for idx, res in enumerate(self.optimizer_.res):
            d = res["params"]
            d[self.metrics] = res["target"]
            frames.append(pd.DataFrame(data=d, index=[idx]))

        df_res = pd.concat(frames)

        return df_res

    def get_best_params(self):
        """
        Function to return the best (tuned) set of hyper-parameters.
        """
        targets = []
        for i, rs in enumerate(self.optimizer_.res):
            targets.append(rs["target"])

        best_params = self.optimizer_.res[targets.index(max(targets))]["params"]
        best_params["max_depth"] = int(best_params["max_depth"])

        return best_params

    def get_best_performance(self):
        """
        Function to return the performance of the best (tuned)
        set of hyper-parameters.
        """
        best_performance = self.optimization_results_.loc[
            self.optimization_results_[self.metrics]
            == self.optimization_results_[self.metrics].max(),
            :,
        ]
        best_performance.reset_index(drop=True, inplace=True)

        return best_performance

    # TODO: Plotting
    def plot_optimization_results(self):
        """
        Function to plot the optimization results.
        """
        pass


class XGBoostClassifierHyperOpt(XGBoostClassifier):
    """XGBoost Classifier - Hyperparameter Optimization.
    This class uses HyperOpt, a Python library for serial and parallel
    optimization over search spaces, which may include real-valued, discrete,
    and conditional dimensions to train a XGBoost model.Main reference is
    HyperOpt GitHub: (https://github.com/hyperopt/hyperopt)

    Parameters
    ----------
    num_boost_round: int, optional (default=200)
        Number of boosting round at each fold of xgboost.cv()

    n_splits: int, optional (default=4)
        Number of folds for cross-validation

    metrics: str or tuple[str], optional (default=("auc"))
        Metric used for evaluation at cross-validation
        using xgboost.cv(). Please note that this is different
        than eval_metric that needs to be passed to params dict.
        Possible values are "auc", "aucpr", "error", "logloss"

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

    scale_mean: bool, optional (default=False)
        Flag to center the data before scaling. This flag should be
        False when using sparse_matrix=True, since it centering the data
        would decrease the sparsity and in practice it does not make any
        sense to use sparse matrix method and it would make it worse.

    scale_std: bool, optional (default=False)
        Flag to scale the data to unit variance
        (or equivalently, unit standard deviation)

    func_name: str
        Function name for performing optimization

    space: dict()
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp

    trials: HyperOpt Object
        Storage for completed, ongoing, and scheduled evaluation points.
        If None, then a temporary `base.Trials` instance will be created.
        If a trials object, then that trials object will be affected by
        side-effect of this call

    algo: HyperOpt Object
        Provides logic for sequential search of the hyperparameter space

    max_evals: int
        Storage for completed, ongoing, and scheduled evaluation points.
        If None, then a temporary `base.Trials` instance will be created.
        If a trials object, then that trials object will be affected by
        side-effect of this call

    verbose: str, optional (default=False)
        Print evaluation results from model

    Attributes
    ----------
    fit(X_train, y_train): instance method
        Returns None and applies the tuning process using
        the (X_train, y_train) and the given set of hyperparameters

    xgb_cv(space): instance method
        Optimization function for XGBoost utilizing cross-validation
        based on space params

    get_optimization_results(): instance method
        Returns pd.DataFrame for best parameters from all runs

    get_optimization_trials(): instance method
        Returns dict of best parameters for each individual trial run
    """

    def __init__(
        self,
        num_boost_round=None,
        n_splits=None,
        metrics=None,
        early_stopping_rounds=None,
        random_state=None,
        stratified=True,
        shuffle=True,
        sparse_matrix=False,
        scale_mean=False,
        scale_std=False,
        func_name=None,
        space=None,
        max_evals=None,
        verbose=False,
    ):
        super().__init__(
            num_boost_round,
            metrics,
            sparse_matrix,
            scale_mean,
            scale_std,
        )

        if n_splits is None:
            self.n_splits = 4
        else:
            if not isinstance(n_splits, int):
                raise TypeError("The input n_splits must have integer dtype.")
            else:
                self.n_splits = n_splits

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

        if isinstance(func_name, str):
            self.fn = getattr(self, func_name)
        else:
            raise TypeError("The input must be a valid function name: 'xgb_cv'.")

        if isinstance(space, dict):
            self.space = space
        else:
            raise TypeError("The input space must have dict dtype.")

        if max_evals is None:
            self.max_evals = 100
        else:
            if not isinstance(max_evals, int):
                raise TypeError("The input max_evals must have integer dtype.")
            else:
                self.max_evals = max_evals

        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise TypeError("The input verbose must be a boolean")

    def fit(self, X_train, y_train):
        """
        Fit model for a given a hyperparameter space
        according to a given algorithm, allowing up to a certain number of
        function evaluations.

        Parameters
        ----------
        X_train: numpy.array or pandas.DataFrame
            Training features data

        y_train: numpy.array[int] or list[int]
            List of training ground truth binary values [0, 1]
        """
        # define train set
        self.dtrain_ = self._dtrain(X_train, y_train)

        # define algo
        self.algo = tpe.suggest

        # define Trials()
        self.trials = Trials()

        try:
            self.optimization_results_ = fmin(
                fn=self.fn,
                space=self.space,
                algo=self.algo,
                max_evals=self.max_evals,
                trials=self.trials,
            )
        except Exception as e:
            self.optimization_results_ = {"status": STATUS_FAIL, "exception": str(e)}

    def xgb_cv(self, space):
        """
        Function to perform XGBoost Cross-Validation with stochastic parameters
        for hyperparameter optimization.

        Parameters
        ----------
        space: dict()
            The set of possible arguments to `fn` is the set of objects
            that could be created with non-zero probability by drawing randomly
            from this stochastic program involving involving hp
        """

        # cvr
        cvr = xgb.cv(
            params=sample(self.space),
            dtrain=self.dtrain_,
            num_boost_round=self.num_boost_round,
            nfold=self.n_splits,
            stratified=self.stratified,
            metrics=self.metrics,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.random_state,
            shuffle=self.shuffle,
            verbose_eval=self.verbose,
        )

        # loss
        loss = cvr.iloc[-1:, 0]

        return {"loss": loss, "status": STATUS_OK}

    def get_optimization_results(self):
        """
        Function to return pd.DataFrame with best results
        """

        return pd.DataFrame(self.optimization_results_, index=[0])

    def get_optimization_trials(self):
        """
        Function to return dict results from all 'n' trials
        """

        return self.trials


class XGBoostRegressorBayesianOpt(XGBoostCVRegressor):
    """XGBoost Hyper-Parameters Tunning using Bayesian Optimization.
    This is wrapper using Bayesian Optimization to tune the parameters
    for XGBoost Regressor using xgboost.cv() model with n-folds
    cross-validation iteratively. This function is pretty useful find
    the optimized set of parameters before training. Please note that,
    the optimizier objective is always to maximize the target. Therefore,
    in case of using a metric such as logloss or error, the negative value
    of the metric will be maximized.

    Parameters
    ----------
    n_iter: int, optional (default=5)
        Number of iteration rounds for hyper-parameters tuning

    init_points: int, optional (default=5)
        Number of initial points to initialize the optimizer

    acq: str, optional (default="ei")
        Type of acquisition function such as expected improvement (ei)

    pbounds: dict, optional
        Set of parameters boundaries for Bayesian Optimization
        (default={"max_depth" : (2, 7),
                  "learning_rate" : (0, 1),
                  "min_child_weight" : (1, 20),
                  "colsample_bytree": (0.1, 1.0)
                  "subsample" : (0.1, 1),
                  "gamma" : (0, 1),
                  "reg_alpha" : (0, 1),
                  "reg_lambda" : (0, 1)})

    num_boost_round: int, optional (default=200)
        Number of boosting round at each fold of xgboost.cv()

    n_splits: int, optional (default=4)
        Number of folds for cross-validation

    metrics: str or tuple[str], optional (default=("rmse"))
        Metric used for evaluation at cross-validation
        using xgboost.cv(). Please note that this is different
        than eval_metric that needs to be passed to params dict.
        Possible values are "rmse", "rmsle", "mae"

    objective: str, optional (default="reg:squarederror")
        Type of objective function regression;
        Other options for objective: "reg:logistic", "reg:squaredlogerror"

    early_stopping_rounds: int, optional (default=20)
        The criterion to early abort the xgboost.cv() phase
        if the test metric is not improved

    random_state: int, optional (default=1367)
        Random seed

    shuffle: bool, optional (default=True)
        Flag to shuffle data to have the ability of building
        stratified folds in xgboost.cv()

    sparse_matrix: bool, optional (default=False)
        Flag to convert data to sparse matrix with csr format.
        This would increase the speed of feature selection for
        relatively large datasets

    scale_mean: bool, optional (default=False)
        Flag to center the data before scaling. This flag should be
        False when using sparse_matrix=True, since it centering the data
        would decrease the sparsity and in practice it does not make any
        sense to use sparse matrix method and it would make it worse.

    scale_std: bool, optional (default=False)
        Flag to scale the data to unit variance
        (or equivalently, unit standard deviation)

    importance_type: str, optional (default="total_gain")
        Importance type of xgboost.train() with possible values
        "weight", "gain", "total_gain", "cover", "total_cover"

    verbose: bool, optional (default=True)
        Flag to show the Bayesian Optimization progress at each iteration

    Attributes
    ----------
    scaler_: StandardScaler object
        Returns the scaler object if any of scale_mean or scale_std
        was passed True.

    X_train_: pandas.DataFrame
        Returns scaled training data set that passed if if any of
        scale_mean or scale_std was passed as True, else X_train.

    d_train_: xgboost.DMatrix object
        Returns the xgboost.DMatrix(X_train_, y_train)

    optimizer_: Bayesian Optimiziation object
        Returns the fitted optimizer on (X_train_, y_train)

    optimization_results_: Optimization results Pandas DataFrame()
        Returns all the optimization results including target and params

    best_params_: Tuned xgboost params dict
        Returns the tuned params dict

    best_performance_: Target value and tuned params Pandas DataFrame()
        Return the dataframe of the best performance

    fit(X_train, y_train): instance method
        Returns None and applies the optimization process using
        the (X_train, y_train) set using xgboost.cv() and Bayesian Opt

    plot_optimization_results(): instance method
        Plot all the optimization results
    """

    def __init__(
        self,
        n_iter=None,
        init_points=None,
        acq=None,
        pbounds=None,
        num_boost_round=None,
        n_splits=None,
        metrics=None,
        objective=None,
        early_stopping_rounds=None,
        random_state=None,
        shuffle=True,
        sparse_matrix=False,
        scale_mean=False,
        scale_std=False,
        importance_type=None,
        verbose=True,
    ):
        super().__init__(
            num_boost_round,
            n_splits,
            metrics,
            early_stopping_rounds,
            random_state,
            shuffle,
            sparse_matrix,
            scale_mean,
            scale_std,
            importance_type,
        )

        if n_iter is None:
            self.n_iter = 5
        else:
            if not isinstance(n_iter, int):
                raise TypeError("The input n_iter must have integer dtype.")
            else:
                self.n_iter = n_iter

        if init_points is None:
            self.init_points = 5
        else:
            if not isinstance(init_points, int):
                raise TypeError("The input init_points must have integer dtype.")
            else:
                self.init_points = init_points

        if acq is None:
            self.acq = "ei"
        else:
            if not isinstance(acq, str):
                raise TypeError("The input acq must have str dtype.")
            else:
                self.acq = acq

        pbounds_ = {
            "max_depth": (2, 7),
            "learning_rate": (0, 1),
            "min_child_weight": (1, 20),
            "colsample_bytree": (0.1, 1.0),
            "subsample": (0.1, 1),
            "gamma": (0, 1),
            "reg_alpha": (0, 1),
            "reg_lambda": (0, 1),
        }
        if pbounds is None:
            self.pbounds = pbounds_
        else:
            if not isinstance(pbounds, dict):
                raise TypeError("The input pbounds must have dict dtype.")
            else:
                self.pbounds = pbounds_
                for key, val in pbounds.items():
                    self.pbounds[key] = val

        if objective is None:
            self.objective = "reg:squarederror"
        else:
            if not isinstance(objective, str):
                raise TypeError("The input objective must be a str dtype.")
            else:
                self.objective = objective

        if not isinstance(verbose, bool):
            raise TypeError("The input verbose must have bool dtype.")
        else:
            if verbose:
                self.verbose = 2
            else:
                self.verbose = 0

    def fit(self, X_train, y_train):
        """
        Function to run xgboost.cv() method first to find the best number of boosting round
        and train a model based on that on (X_train, y_train) set and returns it.
        """

        # helper function for xgboost eval
        def _xgb_eval(
            max_depth,
            subsample,
            colsample_bytree,
            min_child_weight,
            learning_rate,
            gamma,
            reg_alpha,
            reg_lambda,
        ):
            """
            Helper Function to eval bayesian optimization
            """
            params = {
                "eval_metric": "rmse",
                "tree_method": "hist",
                "objective": self.objective,
                "max_delta_step": 1,
                "nthread": 4,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "learning_rate": learning_rate,
                "max_depth": int(max_depth),
                "min_child_weight": min_child_weight,
                "gamma": gamma,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }

            cv_result = xgb.cv(
                params=params,
                dtrain=self.dtrain_,
                num_boost_round=self.num_boost_round,
                nfold=self.n_splits,
                metrics=self.metrics,
                early_stopping_rounds=self.early_stopping_rounds,
                seed=self.random_state,
                shuffle=True,
            )

            return (-1) * cv_result.iloc[-1][2]

        # creating dtrain
        self.dtrain_ = self._dtrain(X_train, y_train)

        # xgb_bo definition
        self.optimizer_ = BayesianOptimization(
            f=_xgb_eval,
            pbounds=self.pbounds,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # maximizing xgb_bo
        self.optimizer_.maximize(
            init_points=self.init_points, n_iter=self.n_iter, acq=self.acq
        )

        # initiate the results
        self.optimization_results_ = self.get_optimization_results()

        # initiate the best params
        self.best_params_ = self.get_best_params()

        # initiate the best performance
        self.best_performance_ = self.get_best_performance()

        return None

    def get_pbounds(self):
        """
        Function to return the hyper-parameters bounds.
        """

        return self.pbounds

    def get_optimizer(self):
        """
        Function to return the Bayesian Optimization object.
        """

        return self.optimizer_

    def get_optimization_results(self):
        """
        Function to return the optimization results.
        """
        frames = []
        for idx, res in enumerate(self.optimizer_.res):
            d = res["params"]
            d[self.metrics] = res["target"]
            frames.append(pd.DataFrame(data=d, index=[idx]))

        df_res = pd.concat(frames)

        return df_res

    def get_best_params(self):
        """
        Function to return the best (tuned) set of hyper-parameters.
        """
        targets = []
        for i, rs in enumerate(self.optimizer_.res):
            targets.append(rs["target"])

        best_params = self.optimizer_.res[targets.index(max(targets))]["params"]
        best_params["max_depth"] = int(best_params["max_depth"])

        return best_params

    def get_best_performance(self):
        """
        Function to return the performance of the best (tuned)
        set of hyper-parameters.
        """
        best_performance = self.optimization_results_.loc[
            self.optimization_results_[self.metrics]
            == self.optimization_results_[self.metrics].max(),
            :,
        ]
        best_performance.reset_index(drop=True, inplace=True)

        return best_performance

    # TODO: Plotting
    def plot_optimization_results(self):
        """
        Function to plot the optimization results.
        """
        pass


class XGBoostRegressorHyperOpt(XGBoostRegressor):
    """XGBoost Regressor - Hyperparameter Optimization.
    This class uses HyperOpt, a Python library for serial and parallel
    optimization over search spaces, which may include real-valued, discrete,
    and conditional dimensions to train a XGBoost model.Main reference is
    HyperOpt GitHub: (https://github.com/hyperopt/hyperopt)

    Parameters
    ----------
    num_boost_round: int, optional (default=200)
        Number of boosting round at each fold of xgboost.cv()

    n_splits: int, optional (default=4)
        Number of folds for cross-validation

    metrics: str or tuple[str], optional (default=("rmse"))
        Metric used for evaluation at cross-validation
        using xgboost.cv(). Please note that this is different
        than eval_metric that needs to be passed to params dict.
        Possible values are "rmse", "rmsle", "mae"

    early_stopping_rounds: int, optional (default=20)
        The criterion to early abort the xgboost.cv() phase
        if the test metric is not improved

    random_state: int, optional (default=1367)
        Random seed

    shuffle: bool, optional (default=True)
        Flag to shuffle data to have the ability of building
        stratified folds in xgboost.cv()

    sparse_matrix: bool, optional (default=False)
        Flag to convert data to sparse matrix with csr format.
        This would increase the speed of feature selection for
        relatively large datasets

    scale_mean: bool, optional (default=False)
        Flag to center the data before scaling. This flag should be
        False when using sparse_matrix=True, since it centering the data
        would decrease the sparsity and in practice it does not make any
        sense to use sparse matrix method and it would make it worse.

    scale_std: bool, optional (default=False)
        Flag to scale the data to unit variance
        (or equivalently, unit standard deviation)

    func_name: str
        Function name for performing optimization

    space: dict()
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp

    trials: HyperOpt Object
        Storage for completed, ongoing, and scheduled evaluation points.
        If None, then a temporary `base.Trials` instance will be created.
        If a trials object, then that trials object will be affected by
        side-effect of this call

    algo: HyperOpt Object
        Provides logic for sequential search of the hyperparameter space

    max_evals: int
        Storage for completed, ongoing, and scheduled evaluation points.
        If None, then a temporary `base.Trials` instance will be created.
        If a trials object, then that trials object will be affected by
        side-effect of this call

    verbose: str, optional (default=False)
        Print evaluation results from model

    Attributes
    ----------
    fit(X_train, y_train): instance method
        Returns None and applies the tuning process using
        the (X_train, y_train) and the given set of hyperparameters

    xgb_cv(space): instance method
        Optimization function for XGBoost utilizing cross-validation
        based on space params

    get_optimization_results(): instance method
        Returns pd.DataFrame for best parameters from all runs

    get_optimization_trials(): instance method
        Returns dict of best parameters for each individual trial run
    """

    def __init__(
        self,
        num_boost_round=None,
        n_splits=None,
        metrics=None,
        early_stopping_rounds=None,
        random_state=None,
        shuffle=True,
        sparse_matrix=False,
        scale_mean=False,
        scale_std=False,
        func_name=None,
        space=None,
        max_evals=None,
        verbose=False,
    ):
        super().__init__(
            num_boost_round,
            metrics,
            sparse_matrix,
            scale_mean,
            scale_std,
        )

        if n_splits is None:
            self.n_splits = 4
        else:
            if not isinstance(n_splits, int):
                raise TypeError("The input n_splits must have integer dtype.")
            else:
                self.n_splits = n_splits

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

        if not isinstance(shuffle, bool):
            raise TypeError("The input shuffle must have bool dtype.")
        else:
            self.shuffle = shuffle

        if isinstance(func_name, str):
            self.fn = getattr(self, func_name)
        else:
            raise TypeError("The input must be a valid function name: 'xgb_cv'.")

        if isinstance(space, dict):
            self.space = space
        else:
            raise TypeError("The input space must have dict dtype.")

        if max_evals is None:
            self.max_evals = 100
        else:
            if not isinstance(max_evals, int):
                raise TypeError("The input max_evals must have integer dtype.")
            else:
                self.max_evals = max_evals

        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise TypeError("The input verbose must be a boolean")

    def fit(self, X_train, y_train):
        """
        Fit model for a given a hyperparameter space
        according to a given algorithm, allowing up to a certain number of
        function evaluations.

        Parameters
        ----------
        X_train: numpy.array or pandas.DataFrame
            Training features data

        y_train: numpy.array[int] or list[int]
            List of training ground truth binary values [0, 1]
        """
        # define train set
        self.dtrain_ = self._dtrain(X_train, y_train)

        # define algo
        self.algo = tpe.suggest

        # define Trials()
        self.trials = Trials()

        try:
            self.optimization_results_ = fmin(
                fn=self.fn,
                space=self.space,
                algo=self.algo,
                max_evals=self.max_evals,
                trials=self.trials,
            )
        except Exception as e:
            self.optimization_results_ = {"status": STATUS_FAIL, "exception": str(e)}

    def xgb_cv(self, space):
        """
        Function to perform XGBoost Cross-Validation with stochastic parameters
        for hyperparameter optimization.

        Parameters
        ----------
        space: dict()
            The set of possible arguments to `fn` is the set of objects
            that could be created with non-zero probability by drawing randomly
            from this stochastic program involving involving hp
        """

        # cvr
        cvr = xgb.cv(
            params=sample(self.space),
            dtrain=self.dtrain_,
            num_boost_round=self.num_boost_round,
            nfold=self.n_splits,
            metrics=self.metrics,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.random_state,
            shuffle=self.shuffle,
            verbose_eval=self.verbose,
        )

        # loss
        loss = cvr.iloc[-1:, 0]

        return {"loss": loss, "status": STATUS_OK}

    def get_optimization_results(self):
        """
        Function to return pd.DataFrame with best results
        """

        return pd.DataFrame(self.optimization_results_, index=[0])

    def get_optimization_trials(self):
        """
        Function to return dict results from all 'n' trials
        """

        return self.trials
