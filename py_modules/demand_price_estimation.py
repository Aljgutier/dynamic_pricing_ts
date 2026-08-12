"""
Demand Estimation Module

- fit_demand_estimator
- vhat_calculator
- get_demand
- _piecewise_regions
- _piecewise_region_index
- piecewise_demand_fit
- select_demand_model
- aggregate_price_demand
- mab_price_optimization_loop
"""

from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RidgeCV,
)


import numpy as np

if __package__:
    # Package-style import (e.g., from py_modules.demand_estimation import ...)
    from .demand_elasticity_profit import (  # pylint: disable=relative-beyond-top-level
        demand_logit,
        demand_constant_elasticity,
        demand_linear,
    )
else:
    # Top-level import (e.g., when py_modules is added to sys.path in notebooks)
    from demand_elasticity_profit import (
        demand_logit,
        demand_constant_elasticity,
        demand_linear,
    )


# ----------------- fit with tilt penalty ------------------#


def _fit_tilt_penalty(
    features,
    target,
    p,
    sample_weights=None,
    tilt_penalty=1.0,
):
    """
    Fit a regression model with a residual-tilt penalty.

    The fit minimizes

        weighted RSS
        + tilt_penalty * (mean_residual_low - mean_residual_high)^2

    where the low and high residual means are calculated over the
    lower and upper thirds of the observed price range.

    This is solved exactly as an augmented weighted least-squares
    problem; no iterative numerical optimizer is required.

    Parameters
    ----------
    features : ndarray
        Regression design matrix excluding the intercept.

    target : ndarray
        Regression response.

        For linear demand:
            target = d

        For log-quadratic demand:
            target = log(d)

    p : array-like
        Observed prices.

    sample_weights : array-like or None, default=None
        Optional regression sample weights.

    tilt_penalty : float, default=1.0
        Strength of the residual-tilt penalty.

    Returns
    -------
    model : LinearRegression
        Model object containing the fitted intercept and coefficients.

    tilt_error : float
        Difference between weighted mean residuals in the low-price
        and high-price regions.
    """

    features = np.asarray(
        features,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    N = len(target)

    # ---------- Observation weights ----------#

    if sample_weights is None:

        weights = np.ones(
            N,
            dtype=float,
        )

    else:

        weights = np.asarray(
            sample_weights,
            dtype=float,
        )

    # ---------- Add intercept column ----------#

    X = np.column_stack(
        (
            np.ones(N),
            features,
        )
    )

    # ---------- Low / high price regions ----------#

    p_low_cut = np.quantile(
        p,
        1.0 / 3.0,
    )

    p_high_cut = np.quantile(
        p,
        2.0 / 3.0,
    )

    low_mask = p <= p_low_cut
    high_mask = p >= p_high_cut

    # ---------- Construct tilt vector ----------#
    #
    # tilt_error = a @ residual
    #
    # where a produces:
    #
    # weighted mean residual_low
    # -
    # weighted mean residual_high

    a = np.zeros(
        N,
        dtype=float,
    )

    weight_low = np.sum(weights[low_mask])

    weight_high = np.sum(weights[high_mask])

    if weight_low <= 0 or weight_high <= 0:

        raise ValueError("Low and high price regions must have positive weight.")

    a[low_mask] = weights[low_mask] / weight_low

    a[high_mask] -= weights[high_mask] / weight_high

    # ---------- Weighted least squares ----------#

    sqrt_weights = np.sqrt(weights)

    X_weighted = X * sqrt_weights[:, None]

    y_weighted = target * sqrt_weights

    # ---------- Add tilt penalty as pseudo-observation ----------#
    #
    # tilt_error = a @ (target - X @ theta)
    #
    # We want the objective to be proportional to:
    #
    #     weighted_mean_squared_error
    #     + tilt_penalty * tilt_error**2
    #
    # Multiplying the pseudo-observation by sqrt(total_weight)
    # puts the tilt penalty on the same scale as the weighted
    # mean squared residual error and prevents its relative
    # strength from shrinking as the sample size increases.

    total_weight = np.sum(weights)

    sqrt_tilt_penalty = np.sqrt(tilt_penalty * total_weight)

    X_tilt = sqrt_tilt_penalty * (a @ X)

    y_tilt = sqrt_tilt_penalty * (a @ target)

    X_augmented = np.vstack(
        (
            X_weighted,
            X_tilt,
        )
    )

    y_augmented = np.concatenate(
        (
            y_weighted,
            [y_tilt],
        )
    )

    # ---------- Exact least-squares solution ----------#

    theta, _, _, _ = np.linalg.lstsq(
        X_augmented,
        y_augmented,
        rcond=None,
    )

    # ---------- Store solution in sklearn model ----------#

    model = LinearRegression()

    model.intercept_ = float(theta[0])

    model.coef_ = np.asarray(
        theta[1:],
        dtype=float,
    )

    model.n_features_in_ = features.shape[1]

    # ---------- Final residual tilt diagnostic ----------#

    target_hat = model.predict(features)

    residual = target - target_hat

    tilt_error = float(a @ residual)

    return model, tilt_error


# ---------------  demand estimator -----------------#
def fit_demand_estimator(
    p,
    d,
    x=None,
    model_type="log_quadratic",
    sample_weights=None,
    tilt_penalty_linear=0.0,
    tilt_penalty_log_log=0.0,
    tilt_penalty_log_quadratic=0.0,
    ridge_cv=False,
    huber=False,
    huber_epsilon=1.35,
):
    """
    Fits several demand models. Supported models include linear, log-level, log-log, and log-quadratic.

    The function returns a callable demand prediction function and a dictionary containing the fitted regression model and supporting information.

    Process for selecting the best model is handled by the select_demand_model() function, which evaluates multiple candidate models and selects the one with the smallest combined selection score as follows:

    * See the select_model funtion for selectng the bset model for providing a good elasticity estimate, which in turn can be used to estimate profit.
    * Or, the best model can simply be selected based on the lowest in-sample fit error (e.g., log-RMSE) without regard to elasticity.


    Supported demand models
    -----------------------

    linear:
        d = beta_0 + beta_1 * p + gamma' x

    log_level:
        log(d) = beta_0 + beta_1 * p + gamma' x

    log_log:
        log(d) = beta_0 + beta_1 * log(p) + gamma' x

        This model is mathematically equivalent to a constant-elasticity
        demand model

            d = A * p^(-v),

        where

            A = exp(beta_0)
            v = -beta_1.

        Consequently, the implied price elasticity is constant and equal
        to -beta_1.

    log_quadratic:
        log(d) = beta_0
               + beta_1 * log(p)
               + beta_2 * log(p)^2
               + gamma' x

        This generalizes the constant-elasticity model by allowing the
        elasticity to vary smoothly with price:

            v(p) = -(beta_1 + 2 * beta_2 * log(p)).

    Parameters
    ----------
    p : array-like
        Historical prices.

    d : array-like
        Historical demand observations.

    x : array-like or None, default=None
        Optional contextual variables. Each row corresponds to one
        observation.

    model_type : {"linear", "log_level", "log_log", "log_quadratic"},
        default="log_quadratic"

    sample_weights : array-like or None, default=None
        Optional regression sample weights.

    alpha : float, default=0.0
        Ridge regularization parameter.
        alpha = 0 uses ordinary least squares.

    Returns
    -------
    dhat_model : callable
        Demand prediction function
            dhat_model(p_new, x_new=None)

    model_info : dict
        Dictionary containing the fitted regression model and
        supporting information.
    """

    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)

    # ----------------- validate inputs --------------
    if ridge_cv and huber:
        raise ValueError("ridge_cv and huber cannot both be True.")

    if len(p) != len(d):
        raise ValueError("p and d must have the same length.")

    if np.any(p <= 0):
        raise ValueError("Prices must be positive.")

    if model_type in ("log_level", "log_log", "log_quadratic"):

        if np.any(d <= 0):
            raise ValueError("Demand must be positive for logarithmic models.")

    # minimum observations
    if model_type == "log_quadratic":

        if len(p) < 3:
            raise ValueError(
                "At least three observations are required "
                "for model_type='log_quadratic'."
            )

    elif model_type in ("linear", "log_level", "log_log"):

        if len(p) < 2:
            raise ValueError(
                f"At least two observations are required "
                f"for model_type='{model_type}'."
            )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}.")

    # --------Helper: construct regression response ----------------------

    def _response(d_values):
        """
        Construct the regression response variable.
        """

        if model_type == "linear":
            return d_values

        return np.log(d_values)

    # -------- Helper: construct price basis functions ------------------

    def _price_basis(p_values):
        """
        Construct the regression basis functions associated with
        price.
        """

        p_values = np.asarray(p_values, dtype=float)  # price values

        if np.any(p_values <= 0):
            raise ValueError("Prices must be positive.")

        if model_type in ("linear", "log_level"):
            return p_values.reshape(-1, 1)

        elif model_type == "log_log":
            log_p = np.log(p_values).reshape(-1, 1)
            return log_p

        elif model_type == "log_quadratic":
            log_p = np.log(p_values).reshape(-1, 1)

            return np.column_stack(
                (
                    log_p,
                    log_p**2,
                )
            )

        raise ValueError(f"Unsupported model_type: {model_type}.")

    # ----- Helper: transform regression output back to demand  ------------ %

    def _inverse_response(y):
        """
        Transform regression predictions back to demand.
        """

        if model_type == "linear":
            # demand cannot be negative
            return np.maximum(y, 0.0)

        return np.exp(y)

    # --------Construct regression response and price basis  ------------#

    target = _response(d)

    price_basis = _price_basis(p)

    # ------- Optional contextual variables -----------------#

    n_x_features = 0

    if x is not None:

        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if len(x) != len(p):
            raise ValueError("x must have the same number of rows as p.")

        n_x_features = x.shape[1]

        features = np.column_stack(
            (
                price_basis,
                x,
            )
        )

    else:

        features = price_basis

    # ---------- Select regression estimator ----------#

    if huber:

        model = HuberRegressor(
            epsilon=huber_epsilon,
            alpha=0.0,
            max_iter=1000,
        )

    elif ridge_cv:

        model = RidgeCV(alphas=np.logspace(-4, 2, 25))

    else:

        model = LinearRegression()

    # ---------- Select and fit regression estimator ----------#

    if model_type == "linear":
        tilt_penalty = tilt_penalty_linear

    elif model_type == "log_log":
        tilt_penalty = tilt_penalty_log_log

    elif model_type == "log_quadratic":
        tilt_penalty = tilt_penalty_log_quadratic

    else:
        tilt_penalty = 0.0

    tilt_error = np.nan

    if (
        tilt_penalty > 0
        and model_type in ("linear", "log-log", "log_quadratic")
        and not ridge_cv
        and not huber
    ):

        model, tilt_error = _fit_tilt_penalty(
            features=features,
            target=target,
            p=p,
            sample_weights=sample_weights,
            tilt_penalty=tilt_penalty,
        )

    else:

        if huber:

            model = HuberRegressor(
                epsilon=huber_epsilon,
                alpha=0.0,
                max_iter=1000,
            )

        elif ridge_cv:

            model = RidgeCV(alphas=np.logspace(-4, 2, 25))

        else:

            model = LinearRegression()

        model.fit(
            features,
            target,
            sample_weight=sample_weights,
        )

    # ---------- Final residual tilt diagnostic ----------#

    target_hat = model.predict(features)

    residual = target - target_hat

    p_low_cut = np.quantile(
        p,
        1.0 / 3.0,
    )

    p_high_cut = np.quantile(
        p,
        2.0 / 3.0,
    )

    low_mask = p <= p_low_cut
    high_mask = p >= p_high_cut

    if sample_weights is None:
        weights = np.ones(
            len(p),
            dtype=float,
        )
    else:
        weights = np.asarray(
            sample_weights,
            dtype=float,
        )

    residual_low = np.average(
        residual[low_mask],
        weights=weights[low_mask],
    )

    residual_high = np.average(
        residual[high_mask],
        weights=weights[high_mask],
    )

    tilt_error = float(residual_low - residual_high)

    # ------  Demand prediction function -------------------

    def dhat_model(p_new, x_new=None):

        scalar_input = np.isscalar(p_new)
        p_new = np.atleast_1d(np.asarray(p_new, dtype=float))

        price_basis_new = _price_basis(p_new)

        # ---------- Optional contextual feature variables -----------

        if n_x_features > 0:

            if x_new is None:
                raise ValueError(
                    "x_new is required because the model "
                    "was fitted with contextual variables."
                )

            x_new = np.asarray(x_new, dtype=float)

            if x_new.ndim == 1:

                if len(p_new) == 1:
                    x_new = x_new.reshape(1, -1)
                else:
                    x_new = x_new.reshape(-1, 1)

            if len(x_new) != len(p_new):
                raise ValueError("x_new must have the same number of rows as p_new.")

            features_new = np.column_stack(
                (
                    price_basis_new,
                    x_new,
                )
            )

        else:

            features_new = price_basis_new

        # ------------ Predict demand -----------------#

        prediction = model.predict(features_new)

        d_hat = _inverse_response(prediction)

        if scalar_input:
            return float(d_hat[0])

        return d_hat

    # -----Store model information --------------------#

    model_info = {
        # fitted demand prediction function
        "dhat_model": dhat_model,
        # fitted regression model
        "model": model,
        # model specification
        "model_type": model_type,
        # fitted regression coefficients
        "intercept": model.intercept_,
        "coef": model.coef_,
        # training data
        "p": p,
        "d": d,
        # model configuration
        "n_x_features": n_x_features,
        # RidgeCV flag
        "ridge_cv": ridge_cv,
        # huber
        "huber": huber,
        "huber_epsilon": huber_epsilon,
        # log-quadratic tilt
        "tilt_penalty": tilt_penalty,
        "tilt_error": tilt_error,
    }

    return dhat_model, model_info


def vhat_piecewise_smoother(
    model_info,
    p,
    degree=2,
):
    """
    Compute a smoothed elasticity estimate for a piecewise demand model.

    Elasticity is first calculated at each piecewise region center.
    A polynomial is then fitted to the regional elasticity estimates
    and evaluated at p.

    Parameters
    ----------
    model_info : dict
        Dictionary returned by piecewise_demand_fit().

    p : float or array-like
        Price(s) at which elasticity is to be evaluated.

    degree : int, default=2
        Maximum polynomial degree used to smooth elasticity.

    Returns
    -------
    float or ndarray
        Smoothed elasticity evaluated at p.
    """

    scalar_input = np.isscalar(p)

    p = np.atleast_1d(
        np.asarray(
            p,
            dtype=float,
        )
    )

    region_centers = np.asarray(
        model_info["region_centers"],
        dtype=float,
    )

    models = model_info["models"]

    piecewise_model_type = model_info["piecewise_model_type"]

    # ---------- Compute elasticity at region centers ----------#

    vhat_regions = np.empty(
        len(region_centers),
        dtype=float,
    )

    for i, price in enumerate(region_centers):

        model = models[i]

        beta1 = model.coef_[0]

        if piecewise_model_type == "linear":

            d_hat = model.predict(np.array([[price]]))[0]

            d_hat = max(
                d_hat,
                1e-12,
            )

            vhat_regions[i] = -(price / d_hat) * beta1

        elif piecewise_model_type == "log_log":

            vhat_regions[i] = -beta1

        else:

            raise ValueError(
                "Unsupported piecewise model type: " f"{piecewise_model_type}"
            )

    # ---------- Select polynomial degree ----------#

    n_unique_centers = len(np.unique(region_centers))

    if n_unique_centers >= 4:
        degree_i = min(degree, 2)

    elif n_unique_centers >= 2:
        degree_i = 1

    else:
        degree_i = 0

    # ---------- Fit smooth elasticity curve ----------#

    coef = np.polyfit(
        region_centers,
        vhat_regions,
        deg=degree_i,
    )

    vhat_regions_smooth = np.polyval(
        coef,
        region_centers,
    )

    vhat_rmse = float(np.sqrt(np.mean((vhat_regions - vhat_regions_smooth) ** 2)))

    vhat = np.polyval(
        coef,
        p,
    )

    # ---------- Regional elasticity summary ----------#

    vhat_region_mean = float(np.mean(vhat_regions))

    vhat_region_median = float(np.median(vhat_regions))

    return {
        "vhat": float(vhat[0]) if scalar_input else vhat,
        "vhat_regions": vhat_regions,
        "vhat_regions_smooth": vhat_regions_smooth,
        "vhat_rmse": vhat_rmse,
        "vhat_region_mean": vhat_region_mean,
        "vhat_region_median": vhat_region_median,
        "coef": coef,
    }


# ----------------- vhat calculator raw for piecewise models ------------------#


def vhat_piecewise_raw(
    model_info,
    p,
):

    price = float(p)

    region_i = _piecewise_region_index(
        price,
        model_info["region_centers"],
    )

    model = model_info["models"][region_i]

    beta1 = model.coef_[0]

    if model_info["piecewise_model_type"] == "linear":

        d_hat = model.predict(np.array([[price]]))[0]

        d_hat = max(
            d_hat,
            1e-12,
        )

        return -(price / d_hat) * beta1

    elif model_info["piecewise_model_type"] == "log_log":

        return -beta1

    else:

        raise ValueError("Unsupported piecewise model type.")


# --------------- vhat calculator ------------------#
def vhat_calculator(model_info, p):
    """
    Compute the price elasticity implied by a fitted demand model.

    Supported demand models
    -----------------------

    linear:
        d = beta_0 + beta_1 * p

    log_level:
        log(d) = beta_0 + beta_1 * p

    log_log:
        log(d) = beta_0 + beta_1 * log(p)

    log_quadratic:
        log(d) = beta_0 + beta_1 * log(p)
                        + beta_2 * log(p)^2

    piecewise_linear:
        Piecewise linear demand model.
        Elasticity is computed from the local regional slope.

    piecewise_log_log:
        Piecewise log-log demand model.
        Elasticity is the constant elasticity of the local region.

    Parameters
    ----------
    model_info : dict
        Dictionary returned by fit_demand_estimator() or
        piecewise_demand_fit().

    p : float or array-like
        Price(s) at which elasticity is to be evaluated.

    Returns
    -------
    float or ndarray
        Estimated elasticity evaluated at p.
    """

    model_type = model_info["model_type"]

    scalar_input = np.isscalar(p)

    p = np.atleast_1d(
        np.asarray(
            p,
            dtype=float,
        )
    )

    if np.any(p <= 0):
        raise ValueError("Prices must be positive.")

    # ---------- Linear demand ----------#

    def _vhat_linear():

        beta1 = model_info["coef"][0]

        d_hat = model_info["dhat_model"](p)

        d_hat = np.maximum(
            d_hat,
            1e-12,
        )

        return -(p / d_hat) * beta1

    # ---------- Log-level demand ----------#

    def _vhat_log_level():

        beta1 = model_info["coef"][0]

        return -beta1 * p

    # ---------- Log-log demand ----------#

    def _vhat_log_log():

        beta1 = model_info["coef"][0]

        return np.full_like(
            p,
            -beta1,
            dtype=float,
        )

    # ---------- Log-quadratic demand ----------#

    def _vhat_log_quadratic():

        beta1 = model_info["coef"][0]
        beta2 = model_info["coef"][1]

        return -(beta1 + 2.0 * beta2 * np.log(p))

    # ---------- Piecewise linear demand ----------#

    def _vhat_piecewise_linear():

        diagnostics = vhat_piecewise_smoother(
            model_info=model_info,
            p=p,
            degree=2,
        )

        return np.atleast_1d(diagnostics["vhat"])

    ## ---------- Piecewise log-log demand ----------#

    def _vhat_piecewise_log_log():

        diagnostics = vhat_piecewise_smoother(
            model_info=model_info,
            p=p,
            degree=2,
        )

        return np.atleast_1d(diagnostics["vhat"])

    # ---------- Model dispatch ----------#

    vhat_functions = {
        "linear": _vhat_linear,
        "log_level": _vhat_log_level,
        "log_log": _vhat_log_log,
        "log_quadratic": _vhat_log_quadratic,
        "piecewise_linear": _vhat_piecewise_linear,
        "piecewise_log_log": _vhat_piecewise_log_log,
    }

    if model_type not in vhat_functions:
        raise ValueError(f"Unsupported model_type: {model_type}")

    vhat = vhat_functions[model_type]()

    if scalar_input:
        return float(vhat[0])

    return vhat


# ---------------- get_demand ------------------#


def get_demand(demand_type, demand_params, p, sigma_log=0):

    scalar_input = np.isscalar(p)

    if demand_type == "constant_elasticity":
        d = demand_constant_elasticity(
            p,
            demand_params["A"],
            demand_params["v"],
            sigma_log=sigma_log,
        )

    elif demand_type == "linear":
        d = demand_linear(
            p,
            demand_params["A"],
            demand_params["b"],
            sigma_log=sigma_log,
        )

    elif demand_type == "logit":
        d = demand_logit(
            p,
            demand_params["A"],
            demand_params["a"],
            demand_params["b"],
            sigma_log=sigma_log,
        )

    else:
        raise ValueError(f"Invalid demand type: {demand_type}")

    if scalar_input:
        return float(d[0])

    return d


# ----------------- piecewise regions ------------------#
def _piecewise_regions(
    N,
    N_regions=3,
    N_overlap=3,
):
    """
    Divide N ordered observations into overlapping regions.

    With few observations, regions may overlap completely.
    As N increases, the regions naturally separate.

    Parameters
    ----------
    N : int
        Number of ordered observations.

    N_regions : int, default=3
        Number of piecewise regions.

    N_overlap : int, default=3
        Number of observations added on each side of the
        base region boundaries.

    Returns
    -------
    list of ndarray
        Observation indices for each region.
    """

    if N < 2:
        raise ValueError("At least two observations are required.")

    if N_regions < 1:
        raise ValueError("N_regions must be at least 1.")

    N_overlap = max(
        0,
        int(N_overlap),
    )

    # ---------- Base region boundaries ----------#

    boundaries = np.linspace(
        0,
        N,
        N_regions + 1,
    )

    boundaries = np.round(boundaries).astype(int)

    # ---------- Construct overlapping regions ----------#

    overlap_left = N_overlap // 2
    overlap_right = N_overlap - overlap_left

    regions = []

    for i in range(N_regions):

        start = boundaries[i]
        end = boundaries[i + 1]

        if i > 0:
            start = max(
                0,
                start - overlap_left,
            )

        if i < N_regions - 1:
            end = min(
                N,
                end + overlap_right,
            )

        regions.append(np.arange(start, end))

    return regions


# ----------------- piecewise region index ------------------#
# For deciding which regional model applies to a new price,
# use the closest region center. That gives the simple hard switching we discussed without blending.


def _piecewise_region_index(
    p,
    region_centers,
):
    """
    Return the index of the region whose center is closest to p.
    """

    region_centers = np.asarray(
        region_centers,
        dtype=float,
    )

    return int(np.argmin(np.abs(region_centers - p)))


# ----------------- piecewise demand fit ------------------#
def piecewise_demand_fit(
    p,
    d,
    N_regions=3,
    N_overlap=5,
    model_type="linear",
    sample_weights=None,
):
    """
    Fit a piecewise demand model over overlapping price regions.

    Supported regional models
    -------------------------

    linear:
        d = beta_0 + beta_1 * p

    log_log:
        log(d) = beta_0 + beta_1 * log(p)

    Each price prediction uses the fitted model from the region
    whose center is closest to that price.

    Parameters
    ----------
    p : array-like
        Historical prices.

    d : array-like
        Historical demand observations.

    N_regions : int, default=3
        Number of piecewise price regions.

    N_overlap : int, default=3
        Number of observations shared around adjacent regions.

    model_type : {"linear", "log_log"}, default="linear"
        Regression model fitted independently within each region.

    sample_weights : array-like or None, default=None
        Optional regression sample weights.

    Returns
    -------
    dhat_model : callable
        Piecewise demand prediction function.

    model_info : dict
        Dictionary containing fitted regional models and
        supporting information.
    """

    p = np.asarray(
        p,
        dtype=float,
    )

    d = np.asarray(
        d,
        dtype=float,
    )

    if len(p) != len(d):
        raise ValueError("p and d must have the same length.")

    if len(p) < 2:
        raise ValueError("At least two observations are required.")

    if np.any(p <= 0):
        raise ValueError("Prices must be positive.")

    if model_type not in (
        "linear",
        "log_log",
    ):
        raise ValueError("model_type must be 'linear' or 'log_log'.")

    if model_type == "log_log" and np.any(d <= 0):
        raise ValueError("Demand must be positive for model_type='log_log'.")

    if sample_weights is not None:

        sample_weights = np.asarray(
            sample_weights,
            dtype=float,
        )

        if len(sample_weights) != len(p):
            raise ValueError("sample_weights must have the same length as p.")

    # ---------- Sort observations by price ----------#

    idx_sort = np.argsort(p)

    p_sorted = p[idx_sort]
    d_sorted = d[idx_sort]

    if sample_weights is not None:

        weights_sorted = sample_weights[idx_sort]

    else:

        weights_sorted = None

    # ---------- Determine price regions ----------#

    regions = _piecewise_regions(
        N=len(p_sorted),
        N_regions=N_regions,
        N_overlap=N_overlap,
    )

    # ---------- Fit regional models ----------#

    models = []
    region_centers = []

    for idx in regions:

        p_region = p_sorted[idx]
        d_region = d_sorted[idx]

        if len(p_region) < 2:
            raise ValueError("Each region must contain at least two observations.")

        if weights_sorted is not None:

            weights_region = weights_sorted[idx]

        else:

            weights_region = None

        # ---------- Construct regional regression ----------#

        if model_type == "linear":

            X_region = p_region.reshape(
                -1,
                1,
            )

            y_region = d_region

        else:

            X_region = np.log(p_region).reshape(
                -1,
                1,
            )

            y_region = np.log(d_region)

        model = LinearRegression()

        model.fit(
            X_region,
            y_region,
            sample_weight=weights_region,
        )

        models.append(model)

        region_centers.append(float(np.mean(p_region)))

    region_centers = np.asarray(
        region_centers,
        dtype=float,
    )

    # ---------- Demand prediction function ----------#

    def dhat_model(p_new):

        scalar_input = np.isscalar(p_new)

        p_new = np.atleast_1d(
            np.asarray(
                p_new,
                dtype=float,
            )
        )

        if np.any(p_new <= 0):
            raise ValueError("Prices must be positive.")

        d_hat = np.empty_like(
            p_new,
            dtype=float,
        )

        for i, price in enumerate(p_new):

            region_i = _piecewise_region_index(
                price,
                region_centers,
            )

            model = models[region_i]

            if model_type == "linear":

                prediction = model.predict(np.array([[price]]))[0]

                d_hat[i] = max(
                    prediction,
                    0.0,
                )

            else:

                prediction = model.predict(np.array([[np.log(price)]]))[0]

                d_hat[i] = np.exp(prediction)

        if scalar_input:
            return float(d_hat[0])

        return d_hat

    # ---------- Store model information ----------#

    model_info = {
        "model_type": f"piecewise_{model_type}",
        "piecewise_model_type": model_type,
        "piecewise_region_index": _piecewise_region_index,
        "dhat_model": dhat_model,
        "models": models,
        "regions": regions,
        "region_centers": region_centers,
        "N_regions": N_regions,
        "N_overlap": N_overlap,
        "p": p,
        "d": d,
        "p_sorted": p_sorted,
    }

    return dhat_model, model_info


# --------------------- select_demand_model ------------------#
def select_demand_model(
    p,
    d,
    candidate_model_types,
    price_min,
    price_max,
    c,
    F=0.0,
    sample_weights=None,
    ridge_cv=False,
    huber=False,
    huber_epsilon=1.35,
    minimum_elasticity=1.0,
    selection_criterion="aic",
    boundary_buffer=0.02,
    N_regions=10,  # piecewise models, number of regions to fit
    N_overlap=7,  # piecewise models, number of observations shared around adjacent regions
    Ngrid=500,
    tilt_penalty_linear=0.0,
    tilt_penalty_log_log=0.0,
    tilt_penalty_log_quadratic=0.0,
    vhat_rmse_max=0.5,
    aic_improvement_min=2.0,
    p_opt_difference_frac_max=0.10,
    log_quad_vhat_range_max=0.15,
    log_models_vhat_delta_max=0.10,
    p_opt_linear_delta_frac_max=0.20,
    logq_aic_improvement_min=2.0,
):
    """
    Select the best demand model from a set of candidate demand
    estimators.

    Each candidate model is fitted, the implied profit-maximizing
    price is computed, and the model is scored using the selected
    information criterion (AIC or BIC). Models whose estimated
    optimal price lies within a specified boundary buffer of the
    price limits are rejected whenever possible. The remaining
    model with the lowest information criterion is selected.

    Supported candidate models
    --------------------------

    linear:
        d = beta_0 + beta_1 * p

    log_level:
        log(d) = beta_0 + beta_1 * p

    log_log:
        log(d) = beta_0 + beta_1 * log(p)

    log_quadratic:
        log(d) = beta_0 + beta_1 * log(p)
                        + beta_2 * log(p)^2

    piecewise_linear:
        Piecewise linear regression over overlapping price regions.

    piecewise_log_log:
        Piecewise log-log regression over overlapping price regions.

    Parameters
    ----------
    p : array-like
        Historical prices.

    d : array-like
        Historical demand observations.

    price_min : float
        Minimum allowable price.

    price_max : float
        Maximum allowable price.

    c : float
        Variable unit cost.

    F : float
        Fixed cost.

    candidate_model_types : sequence of str
        Candidate demand model types to evaluate.

    selection_criteria : {"AIC", "BIC"}, default="AIC"
        Information criterion used for model selection.

    boundary_buffer : float, default=0.0
        Distance from the price boundaries within which a model
        is considered invalid if another valid candidate exists.

    sample_weights : array-like or None, default=None
        Optional regression sample weights.

    ridge_cv : bool, default=False
        If True, use RidgeCV instead of ordinary least squares
        for supported regression models.

    huber : bool, default=False
        If True, use Huber regression instead of ordinary least
        squares for supported regression models.

    Returns
    -------
    dhat_model : callable
        Selected demand prediction function.

    model_info : dict
        Dictionary describing the selected fitted demand model.
    """

    if huber and ridge_cv:
        raise ValueError("huber and ridge_cv cannot both be True.")

    # ---------- Validate selection criterion ----------#

    valid_selection_criteria = {
        "log_rmse",
        "aic",
        "aicc",
        "bic",
    }

    if selection_criterion not in valid_selection_criteria:
        raise ValueError(
            "selection_criterion must be one of " "{'log_rmse', 'aic', 'aicc', 'bic'}."
        )

    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)

    p_grid = np.linspace(
        price_min,
        price_max,
        Ngrid,
    )

    # ---------- Piecewise elasticity diagnostic ----------#

    _, piecewise_model_info = piecewise_demand_fit(
        p=p,
        d=d,
        N_regions=N_regions,
        N_overlap=N_overlap,
        model_type="linear",
        sample_weights=sample_weights,
    )

    piecewise_vhat_diagnostics = vhat_piecewise_smoother(
        model_info=piecewise_model_info,
        p=p,
        degree=2,
    )

    piecewise_vhat_rmse = piecewise_vhat_diagnostics["vhat_rmse"]

    vhat_region_mean = piecewise_vhat_diagnostics["vhat_region_mean"]

    vhat_region_median = piecewise_vhat_diagnostics["vhat_region_median"]

    if vhat_region_mean > 1.0:

        p_opt_region_mean = c * vhat_region_mean / (vhat_region_mean - 1.0)

    else:

        p_opt_region_mean = np.nan

    if vhat_region_median > 1.0:

        p_opt_region_median = c * vhat_region_median / (vhat_region_median - 1.0)

    else:

        p_opt_region_median = np.nan

    # ---------- Ensure prices are above unit cost ----------#

    p_grid = p_grid[p_grid > c]

    if len(p_grid) == 0:
        raise ValueError("price_max must be greater than unit cost c.")

    price_range = price_max - price_min
    boundary_distance = boundary_buffer * price_range

    fitted_models = {}
    model_diagnostics = {}

    # initialize some model diagnostics

    beta1 = np.nan
    beta2 = np.nan
    log_quad_vhat_range = np.nan
    log_quad_vhat_mid = np.nan
    log_log_vhat = np.nan
    log_models_vhat_delta = np.nan

    logq_vs_loglog_aic_improvement = np.nan

    # --------- Fit candidate models and compute diagnostics ---------#

    for model_type in candidate_model_types:

        try:

            # ---------- Fit candidate model ----------#

            if model_type in (
                "piecewise_linear",
                "piecewise_log_log",
            ):

                piecewise_model_type = (
                    "linear" if model_type == "piecewise_linear" else "log_log"
                )

                dhat_model_i, model_info_i = piecewise_demand_fit(
                    p=p,
                    d=d,
                    N_regions=N_regions,
                    N_overlap=N_overlap,
                    model_type=piecewise_model_type,
                    sample_weights=sample_weights,
                )

            else:

                dhat_model_i, model_info_i = fit_demand_estimator(
                    p=p,
                    d=d,
                    x=None,
                    ridge_cv=ridge_cv,
                    huber=huber,
                    huber_epsilon=huber_epsilon,
                    model_type=model_type,
                    sample_weights=sample_weights,
                    tilt_penalty_linear=tilt_penalty_linear,
                    tilt_penalty_log_log=tilt_penalty_log_log,
                    tilt_penalty_log_quadratic=tilt_penalty_log_quadratic,
                )

            # ---------- Log-demand residual diagnostics ----------#

            dhat = np.asarray(
                dhat_model_i(p),
                dtype=float,
            )

            predictions_valid = (
                np.all(np.isfinite(dhat))
                and np.all(dhat > 0)
                and np.all(np.isfinite(d))
                and np.all(d > 0)
            )

            # ---------- Number of fitted parameters ----------#

            if model_type in (
                "piecewise_linear",
                "piecewise_log_log",
            ):

                n_parameters = 2 * model_info_i["N_regions"]

            else:

                n_parameters = len(model_info_i["coef"]) + 1

            if not predictions_valid:

                fit_error = np.inf
                rss = np.inf
                aic = np.inf
                aicc = np.inf
                bic = np.inf

            else:

                residual = np.log(d) - np.log(dhat)

                if sample_weights is None:

                    fit_error = np.sqrt(np.mean(residual**2))

                    rss = np.sum(residual**2)

                else:

                    weights = np.asarray(
                        sample_weights,
                        dtype=float,
                    )

                    fit_error = np.sqrt(
                        np.average(
                            residual**2,
                            weights=weights,
                        )
                    )

                    rss = np.sum(weights * residual**2)

                rss = max(
                    float(rss),
                    1e-12,
                )

                n_observations = len(residual)

                # ---------- Information criteria ----------#

                aic = n_observations * np.log(rss / n_observations) + 2.0 * n_parameters

                if n_observations > n_parameters + 1:

                    aicc = aic + (2.0 * n_parameters * (n_parameters + 1)) / (
                        n_observations - n_parameters - 1
                    )

                else:

                    aicc = np.inf

                bic = n_observations * np.log(
                    rss / n_observations
                ) + n_parameters * np.log(n_observations)

            # ---------- Set model-selection score ----------#

            if selection_criterion == "log_rmse":
                model_fit_score = fit_error

            elif selection_criterion == "aic":
                model_fit_score = aic

            elif selection_criterion == "aicc":
                model_fit_score = aicc

            else:
                model_fit_score = bic

            # ---------- Estimate profit-maximizing price ----------#

            d_grid_hat = np.asarray(
                dhat_model_i(p_grid),
                dtype=float,
            )

            profit_grid_hat = (p_grid - c) * d_grid_hat - F

            valid_grid = (
                np.isfinite(d_grid_hat)
                & np.isfinite(profit_grid_hat)
                & (d_grid_hat > 0)
            )

            if not np.any(valid_grid):
                raise ValueError("No valid demand predictions over price grid.")

            profit_grid_valid = np.where(
                valid_grid,
                profit_grid_hat,
                -np.inf,
            )

            i_opt = int(np.argmax(profit_grid_valid))

            p_opt_hat = float(p_grid[i_opt])

            d_opt_hat = float(d_grid_hat[i_opt])

            profit_opt_hat = float(profit_grid_hat[i_opt])

            # ---------- Elasticity optimality diagnostic ----------#

            vhat_opt = float(
                np.asarray(
                    vhat_calculator(
                        model_info=model_info_i,
                        p=p_opt_hat,
                    )
                ).reshape(-1)[0]
            )

            # ---------- Piecewise-smoothed elasticity at candidate optimum ----------#

            piecewise_vhat_at_popt = float(
                np.asarray(
                    vhat_piecewise_smoother(
                        model_info=piecewise_model_info,
                        p=p_opt_hat,
                        degree=2,
                    )["vhat"]
                ).reshape(-1)[0]
            )

            # vhat_delta = abs(vhat_opt - piecewise_vhat_at_popt)

            inv_umar = p_opt_hat / (p_opt_hat - c)

            optimality_gap = abs(vhat_opt - inv_umar)

            # ---------- Elasticity agreement diagnostic ----------#

            if model_type in (
                "piecewise_linear",
                "piecewise_log_log",
            ):

                vhat_compare = vhat_piecewise_raw(
                    model_info=model_info_i,
                    p=p_opt_hat,
                )

            else:

                vhat_compare = vhat_opt

            vhat_delta = abs(vhat_compare - piecewise_vhat_at_popt)

            # ---------- Boundary diagnostic ----------#

            at_boundary = (
                p_opt_hat <= price_min + boundary_distance
                or p_opt_hat >= price_max - boundary_distance
            )

            # ---------- Minimum elasticity check ----------#

            elasticity_valid = np.isfinite(vhat_opt) and vhat_opt >= minimum_elasticity

            # ---------- Model-selection score ----------#

            if np.isfinite(model_fit_score) and elasticity_valid:
                selection_score = model_fit_score

            else:
                selection_score = np.inf

            # ---------- RidgeCV-selected alpha ----------#
            # piecewise models return models not model ...

            if "model" in model_info_i and hasattr(model_info_i["model"], "alpha_"):
                selected_alpha = float(model_info_i["model"].alpha_)
            else:
                selected_alpha = 0.0

            fitted_models[model_type] = (
                dhat_model_i,
                model_info_i,
            )

            # ---------- Log-model structural diagnostics ----------#

            beta1 = np.nan
            beta2 = np.nan
            log_quad_vhat_range = np.nan

            if model_type == "log_quadratic":

                beta1 = float(model_info_i["coef"][0])

                beta2 = float(model_info_i["coef"][1])

                # Total change in fitted elasticity across price range
                log_quad_vhat_range = (
                    2.0 * abs(beta2) * abs(np.log(price_max / price_min))
                )

                p_mid = np.sqrt(price_min * price_max)

                log_quad_vhat_mid = -(beta1 + 2.0 * beta2 * np.log(p_mid))

            elif model_type == "log_log":

                beta1_log_log = float(model_info_i["coef"][0])

                log_log_vhat = -beta1_log_log
            ###

            model_diagnostics[model_type] = {
                "fit_error": float(fit_error),
                "rss": float(rss),
                "n_parameters": int(n_parameters),
                "aic": float(aic),
                "aicc": float(aicc),
                "bic": float(bic),
                "selection_criterion": selection_criterion,
                "selection_score": float(selection_score),
                "optimality_gap": float(optimality_gap),
                "p_opt_hat": float(p_opt_hat),
                "d_opt_hat": float(d_opt_hat),
                "profit_opt_hat": float(profit_opt_hat),
                "vhat_opt": float(vhat_opt),
                "vhat_compare": float(vhat_compare),
                "piecewise_vhat_at_popt": float(piecewise_vhat_at_popt),
                "vhat_delta": float(vhat_delta),
                "inv_umar": float(inv_umar),
                "at_boundary": bool(at_boundary),
                "elasticity_valid": bool(elasticity_valid),
                "piecewise_vhat_rmse": float(piecewise_vhat_rmse),
                "vhat_region_mean": float(vhat_region_mean),
                "vhat_region_median": float(vhat_region_median),
                "p_opt_region_mean": float(p_opt_region_mean),
                "p_opt_region_median": float(p_opt_region_median),
                "selected_alpha": float(selected_alpha),
                "tilt_penalty": float(model_info_i.get("tilt_penalty", 0.0)),
                "tilt_error": float(model_info_i.get("tilt_error", np.nan)),
                "beta1": beta1,
                "beta2": beta2,
                "log_quad_vhat_range": log_quad_vhat_range,
                "error_message": None,
            }

        except Exception as exc:  # pylint: disable=broad-exception-caught

            model_diagnostics[model_type] = {
                "fit_error": np.inf,
                "rss": np.inf,
                "n_parameters": 0,
                "aic": np.inf,
                "aicc": np.inf,
                "bic": np.inf,
                "selection_criterion": selection_criterion,
                "selection_score": np.inf,
                "optimality_gap": np.inf,
                "p_opt_hat": np.nan,
                "d_opt_hat": np.nan,
                "profit_opt_hat": np.nan,
                "vhat_opt": np.nan,
                "inv_umar": np.nan,
                "at_boundary": False,
                "elasticity_valid": False,
                "piecewise_vhat_rmse": float(piecewise_vhat_rmse),
                "selected_alpha": np.nan,
                "error_message": str(exc),
                "vhat_compare": np.nan,
                "piecewise_vhat_at_popt": np.nan,
                "vhat_delta": np.nan,
                "vhat_region_mean": float(vhat_region_mean),
                "vhat_region_median": float(vhat_region_median),
                "p_opt_region_mean": float(p_opt_region_mean),
                "p_opt_region_median": float(p_opt_region_median),
                "tilt_penalty": np.nan,
                "tilt_error": np.nan,
                "beta1": np.nan,
                "beta2": np.nan,
                "log_quad_vhat_range": np.nan,
            }

    m_log_log = np.nan
    m_log_quadratic = np.nan
    markup_pct_delta = np.nan

    if "log_log" in model_diagnostics and "log_quadratic" in model_diagnostics:
        # eps ... epsilon ... elasticity at the estimated optimum price for each model
        eps_log_log = model_diagnostics["log_log"]["vhat_opt"]

        eps_log_quadratic = model_diagnostics["log_quadratic"]["vhat_opt"]

        if eps_log_log > 1.0:

            m_log_log = eps_log_log / (eps_log_log - 1.0)

        if eps_log_quadratic > 1.0:

            m_log_quadratic = eps_log_quadratic / (eps_log_quadratic - 1.0)

        if np.isfinite(m_log_log) and np.isfinite(m_log_quadratic):

            markup_pct_delta = 100.0 * abs(m_log_quadratic - m_log_log) / m_log_log

    # ---- model diagnostics ... after loop----#

    if np.isfinite(log_quad_vhat_mid) and np.isfinite(log_log_vhat):

        log_models_vhat_delta = abs(log_quad_vhat_mid - log_log_vhat)

    else:

        log_models_vhat_delta = np.nan

    if "log_quadratic" in model_diagnostics:

        model_diagnostics["log_quadratic"]["log_quad_vhat_mid"] = float(
            log_quad_vhat_mid
        )

        model_diagnostics["log_quadratic"]["log_models_vhat_delta"] = float(
            log_models_vhat_delta
        )

    if "log_log" in model_diagnostics:

        model_diagnostics["log_log"]["log_log_vhat"] = float(log_log_vhat)

        model_diagnostics["log_log"]["log_models_vhat_delta"] = float(
            log_models_vhat_delta
        )

    # log-log vs log-quadratic AIC improvement

    logq_vs_loglog_aic_improvement = np.nan

    if "log_log" in model_diagnostics and "log_quadratic" in model_diagnostics:

        logq_vs_loglog_aic_improvement = (
            model_diagnostics["log_log"]["aic"]
            - model_diagnostics["log_quadratic"]["aic"]
        )

    #  log model diagnostics so they are available for reporting and plotting
    for model_type in ("log_log", "log_quadratic"):

        if model_type in model_diagnostics:

            model_diagnostics[model_type].update(
                {
                    "log_quad_vhat_mid": float(log_quad_vhat_mid),
                    "log_log_vhat": float(log_log_vhat),
                    "log_models_vhat_delta": float(log_models_vhat_delta),
                    "m_log_log": float(m_log_log),
                    "m_log_quadratic": float(m_log_quadratic),
                    "markup_pct_delta": float(markup_pct_delta),
                    "logq_vs_loglog_aic_improvement": float(
                        logq_vs_loglog_aic_improvement
                    ),
                }
            )

    # ---------- Identify valid candidate models ----------#

    valid_models = [
        model_type
        for model_type, diagnostics in model_diagnostics.items()
        if (
            diagnostics["elasticity_valid"]
            and np.isfinite(diagnostics["selection_score"])
        )
    ]

    if len(valid_models) == 0:

        error_messages = {
            model_type: diagnostics["error_message"]
            for model_type, diagnostics in model_diagnostics.items()
        }

        raise ValueError(
            "None of the candidate models could be selected. "
            f"Errors: {error_messages}"
        )

    # ---------- Prefer models with interior optima ----------#

    interior_models = [
        model_type
        for model_type in valid_models
        if not model_diagnostics[model_type]["at_boundary"]
    ]

    candidate_models = interior_models if len(interior_models) > 0 else valid_models

    # ---------- Select model ----------#
    # assume 3 models present - linear, log-log, log-quadratic

    #  1. Select best-fitting model by AIC

    selected_model_type = min(
        ("linear", "log_log", "log_quadratic"),
        key=lambda model_type: model_diagnostics[model_type]["aic"],
    )

    #  2. Check whether log-quadratic complexity
    #    is justified relative to log-log

    logq_aic_improvement = (
        model_diagnostics["log_log"]["aic"] - model_diagnostics["log_quadratic"]["aic"]
    )

    if (
        selected_model_type == "log_quadratic"
        and logq_aic_improvement < logq_aic_improvement_min
    ):
        selected_model_type = "log_log"

    #  3. Check log-model divergence from linear #

    p_opt_linear_delta_frac = np.nan

    if selected_model_type != "linear":

        p_opt_log = model_diagnostics[selected_model_type]["p_opt_hat"]

        p_opt_linear = model_diagnostics["linear"]["p_opt_hat"]

        p_opt_linear_delta_frac = abs(p_opt_log - p_opt_linear) / (
            price_max - price_min
        )

        # Fall back to linear if the chosen log model
        # diverges too far from the conservative linear model.
        if p_opt_linear_delta_frac > p_opt_linear_delta_frac_max:
            selected_model_type = "linear"

    # selection diagnostics
    for model_type in (
        "linear",
        "log_log",
        "log_quadratic",
    ):

        model_diagnostics[model_type]["logq_aic_improvement"] = float(
            logq_aic_improvement
        )

        model_diagnostics[model_type]["p_opt_linear_delta_frac"] = float(
            p_opt_linear_delta_frac
        )

    dhat_model, model_info = fitted_models[selected_model_type]

    model_info["selection_diagnostics"] = model_diagnostics[selected_model_type]

    return (
        dhat_model,
        model_info,
        selected_model_type,
        model_diagnostics,
    )


# --------------------- aggregate_price_demand ------------------#


def aggregate_price_demand(
    p,
    d,
    price_decimals=10,
):
    """
    Aggregate repeated prices for a log-demand regression.

    Demand is aggregated using its geometric mean, and the
    number of observations at each price is returned for use
    as regression sample weights.

    Parameters
    ----------
    p : array-like
        Historical prices.

    d : array-like
        Positive historical demand observations.

    price_decimals : int, default=10
        Number of decimal places used to identify repeated prices.

    Returns
    -------
    p_unique : ndarray
        Sorted unique prices.

    d_geo_mean : ndarray
        Geometric mean demand at each unique price.

    counts : ndarray
        Number of observations at each unique price.
    """

    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)

    if len(p) != len(d):
        raise ValueError("p and d must have the same length.")

    if np.any(p <= 0):
        raise ValueError("Prices must be positive.")

    if np.any(d <= 0):
        raise ValueError("Demand must be positive.")

    p_group = np.round(p, decimals=price_decimals)

    p_unique = np.unique(p_group)

    d_geo_mean = np.empty(len(p_unique))
    counts = np.empty(len(p_unique), dtype=int)

    for j, price in enumerate(p_unique):
        mask = p_group == price

        d_geo_mean[j] = np.exp(np.mean(np.log(d[mask])))

        counts[j] = np.sum(mask)

    return p_unique, d_geo_mean, counts


# ------------------ mab_price_optimization_loop -----------------#
def mab_price_optimization_loop(
    price_min: float,
    price_max: float,
    c: float,
    F: float,
    p_opt: float,
    demand_type: str,
    demand_params: dict,
    sigma_log: float,
    N_loop=20,
    N_price_repeat=1,
    ridge_cv=False,
    huber=False,
    huber_epsilon=1.35,
    candidate_model_types=None,
    N_explore=12,
    N_initial_prices=10,
    price_delta=0.05,  #
    boundary_buffer=0.02,
    selection_criterion="aic",
    tilt_penalty_linear=0.0,
    tilt_penalty_log_log=0.0,
    tilt_penalty_log_quadratic=0.0,
    logq_aic_improvement_min=2.0,
    p_opt_linear_delta_frac_max=0.20,
    N_regions=3,  # for piecewise models, number of regions to fit
    N_overlap=5,  # for piecwise models, number of observations shared around adjacent regions
    verbose=2,
):
    """Multi-armed bandit (MAB) price optimization loop.

        This function simulates a price optimization process using a multi-armed bandit approach.
        It iteratively selects prices, observes simulated demand, and updates the estimated profit-maximizing price.
        The function keeps track of price and demand histories, fits demand models, and selects the optimal price based on the estimated demand models.

        The loop consists of an initial exploration phase followed by an exploitation phase where the estimated optimal price is used.

        The function returns a dictionary containing the selected demand model information, including the fitted model, model diagnostics, and the selected model type.


    Args:
        price_min (float): Minimum allowable price.
        price_max (float): Maximum allowable price.
        c (float): Unit variable cost.
        F (float): Fixed cost.
        p_opt (float): True or theoretical optimal price used for evaluation.
        demand_type (str): Demand model type. Supported values include "linear",
            "constant_elasticity", and "logit".
        demand_params (dict): Parameters for the selected demand model.
        sigma_log (float): Standard deviation of the log-normal noise added to the
            demand simulation.
        N_loop (int, optional): Number of iterations of the MAB price optimization loop.
            Defaults to 20.
        N_price_repeat (int, optional): Number of times each price is repeated during the
            initial exploration phase. Defaults to 1.
        ridge_cv (bool, optional): If True, fit candidate models with RidgeCV.
            Defaults to False.
        candidate_model_types (list[str] or None, optional): Candidate demand model types
            to evaluate. If None, defaults to ["linear", "log_level", "log_log",
            "log_quadratic"].
        N_explore (int, optional): Number of iterations to keep exploring around the
            estimated optimum. Defaults to 12.
        N_initial_prices (int, optional): Number of distinct prices used during initial
            exploration. Defaults to 10.
        price_delta (float, optional): Step size used when exploring around the current
            estimated optimal price. Defaults to 0.05.
        verbose (int, optional): Verbosity level. Higher values print more information.
            Defaults to 2.

    Returns:
        dict: Dictionary containing the simulation outcomes, including:
            - p_history
            - d_history
            - selected_model
            - selected_model_type
            - dhat_model
            - model_info
            - model_fit_errors
            - p_opt_hat_history
            - p_opt_vhat_history
            - p_opt_hat_error_history
            - d_opt_hat_history
            - d_sigma0_history
            - vhat_opt_history
            - inv_umar_history
            - profit_opt_history
    """

    if huber and ridge_cv:
        raise ValueError("huber and ridge_cv cannot both be True.")

    # ---------- Aserctions -----------------#

    assert (
        N_loop > N_price_repeat * N_initial_prices
    ), f" Nloop={N_loop} must be >= Nprice_repeat * N_initial_prices = {N_price_repeat * N_initial_prices}"

    # ---------- candidate_model_types -----------------#

    if candidate_model_types is None:

        candidate_model_types = [
            "linear",
            "log_level",
            "log_log",
            "log_quadratic",
            "piecewise_linear",
            "piecewise_log_log",
        ]

    # --------- Price Exploration Initialization --------

    prices_initial = np.linspace(
        start=price_min, stop=price_max, num=N_initial_prices
    ).tolist()

    prices_initial = [
        round(p, 2) for p in prices_initial for _ in range(N_price_repeat)
    ]

    N_explore = max(
        N_explore, len(prices_initial)
    )  # explore up to N_explore, then exploit p_opt_hat

    # --------- Price Grid used to maximize estimated profit ----------#

    p_grid = np.maximum(
        np.linspace(price_min, price_max, 500),
        c + 0.02,
    )

    # ---- min fit observations log_quadratic requires at least three unique prices ---- #

    min_fit_obs = 3
    # min_fit_obs = max(min_fit_obs, len(prices_initial))

    # ------------Histories -------------------------#

    p_history = []
    d_history = []

    selected_model_history = []
    model_fit_error_history = []

    dhat_model = None
    model_info = None
    selected_model_type = None

    p_opt_hat_history = []
    p_opt_vhat_history = []
    d_opt_hat_history = []
    vhat_opt_history = []
    inv_umar_history = []
    profit_opt_history = []
    p_opt_hat_error_history = []
    d_sigma0_history = []
    vhat_rmse_history = []

    piecewise_vhat_at_popt_history = []
    vhat_delta_history = []

    m_log_log_history = []

    m_log_quadratic_history = []

    markup_pct_delta_history = []

    model_diagnostics_history = []

    # ---- Initial values used by the price-control logic ------#

    pos_neg = 1
    p_opt_hat = np.nan
    vhat_opt_i = np.nan
    inv_umar_i = np.nan

    # ======== MAB loop ===============================#

    repeat_count = 0
    # prev_pi = prices_initial[0]
    for i in range(N_loop):

        if verbose > 0:
            print(f"i = {i}", end=", ")

        # ----price selection ---------#

        repeat_count += 1
        # print("repeat_count =", repeat_count)

        if i < len(prices_initial):
            # Initial structured exploration
            pi = prices_initial[i]

        # done with initial exploration, now exploit the estimated profit-maximizing price
        elif repeat_count == N_price_repeat or i == len(prices_initial):

            if i <= N_explore:

                # Continue exploring around the currently estimated
                # profit-maximizing price.
                if vhat_opt_i < inv_umar_i:
                    # Demand is not elastic enough relative to
                    # inverse unit margin, so increase price.
                    pos_neg = 1

                elif vhat_opt_i > inv_umar_i:
                    # Demand is too elastic relative to
                    # inverse unit margin, so decrease price.
                    pos_neg = -1

                else:
                    # Reverse direction if they are equal.
                    pos_neg = -pos_neg

                pi = np.clip(
                    round(
                        p_opt_hat + pos_neg * price_delta,
                        2,
                    ),
                    price_min,
                    price_max,
                )

                pi = float(pi)  # make sure price is a float

                # print("Explore price, i = ", i, ", pi = ", pi)

            else:
                # if into explore rnage then exploit
                # Exploit the estimated profit-maximizing price
                pi = p_opt_hat

        # repeat price
        else:
            pi = prev_pi

        if repeat_count >= N_price_repeat:
            repeat_count = 0

        # -------- Simulated observed demand ------------------#

        di = round(
            get_demand(
                demand_type,
                demand_params,
                pi,
                sigma_log=sigma_log,
            ),
            1,
        )

        # Noiseless demand is used only as a simulation diagnostic.
        # It is not used for model fitting or model selection.
        di_sigma0_i = round(
            get_demand(
                demand_type,
                demand_params,
                pi,
                sigma_log=0,
            ),
            1,
        )
        if verbose > 1:
            print(
                f"pi = {pi}, " f"di = {di}, " f"di_sigma0_i = {di_sigma0_i}",
                end=", ",
            )

        d_sigma0_history.append(di_sigma0_i)

        # -----------Update price and demand history ------------------#

        p_history.append(pi)
        d_history.append(di)

        # ------ Aggregate repeated prices ---------------------------#

        p_fit, d_fit, counts = aggregate_price_demand(
            p=p_history,
            d=d_history,
        )
        p_fit = np.asarray(
            p_fit,
            dtype=float,
        )
        d_fit = np.asarray(
            d_fit,
            dtype=float,
        )
        counts = np.asarray(
            counts,
            dtype=float,
        )

        # Repeated prices receive more weight, but the square root
        # prevents high-repeat prices from dominating completely.
        sample_weights = np.sqrt(counts)

        # ---- Wait until enough unique prices are available -------#

        if len(p_fit) < min_fit_obs:

            if verbose > 1:
                print("selected_model = none, p_opt_hat = nan")

            selected_model_history.append(None)
            model_fit_error_history.append(None)
            p_opt_hat_history.append(np.nan)
            p_opt_vhat_history.append(np.nan)
            d_opt_hat_history.append(np.nan)
            vhat_opt_history.append(np.nan)
            inv_umar_history.append(np.nan)
            profit_opt_history.append(np.nan)
            p_opt_hat_error_history.append(np.nan)

            continue

        # ------ Fit and select the demand model -------------

        try:
            dhat_model, model_info, selected_model_type, model_diagnostics = (
                select_demand_model(
                    p=p_fit,
                    d=d_fit,
                    candidate_model_types=candidate_model_types,
                    price_min=price_min,
                    price_max=price_max,
                    c=c,
                    F=F,
                    sample_weights=sample_weights,
                    boundary_buffer=boundary_buffer,
                    selection_criterion=selection_criterion,
                    ridge_cv=ridge_cv,
                    huber=huber,
                    huber_epsilon=huber_epsilon,
                    tilt_penalty_linear=tilt_penalty_linear,
                    tilt_penalty_log_log=tilt_penalty_log_log,
                    tilt_penalty_log_quadratic=tilt_penalty_log_quadratic,
                    N_regions=N_regions,
                    N_overlap=N_overlap,
                    logq_aic_improvement_min=logq_aic_improvement_min,
                    p_opt_linear_delta_frac_max=p_opt_linear_delta_frac_max,
                )
            )

        except ValueError as exc:

            if verbose > 1:
                print(f"selected_model = none, " f"model selection failed: {exc}")

            selected_model_history.append(None)
            model_fit_error_history.append(None)
            p_opt_hat_history.append(np.nan)
            p_opt_vhat_history.append(np.nan)
            d_opt_hat_history.append(np.nan)
            vhat_opt_history.append(np.nan)
            inv_umar_history.append(np.nan)
            profit_opt_history.append(np.nan)
            p_opt_hat_error_history.append(np.nan)

            continue

        _selected_model_error = model_diagnostics[selected_model_type]["fit_error"]

        piecewise_vhat_rmse = model_diagnostics[selected_model_type][
            "piecewise_vhat_rmse"
        ]

        piecewise_vhat_at_popt = model_diagnostics[selected_model_type][
            "piecewise_vhat_at_popt"
        ]

        vhat_delta = model_diagnostics[selected_model_type]["vhat_delta"]

        m_log_log = model_diagnostics[selected_model_type].get(
            "m_log_log",
            np.nan,
        )

        m_log_quadratic = model_diagnostics[selected_model_type].get(
            "m_log_quadratic",
            np.nan,
        )

        markup_pct_delta = model_diagnostics[selected_model_type].get(
            "markup_pct_delta",
            np.nan,
        )

        # ---Estimate demand and profit over the price grid ---------------------#

        d_grid_hat = np.asarray(
            dhat_model(p_grid),
            dtype=float,
        )

        profit_grid_hat = (p_grid - c) * d_grid_hat - F

        # Ignore any invalid grid predictions
        valid_grid = (
            np.isfinite(d_grid_hat) & np.isfinite(profit_grid_hat) & (d_grid_hat >= 0)
        )

        if not np.any(valid_grid):
            print(
                f"selected_model = {selected_model_type}, "
                "no valid profit-grid predictions"
            )
            print(f"d_grid_hat = {d_grid_hat}")

            selected_model_history.append(selected_model_type)

            model_fit_error_history.append(
                model_diagnostics[selected_model_type]["fit_error"]
            )

            p_opt_hat_history.append(np.nan)
            p_opt_vhat_history.append(np.nan)
            d_opt_hat_history.append(np.nan)
            vhat_opt_history.append(np.nan)
            inv_umar_history.append(np.nan)
            profit_opt_history.append(np.nan)
            p_opt_hat_error_history.append(np.nan)

            continue

        profit_grid_valid = np.where(
            valid_grid,
            profit_grid_hat,
            -np.inf,
        )

        i_opt = int(np.argmax(profit_grid_valid))

        # ------- Estimated optimum ----------------#

        p_opt_hat = round(
            float(p_grid[i_opt]),
            2,
        )
        d_opt_hat = round(
            float(d_grid_hat[i_opt]),
            1,
        )
        profit_opt_hat = round(
            float(profit_grid_hat[i_opt]),
            2,
        )

        # ---- Elasticity at the estimated optimum  -----#

        vhat_opt_i = round(
            float(
                np.asarray(
                    vhat_calculator(
                        model_info=model_info,
                        p=p_opt_hat,
                    )
                ).reshape(-1)[0]
            ),
            2,
        )

        inv_umar_i = round(
            p_opt_hat / (p_opt_hat - c),
            2,
        )  # inverse unit margine

        # price = c * vhat / (vhat - 1) ...  constant elasticity assumption
        if not np.isfinite(vhat_opt_i) or np.isclose(vhat_opt_i, 1.0):
            p_opt_vhat_i = np.nan
        else:
            p_opt_vhat_i = round(
                c * vhat_opt_i / (vhat_opt_i - 1.0),
                2,
            )

        # ------- price  vs true price error -----------------#
        # p_opt is the known true optimum in the simulation.
        # It is not used by model fitting or model selection.

        p_opt_hat_error = round(
            abs(p_opt_hat - p_opt),
            2,
        )

        # histories

        selected_model_history.append(selected_model_type)
        model_fit_error_history.append(
            model_diagnostics[selected_model_type]["fit_error"]
        )

        vhat_rmse_history.append(piecewise_vhat_rmse)
        p_opt_hat_history.append(p_opt_hat)
        p_opt_vhat_history.append(p_opt_vhat_i)
        d_opt_hat_history.append(d_opt_hat)
        vhat_opt_history.append(vhat_opt_i)
        inv_umar_history.append(inv_umar_i)
        profit_opt_history.append(profit_opt_hat)
        p_opt_hat_error_history.append(p_opt_hat_error)

        piecewise_vhat_at_popt_history.append(piecewise_vhat_at_popt)

        m_log_log_history.append(m_log_log)

        m_log_quadratic_history.append(m_log_quadratic)

        markup_pct_delta_history.append(markup_pct_delta)

        model_diagnostics_history.append(model_diagnostics)

        vhat_delta_history.append(vhat_delta)

        # ------------------ loop administration -------------#
        prev_pi = pi

        #  Print iteration results
        if verbose > 1 and i < N_loop - 1:
            print(
                f"selected_model = {selected_model_type}",
                f", model_fit_error = {round(model_diagnostics[selected_model_type]["fit_error"],2)}",
                f", p_opt_hat = {p_opt_hat}",
                f", p_opt_vhat_i = {p_opt_vhat_i}",
                f", p_opt_hat_error = {p_opt_hat_error}",
                f", vhat_opt_i = {vhat_opt_i}",
                f", inv_umar_i = {inv_umar_i}",
                f", d_opt_hat = {d_opt_hat}",
                f", d_sigma0_i = {di_sigma0_i}",
                f", profit_opt_hat = {profit_opt_hat}",
            )

    # afer loop results
    if verbose > 0:
        print(
            f"selected_model = {selected_model_type}",
            f", p_opt_hat = {p_opt_hat}",
            f", p_opt_vhat_i = {p_opt_vhat_i}",
            f", p_opt_hat_error = {p_opt_hat_error}",
            f", vhat_opt_i = {vhat_opt_i}",
            f", inv_umar_i = {inv_umar_i}",
            f", d_opt_hat = {d_opt_hat}",
            f", d_sigma0_i = {di_sigma0_i}",
            f", profit_opt_hat = {profit_opt_hat}",
        )

    results_dict = {
        "p_history": p_history,
        "d_history": d_history,
        "selected_model": selected_model_history,
        "vhat_rmse_history": vhat_rmse_history,
        "selected_model_type": selected_model_type,
        "dhat_model": dhat_model,
        "model_info": model_info,
        "model_fit_errors": model_fit_error_history,
        "p_opt_hat_history": p_opt_hat_history,
        "p_opt_vhat_history": p_opt_vhat_history,
        "p_opt_hat_error_history": p_opt_hat_error_history,
        "d_opt_hat_history": d_opt_hat_history,
        "d_sigma0_history": d_sigma0_history,
        "vhat_opt_history": vhat_opt_history,
        "piecewise_vhat_at_popt_history": piecewise_vhat_at_popt_history,
        "vhat_delta_history": vhat_delta_history,
        "inv_umar_history": inv_umar_history,
        "profit_opt_history": profit_opt_history,
        "m_log_log_history": m_log_log_history,
        "m_log_quadratic_history": m_log_quadratic_history,
        "markup_pct_delta_history": markup_pct_delta_history,
        "model_diagnostics_history": model_diagnostics_history,
    }

    return results_dict
