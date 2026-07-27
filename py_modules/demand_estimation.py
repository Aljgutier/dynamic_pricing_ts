"""
Demand Estimation Module

- fit_demand_estimator
- vhat_calculator
- select_demand_model
- get_demand
- mab_price_optimization_loop
"""

from sklearn.linear_model import LinearRegression, RidgeCV
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


# ---------------  demand estimator -----------------#
def fit_demand_estimator(
    p,
    d,
    x=None,
    model_type="log_quadratic",
    sample_weights=None,
    ridge_cv=False,
):
    """
    Fit a parametric demand model.

    Supported demand models
    -----------------------

    linear:   d = beta_0 + beta_1 * p    + gamma' x
    log_level:  log(d) = beta_0  + beta_1 * p + gamma' x
    log_log: log(d) = beta_0 + beta_1 * log(p) + gamma' x
    log_quadratic:  log(d) = beta_0   + beta_1 * log(p)  + beta_2 * log(p)^2 + gamma' x

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

    # ----------------------------------------------------------
    # Helper: construct regression response
    # ----------------------------------------------------------

    def _response(d_values):
        """
        Construct the regression response variable.
        """

        if model_type == "linear":
            return d_values

        return np.log(d_values)

    # ----------------------------------------------------------
    # Helper: construct price basis functions
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Helper: transform regression output back to demand
    # ----------------------------------------------------------

    def _inverse_response(y):
        """
        Transform regression predictions back to demand.
        """

        if model_type == "linear":
            # demand cannot be negative
            return np.maximum(y, 0.0)

        return np.exp(y)

    # ----------------------------------------------------------
    # Construct regression response and price basis
    # ----------------------------------------------------------

    target = _response(d)

    price_basis = _price_basis(p)

    # ----------------------------------------------------------
    # Optional contextual variables
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Fit regression model
    # ----------------------------------------------------------

    if ridge_cv == True:
        model = RidgeCV(alphas=np.logspace(-4, 2, 25))
    else:
        model = LinearRegression()

    model.fit(
        features,
        target,
        sample_weight=sample_weights,
    )

    # ----------------------------------------------------------
    # Demand prediction function
    # ----------------------------------------------------------

    def dhat_model(p_new, x_new=None):

        scalar_input = np.isscalar(p_new)
        p_new = np.atleast_1d(np.asarray(p_new, dtype=float))

        price_basis_new = _price_basis(p_new)

        # ------------------------------------------------------
        # Optional contextual feature variables
        # ------------------------------------------------------

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
                raise ValueError("x_new must have the same number of rows " "as p_new.")

            features_new = np.column_stack(
                (
                    price_basis_new,
                    x_new,
                )
            )

        else:

            features_new = price_basis_new

        # ------------------------------------------------------
        # Predict demand
        # ------------------------------------------------------

        prediction = model.predict(features_new)

        d_hat = _inverse_response(prediction)

        if scalar_input:
            return float(d_hat[0])

        return d_hat

    # ----------------------------------------------------------
    # Store model information
    # ----------------------------------------------------------

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
    }

    return dhat_model, model_info


# --------------- vhat calculator ------------------#
def vhat_calculator(model_info, p):
    """
    Compute the price elasticity implied by a fitted demand model.

    Parameters
    ----------
    model_info : dict
        Dictionary returned by fit_demand_estimator().

    p : float or array-like
        Price(s) at which elasticity is to be evaluated.

    Returns
    -------
    float or ndarray
        Estimated elasticity evaluated at p.
    """

    model_type = model_info["model_type"]

    scalar_input = np.isscalar(p)

    p = np.atleast_1d(np.asarray(p, dtype=float))

    if np.any(p <= 0):
        raise ValueError("Prices must be positive.")

    # ----------------------------------------------------------
    # Linear demand
    # ----------------------------------------------------------

    def _vhat_linear():

        beta1 = model_info["coef"][0]
        d_hat = model_info["dhat_model"](p)

        # avoid divide-by-zero
        d_hat = np.maximum(d_hat, 1e-12)

        return -(p / d_hat) * beta1

    # ----------------------------------------------------------
    # Log-level demand
    # ----------------------------------------------------------

    def _vhat_log_level():
        beta1 = model_info["coef"][0]
        return -beta1 * p

    # ----------------------------------------------------------
    # Log-log demand
    # ----------------------------------------------------------

    def _vhat_log_log():
        beta1 = model_info["coef"][0]
        return np.full_like(p, -beta1, dtype=float)

    # ----------------------------------------------------------
    # Log-quadratic demand
    # ----------------------------------------------------------

    def _vhat_log_quadratic():

        beta1 = model_info["coef"][0]
        beta2 = model_info["coef"][1]

        return -(beta1 + 2.0 * beta2 * np.log(p))

    # ----------------------------------------------------------
    # Dispatch by model type
    # ----------------------------------------------------------

    if model_type == "linear":
        vhat = _vhat_linear()

    elif model_type == "log_level":
        vhat = _vhat_log_level()

    elif model_type == "log_log":
        vhat = _vhat_log_log()

    elif model_type == "log_quadratic":
        vhat = _vhat_log_quadratic()

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

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
    minimum_elasticity=1.0,
    lambda_optimality=0.25,
    boundary_penalty=0.25,
    boundary_buffer=0.02,
    Ngrid=500,
):
    """
    Fit multiple candidate demand models and select the model with the
    smallest combined selection score.

    For each candidate model:

    1. Fit the demand model using fit_demand_estimator().
    2. Compute the weighted log-RMSE between the observed demand and the
       fitted demand at the observed prices.
    3. Estimate the profit-maximizing price over the interval
       [price_min, price_max].
    4. Evaluate the elasticity at the estimated optimum and compare it
       with the inverse unit margin.
    5. Penalize solutions whose estimated optimum lies near either
       pricing boundary.
    6. Reject models whose elasticity at the estimated optimum is below
       minimum_elasticity.

    The model with the smallest selection score is returned.

    Parameters
    ----------
    p : array-like
        Observed prices.

    d : array-like
        Observed demand.

    candidate_model_types : list of str
        Candidate demand models, for example

            [
                "linear",
                "log_level",
                "log_log",
                "log_quadratic",
            ]

    price_min : float
        Minimum allowable price.

    price_max : float
        Maximum allowable price.

    c : float
        Unit variable cost.

    F : float, default=0.0
        Fixed cost used when estimating the profit-maximizing price.

    sample_weights : array-like or None, default=None
        Optional observation weights used when fitting the regression
        model and computing the weighted log-RMSE.

    ridge_cv : bool, default=False
        If True, fit each candidate model using RidgeCV.
        Otherwise use ordinary least-squares regression.

    minimum_elasticity : float, default=1.0
        Reject any model whose estimated elasticity at the estimated
        optimum is below this value.

    lambda_optimality : float, default=0.25
        Weight applied to the elasticity optimality gap

            abs(vhat_opt - inv_umar)

        when computing the selection score.

    boundary_penalty : float, default=0.25
        Additional penalty applied when the estimated optimum lies near
        either pricing boundary.

    boundary_buffer : float, default=0.02
        Fraction of the allowable price range that is treated as a
        boundary region.

    Ngrid : int, default=500
        Number of grid points used to estimate the profit-maximizing
        price.

    Returns
    -------
    dhat_model : callable
        Demand prediction function for the selected model.

    model_info : dict
        Model information returned by fit_demand_estimator() for the
        selected model.

    selected_model_type : str
        Name of the selected demand model.

    model_diagnostics : dict
        Dictionary containing diagnostics for every candidate model.
        For each model the following quantities are stored:

            fit_error
            optimality_gap
            boundary_penalty
            selection_score
            p_opt_hat
            d_opt_hat
            profit_opt_hat
            vhat_opt
            inv_umar
            at_boundary
            elasticity_valid
            selected_alpha
            error_message
    """

    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)

    p_grid = np.linspace(
        price_min,
        price_max,
        Ngrid,
    )

    # Ensure prices are above unit cost.
    p_grid = p_grid[p_grid > c]

    if len(p_grid) == 0:
        raise ValueError("price_max must be greater than unit cost c.")

    price_range = price_max - price_min

    boundary_distance = boundary_buffer * price_range

    fitted_models = {}
    model_diagnostics = {}

    for model_type in candidate_model_types:

        try:

            # --------------------------------------------------
            # Fit candidate model
            # --------------------------------------------------

            dhat_model_i, model_info_i = fit_demand_estimator(
                p=p,
                d=d,
                x=None,
                ridge_cv=ridge_cv,
                model_type=model_type,
                sample_weights=sample_weights,
            )

            # --------------------------------------------------
            # Weighted log-RMSE at observed prices
            # --------------------------------------------------

            dhat = np.asarray(
                dhat_model_i(p),
                dtype=float,
            )

            if np.any(~np.isfinite(dhat)) or np.any(dhat <= 0) or np.any(d <= 0):
                fit_error = np.inf

            else:
                residual = np.log(d) - np.log(dhat)

                if sample_weights is None:
                    # log-RMSE
                    fit_error = np.sqrt(np.mean(residual**2))

                else:
                    # Weighted log-RMSE
                    fit_error = np.sqrt(
                        np.average(
                            residual**2,
                            weights=sample_weights,
                        )
                    )

            # --------------------------------------------------
            # Estimate profit-maximizing price
            # --------------------------------------------------

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

            # np.where(condition, x,y) ... use x when condition true, y otherwise
            #  ... if profit_grid_hat is not valid results in -np.inf so it can never be selected
            profit_grid_valid = np.where(valid_grid, profit_grid_hat, -np.inf)

            i_opt = int(np.argmax(profit_grid_valid))
            p_opt_hat = float(p_grid[i_opt])
            d_opt_hat = float(d_grid_hat[i_opt])
            profit_opt_hat = float(profit_grid_hat[i_opt])

            # --------------------------------------------------
            # Elasticity optimality condition
            # --------------------------------------------------

            vhat_opt = float(
                np.asarray(
                    vhat_calculator(
                        model_info=model_info_i,
                        p=p_opt_hat,
                    )
                ).reshape(-1)[0]
            )

            inv_umar = p_opt_hat / (p_opt_hat - c)

            optimality_gap = abs(vhat_opt - inv_umar)

            # --------------------------------------------------
            # Boundary penalty
            # --------------------------------------------------

            at_boundary = (
                p_opt_hat <= price_min + boundary_distance
                or p_opt_hat >= price_max - boundary_distance
            )

            boundary_penalty_i = boundary_penalty if at_boundary else 0.0

            # --------------------------------------------------
            # Minimum elasticity check
            # --------------------------------------------------

            elasticity_valid = np.isfinite(vhat_opt) and vhat_opt >= minimum_elasticity

            # --------------------------------------------------
            # Combined selection score
            # --------------------------------------------------

            if np.isfinite(fit_error) and elasticity_valid:

                selection_score = (
                    fit_error + lambda_optimality * optimality_gap + boundary_penalty_i
                )

            else:
                selection_score = np.inf

            # RidgeCV-selected alpha
            selected_alpha = (
                float(model_info_i["model"].alpha_)
                if hasattr(
                    model_info_i["model"],
                    "alpha_",
                )
                else 0.0
            )

            fitted_models[model_type] = (
                dhat_model_i,
                model_info_i,
            )

            model_diagnostics[model_type] = {
                "fit_error": float(fit_error),
                "optimality_gap": float(optimality_gap),
                "boundary_penalty": float(boundary_penalty_i),
                "selection_score": float(selection_score),
                "p_opt_hat": float(p_opt_hat),
                "d_opt_hat": float(d_opt_hat),
                "profit_opt_hat": float(profit_opt_hat),
                "vhat_opt": float(vhat_opt),
                "inv_umar": float(inv_umar),
                "at_boundary": bool(at_boundary),
                "elasticity_valid": bool(elasticity_valid),
                "selected_alpha": float(selected_alpha),
                "error_message": None,
            }

        except Exception as exc:

            model_diagnostics[model_type] = {
                "fit_error": np.inf,
                "optimality_gap": np.inf,
                "boundary_penalty": np.inf,
                "selection_score": np.inf,
                "p_opt_hat": np.nan,
                "d_opt_hat": np.nan,
                "profit_opt_hat": np.nan,
                "vhat_opt": np.nan,
                "inv_umar": np.nan,
                "at_boundary": False,
                "elasticity_valid": False,
                "selected_alpha": np.nan,
                "error_message": str(exc),
            }

    # ------------------------------------------------------
    # Select model with smallest combined score
    # ------------------------------------------------------

    selected_model_type = min(
        model_diagnostics,
        key=lambda model_type: (model_diagnostics[model_type]["selection_score"]),
    )

    selected_score = model_diagnostics[selected_model_type]["selection_score"]

    if not np.isfinite(selected_score):
        error_messages = {
            model_type: diagnostics["error_message"]
            for model_type, diagnostics in model_diagnostics.items()
        }

        raise ValueError(
            "None of the candidate models could be selected. "
            f"Errors: {error_messages}"
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
    price_min, price_max, c, F, p_opt, demand_type, demand_params, sigma_log, Nloop=20
):
    """_summary_

    Args:
        price_min (_type_): _description_
        price_max (_type_): _description_
        c (_type_): variable demand
        F (_type_): Fixed price
        p_opt (_type_): theoretical optimimum price
        demand_type (_type_): _description_
        demand_params (_type_): _description_
        sigma_log (_type_): _description_
        Nloop (int, optional): _description_. Defaults to 20.

    Returns:
        dict: dictionary containing the selected demand model information, including the fitted model, model diagnostics, and the selected model type, including the following keys:
            - p_history
            - d_history
            - selected_model
            - model_fit_errors
            - p_opt_hat_history
            - p_opt_vhat_history
            - p_opt_hat_error_history
            - d_opt_hat_history
            - vhat_opt_history
            - inv_umar_history
            - profit_opt_history
    """

    # --------- Price Exploration Initialization --------
    price_range = price_max - price_min
    price_delta = 0.1 * price_range

    p1 = price_min
    p2 = price_min + 0.25 * price_range
    p3 = price_min + 0.50 * price_range
    p4 = price_min + 0.75 * price_range
    p5 = price_max

    Nprice_repeat = 2
    prices_initial = [
        p1,
        p2,
        p3,
        p4,
        p5,
    ]
    prices_initial = [p for p in prices_initial for _ in range(Nprice_repeat)]

    print(f"prices_initial = {prices_initial}")

    N_explore = 16  # explore up to N_explore, then explot p_opt_hat

    # Candidate models used by select_demand_model()
    candidate_model_types = ["linear", "log_level", "log_log", "log_quadratic"]

    # Grid used to maximize estimated profit
    p_grid = np.maximum(
        np.linspace(price_min, price_max, 500),
        c + 0.02,
    )

    # log_quadratic requires at least three unique prices
    min_fit_obs = 3

    # ----------------------------------------------------------
    # Histories
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Initial values used by the price-control logic
    # ----------------------------------------------------------

    pos_neg = 1
    p_opt_hat = np.nan
    vhat_opt_i = np.nan
    inv_umar_i = np.nan

    # ==========================================================
    # MAB loop
    # ==========================================================

    for i in range(Nloop):

        print(f"i = {i}", end=", ")

        # ------------------------------------------------------
        # Select the next price
        # ------------------------------------------------------

        if i < len(prices_initial):
            # Initial structured exploration
            pi = prices_initial[i]

        elif i <= N_explore:

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

        else:
            # Exploit the estimated profit-maximizing price
            pi = p_opt_hat

        pi = float(pi)

        # ------------------------------------------------------
        # Simulated observed demand
        # ------------------------------------------------------

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

        print(
            f"pi = {pi}, " f"di = {di}, " f"di_sigma0_i = {di_sigma0_i}",
            end=", ",
        )

        # ------------------------------------------------------
        # Update price and demand history
        # ------------------------------------------------------

        p_history.append(pi)
        d_history.append(di)

        # ------------------------------------------------------
        # Aggregate repeated prices
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Wait until enough unique prices are available
        # ------------------------------------------------------

        if len(p_fit) < min_fit_obs:

            print("selected_model = none, " "p_opt_hat = nan")

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

        # ------------------------------------------------------
        # Fit and select the demand model
        # ------------------------------------------------------

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
                    ridge_cv=False,
                )
            )

        except ValueError as exc:

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

        selected_model_error = model_diagnostics[selected_model_type]["fit_error"]

        # ------------------------------------------------------
        # Estimate demand and profit over the price grid
        # ------------------------------------------------------

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
                model_diagnostics[selected_model_type]["fit_errpr"]
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

        # ------------------------------------------------------
        # Estimated optimum
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Elasticity at the estimated optimum
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Simulation error diagnostic
        # ------------------------------------------------------
        # p_opt is the known true optimum in the simulation.
        # It is not used by model fitting or model selection.

        p_opt_hat_error = round(
            abs(p_opt_hat - p_opt),
            2,
        )

        # ------------------------------------------------------
        # Store histories
        # ------------------------------------------------------

        selected_model_history.append(selected_model_type)
        model_fit_error_history.append(
            model_diagnostics[selected_model_type]["fit_error"]
        )
        p_opt_hat_history.append(p_opt_hat)
        p_opt_vhat_history.append(p_opt_vhat_i)
        d_opt_hat_history.append(d_opt_hat)
        vhat_opt_history.append(vhat_opt_i)
        inv_umar_history.append(inv_umar_i)
        profit_opt_history.append(profit_opt_hat)
        p_opt_hat_error_history.append(p_opt_hat_error)

        # ------------------------------------------------------
        # Print iteration results
        # ------------------------------------------------------

        print(
            f"selected_model = {selected_model_type}",
            f", model_fit_error = {round(model_diagnostics[selected_model_type]["fit_error"],2)}",
            f", p_opt_hat = {p_opt_hat}",
            f", p_opt_vhat_i = {p_opt_vhat_i}",
            f", p_opt_hat_error = {p_opt_hat_error}",
            f", vhat_opt_i = {vhat_opt_i}",
            f", inv_umar_i = {inv_umar_i}",
            f", d_opt_hat = {d_opt_hat}",
            f", profit_opt_hat = {profit_opt_hat}",
        )

    results_dict = {
        "p_history": p_history,
        "d_history": d_history,
        "selected_model": selected_model_history,
        "selected_model_type": selected_model_type,
        "dhat_model": dhat_model,
        "model_info": model_info,
        "model_fit_errors": model_fit_error_history,
        "p_opt_hat_history": p_opt_hat_history,
        "p_opt_vhat_history": p_opt_vhat_history,
        "p_opt_hat_error_history": p_opt_hat_error_history,
        "d_opt_hat_history": d_opt_hat_history,
        "vhat_opt_history": vhat_opt_history,
        "inv_umar_history": inv_umar_history,
        "profit_opt_history": profit_opt_history,
    }

    return results_dict
