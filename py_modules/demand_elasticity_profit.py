"""demand_elasticity_profit methods

Public Functions:
- add_normal_demand_variation
- demand_constant_elasticity
- elasticity_constant_elasticity
- demand_linear
- elasticity_linear
- demand_logit
- elasticity_logit
- profit
- get_vhat_invumar
"""

import logging

import pandas as pd

import numpy as np

logger = logging.getLogger(__name__)

_CONST_ELASTICITY_PRICE_FLOOR = 1e-12
_LOGIT_EXP_CLIP = 60.0

# ####### Demand Variation ###########


# -------- add_normal_demand_variation --------#
def add_normal_demand_variation(demand: np.ndarray, sigma: float) -> np.ndarray:
    """Apply additive demand variation. This additive variation is applied to log D (log-space) to produce multiplicative variation ("demand shocks") in D. Demand is returned on a linear scale.

    Args:
        demand (np.ndarray): demand values (must be positive)
        sigma (float): std-dev of additive variation in log-space

    Returns:
        np.ndarray: demand with variation applied
    """
    if sigma == 0:
        return demand

    mask = demand > 0
    if not np.any(mask):
        return demand

    noise = np.random.normal(0, sigma, size=demand.shape)
    demand_noisy = np.zeros_like(demand, dtype=float)
    demand_noisy[mask] = np.exp(np.log(demand[mask]) + noise[mask])
    return demand_noisy


########### Constant Elasticity Demand ###########


# -------- _demand_constant_elasticity_raw --------#
def _demand_constant_elasticity_raw(p: np.ndarray, A: float, v: float) -> np.ndarray:
    """Compute constant elasticity demand (noiseless).

    Args:
        p (np.ndarray): price array
        A (float): scaling constant
        v (float): price elasticity of demand

    Returns:
        np.ndarray: demand
    """
    p_arr = np.asarray(p, dtype=float)
    p_safe = np.maximum(p_arr, _CONST_ELASTICITY_PRICE_FLOOR)
    floored = p_arr < _CONST_ELASTICITY_PRICE_FLOOR

    if np.any(floored):
        bad_prices = np.atleast_1d(p_arr)[floored]
        preview = np.array2string(bad_prices[:5], precision=6, separator=", ")
        if bad_prices.size > 5:
            preview = f"{preview} ..."
        logger.warning(
            "_demand_constant_elasticity_raw: applied price floor %.1e to %d value(s). prices=%s",
            _CONST_ELASTICITY_PRICE_FLOOR,
            bad_prices.size,
            preview,
        )

    mu_log = np.log(A) - v * np.log(p_safe)
    return np.exp(mu_log)


def _clip_logit_exponent(exponent: np.ndarray, context: str) -> np.ndarray:
    """Clip logit exponents to avoid overflow in exp and warn when clipping occurs."""
    exp_arr = np.asarray(exponent, dtype=float)
    exp_clipped = np.clip(exp_arr, -_LOGIT_EXP_CLIP, _LOGIT_EXP_CLIP)
    clipped = exp_clipped != exp_arr

    if np.any(clipped):
        bad_exponents = np.atleast_1d(exp_arr)[clipped]
        preview = np.array2string(bad_exponents[:5], precision=6, separator=", ")
        if bad_exponents.size > 5:
            preview = f"{preview} ..."
        logger.warning(
            "%s: clipped logit exponent to [%.1f, %.1f] for %d value(s). exponents=%s",
            context,
            -_LOGIT_EXP_CLIP,
            _LOGIT_EXP_CLIP,
            bad_exponents.size,
            preview,
        )

    return exp_clipped


# -------- elasticity_constant_elasticity --------#
def elasticity_constant_elasticity(p: np.ndarray, v: float) -> np.ndarray:
    """Compute elasticity for constant elasticity demand.

    Args:
        p (np.ndarray): price array
        v (float): price elasticity of demand

    Returns:
        np.ndarray: elasticity (constant, equal to v)
    """
    return np.full_like(p, v, dtype=float)


# -------- demand_constant_elasticity --------#
def demand_constant_elasticity(
    p: float, A: float, v: float, sigma_log: float = 0, size: int = 1
) -> np.ndarray:
    """Demand with multiplicative log-normal noise.

    Args:
        p (float): price or price vector
        A (float): scaling constant
        v (float): price elasticity of demand
        sigma_log (float): std-dev of additive noise in log-space
        size (int, optional): sample size when ``p`` is scalar

    Returns:
        np.ndarray: demand samples
    """
    p_arr = np.asarray(p, dtype=float)

    if np.any(p_arr <= 0):
        raise ValueError("Price must be positive")
    if A <= 0:
        raise ValueError("Scaling constant A must be positive")
    if v <= 0:
        raise ValueError("Price elasticity of demand must be positive")

    if np.ndim(p_arr) == 0:
        demand_at_price = float(_demand_constant_elasticity_raw(p_arr, A, v))
        demand = np.full(size, demand_at_price, dtype=float)
        return add_normal_demand_variation(demand, sigma_log)

    demand = _demand_constant_elasticity_raw(p_arr, A, v)
    return add_normal_demand_variation(demand, sigma_log)


########### Linear Demand ###########


# -------- _demand_linear_raw --------#
def _demand_linear_raw(p: np.ndarray, A: float, b: float) -> np.ndarray:
    """Compute linear demand (noiseless).

    Args:
        p (np.ndarray): price array
        A (float): demand multiplier constant
        b (float): slope of the demand curve

    Returns:
        np.ndarray: demand
    """
    raw_demand = A * (1 - b * p)
    return np.maximum(raw_demand, 0.0)


# -------- elasticity_linear --------#
def elasticity_linear(
    p: np.ndarray,
    A: float,
    b: float,
    regularization_mode: str | None = "regularized",
    regularization_eps: float = 1e-6,
) -> np.ndarray:
    """Compute elasticity for linear demand.

    Args:
        p (np.ndarray): price array
        A (float): demand multiplier constant
        b (float): slope of the demand curve
        regularization_mode (str | None):
            - "regularized" (default): one-sided regularization from feasible side (denominator floor)
            - None: return inf at singularity (1 - b*p ~= 0) and nan for p above choke price
        regularization_eps (float): regularization epsilon, denominator floor used in "regularized" mode

    Returns:
        np.ndarray: elasticity values

    Notes:
        - In "regularized" mode, regularization_eps acts as a denominator floor near
            the choke-price singularity and limits numerical blow-up.
        - Smaller regularization_eps yields larger elasticity near the boundary;
            larger regularization_eps gives stronger damping and more stability.
        - Use regularization_mode=None when you prefer explicit undefined values
            (inf at singularity, nan in infeasible region) instead of regularization.
    """
    _ = A  # Kept for API consistency with other elasticity_* functions.

    if regularization_mode not in {None, "regularized"}:
        raise ValueError("regularization_mode must be 'regularized' or None")
    if regularization_eps <= 0:
        raise ValueError("regularization_eps must be positive")

    p_arr = np.asarray(p, dtype=float)
    denom = 1 - b * p_arr
    near_zero = np.isclose(denom, 0.0, atol=1e-12, rtol=1e-9)
    infeasible = denom < 0

    flagged = near_zero | infeasible
    if np.any(flagged):
        bad_prices = np.atleast_1d(p_arr)[flagged]
        preview = np.array2string(bad_prices[:5], precision=6, separator=", ")
        if bad_prices.size > 5:
            preview = f"{preview} ..."
        logger.warning(
            "elasticity_linear: unstable/undefined at %d value(s) (regularization_mode=%s). prices=%s",
            bad_prices.size,
            regularization_mode,
            preview,
        )

    if regularization_mode is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            elasticity = b * p_arr / denom
        elasticity = np.where(near_zero, np.inf, elasticity)
        elasticity = np.where(infeasible, np.nan, elasticity)
        return elasticity

    # One-sided regularization only on feasible side; infeasible side stays undefined.
    elasticity = np.full_like(p_arr, np.nan, dtype=float)
    feasible = ~infeasible
    denom_safe = np.maximum(denom[feasible], regularization_eps)
    elasticity[feasible] = b * p_arr[feasible] / denom_safe

    return elasticity


# -------- demand_linear --------#
def demand_linear(p, A, b, sigma_log=0, size: int = 1):
    """Linear demand with optional log-normal noise.

    Computes the demand at the given price(s).

    d(p) = A * (1 - b * p)

    pmax = 1/b is the price at which demand is zero.

    Args:
        p (float): price or price vector
        A (float): demand multiplier constant. Market size at p = 0.
        b (float): slope of the demand curve
        sigma_log (float): std-dev of additive demand variation
        size (int, optional): sample size when ``p`` is scalar

    Returns:
        np.ndarray: demand samples
    """
    # Check for valid inputs
    if np.any(np.asarray(p) <= 0):
        raise ValueError("Price must be positive")
    if A <= 0:
        raise ValueError("Scaling constant A must be positive")
    if b <= 0:
        raise ValueError("Slope parameter b must be positive")

    p_arr = np.asarray(p, dtype=float)

    if np.ndim(p_arr) == 0:
        demand_at_price = float(_demand_linear_raw(p_arr, A, b))
        demand = np.full(size, demand_at_price, dtype=float)
        return add_normal_demand_variation(demand, sigma_log)

    demand = _demand_linear_raw(p_arr, A, b)
    return add_normal_demand_variation(demand, sigma_log)


########### Logit Demand ###########


# -------- _demand_logit_raw --------#
def _demand_logit_raw(p: np.ndarray, A: float, a: float, b: float) -> np.ndarray:
    """Compute logit demand (noiseless).

    Args:
        p (np.ndarray): price array
        A (float): scaling constant
        a (float): intercept parameter
        b (float): slope parameter

    Returns:
        np.ndarray: demand
    """
    exponent = _clip_logit_exponent(a - b * p, context="_demand_logit_raw")
    exp_term = np.exp(exponent)
    return A * exp_term / (1 + exp_term)


# -------- elasticity_logit --------#
def elasticity_logit(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Compute elasticity for logit demand.

    Args:
        p (np.ndarray): price array
        a (float): intercept parameter
        b (float): slope parameter

    Returns:
        np.ndarray: elasticity (positive magnitude)
    """
    exponent = _clip_logit_exponent(a - b * p, context="elasticity_logit")
    return b * p / (1 + np.exp(exponent))


# -------- demand_logit --------#
def demand_logit(p, A, a, b, sigma_log=0, size: int = 1):
    """Logit demand function with optional log-normal noise.

    Computes logit-based demand using exponential function.

    d(p) = A * exp(a - b * p) / (1 + exp(a - b * p))

    Args:
        p (float): price or price vector
        A (float): scaling constant
        a (float): intercept parameter
        b (float): slope parameter
        sigma_log (float): std-dev of additive demand variation
        size (int, optional): sample size when ``p`` is scalar

    Returns:
        np.ndarray: demand samples
    """

    # Check for valid inputs
    if np.any(np.asarray(p) <= 0):
        raise ValueError("Price must be positive")
    if A <= 0:
        raise ValueError("Scaling constant A must be positive")
    if b <= 0:
        raise ValueError("Slope parameter b must be positive")

    p_arr = np.asarray(p, dtype=float)

    if np.ndim(p_arr) == 0:
        demand_at_price = float(_demand_logit_raw(p_arr, A, a, b))
        demand = np.full(size, demand_at_price, dtype=float)
        return add_normal_demand_variation(demand, sigma_log)

    demand = _demand_logit_raw(p_arr, A, a, b)
    return add_normal_demand_variation(demand, sigma_log)


########### Profit ###########


# -------- profit --------#
def profit(p: np.ndarray, demand: np.ndarray, c: float, F: float) -> np.ndarray:
    """Compute profit given price, demand, and costs.

    Args:
        p (np.ndarray): price or price array
        demand (np.ndarray): demand values
        c (float): variable cost per unit
        F (float): fixed cost

    Returns:
        np.ndarray: profit = (p - c) * demand - F
    """
    return (p - c) * demand - F


########### Elasticity and Inverse Unit Margin ###########

# -------- profit --------#


def get_vhat_invumar_df(
    p: list | np.ndarray,
    demand_type: str,
    demand_params: dict[str:str],
    c: float,
    sigma_log=0,
) -> pd.DataFrame:
    """
    Generate a DataFrame containing estimated demand and inverse marginal revenue for a given set of prices.

    Parameters
    ----------
    p: list or np.ndarray
        List or 1-dimensional numpy array of prices. At least 2 prices are required to estimate elasticity.
    demand_type : str
        Type of demand function ("constant_elasticity", "linear", "logit").

    demand_params : dict[str:str]
        Parameters for the demand function.

    c : float
        Variable cost per unit, used to compute inverse unit margin.

    sigma_log : float, optional
        Standard deviation of the log-normal noise (default is 0).

    Nsamples : int, optional
        Number of samples to generate (default is 100).

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
          p_mid - midpoint price between consecutive prices,
          p - price (input),
          delta - price change between consecutive prices,
          p_delta - price + delta, the next price,
          d - demand at the given price,
          d_delta - demand at p_delta, i.e., demand between consecutive prices,
          vhat - estimated elasticity based on delta demand and price change,
          inv_umar - inverse unit margin p_mid/(p_mid-c) for the current price,
          diff_vhat_inv_umar - difference between vhat and inv_umar
          demand_type - type of demand function used to generate the data.
    """

    p = np.asarray(p, dtype=float)

    if p.ndim != 1:
        raise ValueError("p must be a one-dimensional array or list.")

    if len(p) < 2:
        raise ValueError("At least two prices are required to estimate elasticity.")

    required_params = {
        "constant_elasticity": {"A", "v"},
        "linear": {"A", "b"},
        "logit": {"A", "a", "b"},
    }

    if demand_type not in required_params:
        raise ValueError(
            f"Unknown demand_type '{demand_type}'. "
            f"Expected one of {list(required_params.keys())}."
        )

    missing = required_params[demand_type] - set(demand_params.keys())
    if missing:
        raise ValueError(
            f"demand_type='{demand_type}' requires "
            f"demand_params={required_params[demand_type]}, "
            f"but missing {missing}."
        )

    N = len(p)

    d = np.nan

    if demand_type == "constant_elasticity":
        d = demand_constant_elasticity(
            p,
            demand_params["A"],
            demand_params["v"],
            sigma_log=sigma_log,
            size=N,
        )

    elif demand_type == "linear":
        d = demand_linear(
            p,
            demand_params["A"],
            demand_params["b"],
            sigma_log=sigma_log,
            size=N,
        )

    elif demand_type == "logit":
        d = demand_logit(
            p,
            demand_params["A"],
            demand_params["a"],
            demand_params["b"],
            sigma_log=sigma_log,
            size=N,
        )

    _df = pd.DataFrame(
        {
            "p": p,
            "d": d,
        }
    )

    _df["p_delta"] = _df["p"].shift(-1)
    _df["d_delta"] = _df["d"].shift(-1)

    _df["delta"] = _df["p_delta"] - _df["p"]

    # Geometric mean in linear space (arithmetic mean in log space)
    _df["p_mid"] = np.sqrt(_df["p"] * _df["p_delta"])

    # Elasticity estimate using log differences
    _df["vhat"] = np.abs(
        (np.log(_df["d_delta"]) - np.log(_df["d"]))
        / (np.log(_df["p_delta"]) - np.log(_df["p"]))
    )

    # Inverse unit margin
    _df["inv_umar"] = _df["p_mid"] / (_df["p_mid"] - c)

    _df["diff_vhat_inv_umar"] = _df["vhat"] - _df["inv_umar"]

    _df["demand_type"] = demand_type

    cols = [
        "p_mid",
        "p",
        "delta",
        "p_delta",
        "d",
        "d_delta",
        "vhat",
        "inv_umar",
        "diff_vhat_inv_umar",
        "demand_type",
    ]

    return _df[cols]
