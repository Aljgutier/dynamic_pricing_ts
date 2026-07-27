"""Probability Convenience functions for PyMC TS
- truncated_normal_pdf
- half_normal_pdf
- half_normal_
- log_normal_pdf
- lognormal_pdf_from_linear
- truncated normal_pdf
- truncated_normal_mean
- demand_lognormal_pdf
- demand_sample
"""

import math
import numpy as np
from scipy.stats import truncnorm, norm


###############################
##### truncated normal pdf #####
################################
def truncated_normal_pdf(x, m, sigma, lower_limit, upper_limit):
    """
    Truncated normal probability distribution. Receives x, mean, and sigma (standard deviatin), lower_limit, upper_limit, and returns the corresponding probability for point(s) x.

    This truncated normal PDF uses underlying Normal parameters (m, sigma). Matches PyMC pm.TruncatedNormal(mu=m, sigma=sigma, lower=lower_limit).

    Parameters:
        x : float or arraylike
           Points to evaluate the corresponding truncated probability.  Must satisfy x > lower_limit and x < upper_limit, otherwise returns 0.

        m : location offset, the mean of the underlying (non truncated normal)


        sigma: mumeric
            standard deviation

        lower_limit : numeric
            truncated_normal lower_limit

        upper_limit : numeric

    """
    # rescale for linear scale to agree with trunc_normal limit
    #  ... pyMC lower_limit and SciPy lower limit agreement
    x = np.asarray(x)

    # standardize truncation bounds (CRITICAL)
    a = (lower_limit - m) / sigma
    b = (upper_limit - m) / sigma

    pdf = np.zeros_like(x, dtype=float)

    mask = (x >= lower_limit) & (x <= upper_limit)
    pdf[mask] = truncnorm.pdf(x[mask], a=a, b=b, loc=m, scale=sigma)

    return pdf


####################################
#####   truncated_normal_mean   #####
####################################
def truncated_normal_mean(mu, sigma, a):
    """
    Compute the mean of a lower-truncated normal distribution.

    Let X be a normal random variable with mean mu and standard deviation sigma.
    This function returns the conditional mean

        E[X | X >= a]

    where the distribution is truncated below at the point a.

    The formula used is:

        mu_tn = mu + sigma * phi(alpha) / (1 - Phi(alpha))

    where

        alpha = (a - mu) / sigma

    Lower case phi denotes the probability density function of the standard
    normal distribution, and Phi (upper case) denotes the cumulative distribution function
    of the standard normal distribution.

    Parameters
    ----------
    mu : float
        Mean of the original (untruncated) normal distribution.
    sigma : float
        Standard deviation of the original normal distribution. Must be positive.
    a : float
        Lower truncation point.

    Returns
    -------
    float
        Mean of the lower-truncated normal distribution.

    Notes
    -----
    The quantity phi(alpha) divided by (1 minus Phi(alpha)) is known as the
    inverse Mills ratio. The truncated mean is always greater than or equal
    to a, and greater than mu when a is greater than negative infinity.

    Examples
    --------
    >>> truncated_normal_mean(0.0, 1.0, 0.0)
    0.7978845608028654
    """

    alpha = (a - mu) / sigma
    return mu + sigma * norm.pdf(alpha) / (1 - norm.cdf(alpha))


###############################
#####   half_normal_pdf   #####
################################
def half_normal_pdf(x, sigma):
    """
    Half-Normal distribution PDF with scale sigma. Note the mean of
    Half Normal is not an input, it is implied and given by sigma * sqrt(2/pi). This definition matches the HalfNormal PyMC definition.

    Note, do not shift this or will turn it into a truncated normal.

    Args:
        x (float or array): Evaluation point(s), x >= 0
        sigma (float): Scale parameter (same as PyMC HalfNormal)

    Returns:
        float or array: pdf value(s)
    """
    x = np.asarray(x)

    density_x = np.where(
        x >= 0, np.sqrt(2 / np.pi) / sigma * np.exp(-(x**2) / (2 * sigma**2)), 0.0
    )

    return density_x


####################################
#####   half_normal_mean   #####
####################################
def half_normal_mean(mu, sigma, shifted=False):
    """
    Compute the mean of a half-normal distribution derived from
    a normal distribution with parameters (mu, sigma).

    Parameters
    ----------
    mu : float
        Mean of the original normal distribution.
    sigma : float
        Standard deviation of the original normal distribution (must be > 0).
    shifted : bool, optional (default=False)
        If False, returns E[|X - mu|], i.e. a standard half-normal.
        If True, returns E[mu + |X - mu|], i.e. a shifted half-normal.

    Returns
    -------
    float
        Mean of the half-normal distribution.
    """
    mean_half = sigma * math.sqrt(2.0 / math.pi)
    return mu + mean_half if shifted else mean_half


###############################
#####   log_normal_pdf   #####
################################
def lognormal_pdf(x, m, sigma):
    """
    Compute the probability density (PDF) of a log-normal distribution. lnX ~ N(m, sigma^2)

    Note on LogNormal
    log(x) is Normal, therefore X is logNormal. LogNormal is the distribution over x (not log x ). Note that logNormal includes 1/x, Jacobiann normalization factor.

    Note, this matches the PyMc lognoormal function which accepts the log-space parameters inputs.


    Args
    ----
    x (float, array-like) :  Evaluation point(s). Must satisfy x > 0. Values <= 0 return 0.

    m (float): Mean (mu) of the underlying normal distribution lnX.

    sigma (float):  Standard deviation of the underlying normal distribution lnX.

    Returns
    -------
    float or np.darray :  Log-normal PDF evaluated at x.

    Notes
    -----
    The log-normal PDF is defined as:
        f(x) = (1 / (x * sigma * sqrt(2π)))
               * exp( - (ln(x) - m)^2 / (2 * sigma^2) ),  for x > 0

    For numerical safety, the implementation explicitly enforces the support x > 0 using a boolean mask, avoiding log(x) for invalid values.
    """
    x = np.asarray(x)
    pdf = np.zeros_like(x, dtype=float)

    mask = x > 0
    pdf[mask] = (1.0 / (x[mask] * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((np.log(x[mask]) - m) / sigma) ** 2
    )

    return pdf


########################################
#####  lognormal_from_linear_pdf   #####
########################################


def lognormal_pdf_from_linear(x, mean, cv):
    """
    LogNormal PDF parameterized by linear-space mean and CV.

    Args:
        x (array-like): linear-space evaluation points
        mean (_type_): linear-space mean
        cv (_type_): coefficient of variation (std/mean) in linear space

    Returns:
       np.array: log-normal PDF evaluated at x
    """

    sigma_log = np.sqrt(np.log1p(cv**2))
    mu_log = np.log(mean) - 0.5 * sigma_log**2
    return lognormal_pdf(x, mu_log, sigma_log)


####################################
#####   demand_lognormal_pdf   #####
####################################
def demand_lognormal_pdf(d, a, v, p, CV):
    """
    Computes the LogNormal density of demand under a constant-elasticity
    demand model with multiplicative noise.

    Model:
        log(D) is Normal, therefore D is logNormal. LogNormal is the distribution over D . Note that logNormal includes 1/d, Jacobiann normalization factor.

    This function returns the density in linear space:
        f_D(d)

    Args:
        d (array-like): Demand values (D > 0) at which to evaluate the density.
        a (float): Demand scale parameter.
        v (float): Price elasticity of demand.
        p (float): Price at which demand is evaluated.
        CV (float): Coeficient of Variation, given be
        std/mean (linear-space). This variable is transformed
        into sigma_log = sqrt(ln(1 + CV^2))

    Returns:
        array-like: LogNormal density values of demand evaluated at d.
    """

    sigma_log = np.sqrt(np.log1p(CV**2))  # sqrt(ln( 1 + CV^2 ))

    mu_log = np.log(a) - v * np.log(p)  # this is mu_i = ln (a p^ -v)
    d_log = np.log(d)

    density = (1 / (d * np.sqrt(2 * np.pi) * sigma_log)) * np.exp(
        -0.5 * ((d_log - mu_log) / sigma_log) ** 2
    )
    return density


def demand_sample(
    p: float, a: float, v: float, sigma_log: float = 0, size: int = 1
) -> np.ndarray:
    """Demand with multiplicative noise.

    Args:
        p (float): price
        a (float): demand multiplier constant
        v (float): price elasticity of demand
        sigma_log (float): standard deviation of the log-normal distribution
        size (int, optional): Then number of samples to return. Defaults to 1.

    Returns:
        float: linear demand sample
    """

    mu_log = np.log(a) - v * np.log(p)
    
    if sigma_log == 0:
        return np.exp(mu_log) * np.ones(size)
    else:
        return np.exp(mu_log + np.random.normal(0, sigma_log, size=size))
