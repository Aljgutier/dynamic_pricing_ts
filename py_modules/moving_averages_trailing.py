"""
Trailing Moving Averages
 -  trailing_exponential_moving_average
"""

# ----------- exponential trailing moving average --------------#


def trailing_exponential_moving_average(x, alpha=0.3):
    """
    Compute the trailing exponential moving average.

    Parameters
    ----------
    x : sequence of numbers
        Input data.
    alpha : float
        Smoothing parameter, where 0 < alpha <= 1.

    Returns
    -------
    list of float
        Exponential moving average with the same length as x.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")

    if len(x) == 0:
        return []

    y = [x[0]]

    for i in range(1, len(x)):
        y.append(alpha * x[i] + (1 - alpha) * y[i - 1])

    return y


# ------------- trailing moving average ------------------#


def trailing_moving_average(x, Nwindow=5):
    """
    Compute the trailing moving average.

    Parameters
    ----------
    x : sequence of numbers
        Input data.
    Nwindow : int, optional
        Trailing window size (default 5).

    Returns
    -------
    list of float
        Trailing moving average with the same length as x.
        For the first Nwindow-1 elements, the average is
        computed using all available preceding values.
    """
    if Nwindow < 1:
        raise ValueError("Nwindow must be >= 1")

    y = []
    for i in range(len(x)):
        start = max(0, i - Nwindow + 1)
        y.append(sum(x[start : i + 1]) / (i - start + 1))

    return y
